import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256


REQUESTS = ['whole1', 'whole_half', 'H1', 'H2', 'union', 'complement']
METHODS = ['source1', 'source2', 'graph', 'PW_selected', 'raw']


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    base = load_config(config['base_config']) if config.get('base_config') else {}
    base.update(config)
    return base


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def request_members(masks, request, width):
    whole = np.asarray(masks['whole'], dtype=np.int64)
    if request in ['whole1', 'whole_half']:
        members = whole
    elif request in ['H1', 'H2']:
        members = np.asarray(masks[request], dtype=np.int64)
    elif request == 'union':
        members = np.union1d(masks['H1'], masks['H2']).astype(np.int64)
    elif request == 'complement':
        members = np.setdiff1d(np.arange(width), whole)
    else:
        raise ValueError(request)
    assert len(np.unique(members)) == len(members)
    assert np.all((members >= 0) & (members < width))
    return members, 0.5 if request == 'whole_half' else 1.0


def response_statistics(torch, logits, clean_logits, source_logits, next_tokens, top_count):
    clean = clean_logits.double()
    current = logits.double()
    reference = source_logits.double()
    logp0, logp, source_logp = clean.log_softmax(-1), current.log_softmax(-1), reference.log_softmax(-1)
    p0 = logp0.exp()
    effect, source_effect = current-clean, reference-clean
    effect -= (p0*effect).sum(-1, keepdim=True)
    source_effect -= (p0*source_effect).sum(-1, keepdim=True)
    squared_error = (p0*(effect-source_effect).square()).sum(-1)
    source_energy = (p0*source_effect.square()).sum(-1)
    ids = effect.abs().topk(top_count, dim=-1).indices
    rows = torch.arange(len(logits), device=logits.device)
    values = dict(weighted_response_error=squared_error, source_response_energy=source_energy,
        response_energy=(p0*effect.square()).sum(-1), next_token_logprob=logp[rows, next_tokens],
        clean_next_token_logprob=logp0[rows, next_tokens], source_next_token_logprob=source_logp[rows, next_tokens],
        argmax=current.argmax(-1), clean_argmax=clean.argmax(-1), source_argmax=reference.argmax(-1),
        kl_clean_to_operation=(p0*(logp0-logp)).sum(-1),
        kl_source_to_operation=(source_logp.exp()*(source_logp-logp)).sum(-1),
        top_change_token_ids=ids, top_change_centered_logits=effect.gather(-1, ids),
        top_change_clean_probability=p0.gather(-1, ids), top_change_logprob=logp.gather(-1, ids))
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    work = MultisiteWork(config, args.config, ['scripts/run_operation_groups.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py', 'src/ccad/activation_contract.py'])
    error = None
    try:
        import torch
        import transformers
        sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
        sys.path.insert(0, config.get('scipy_overlay_dir', 'D:/CCAD_Storage/environments/f4_sparse_overlay_v1'))
        from scipy import sparse
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
        work.torch, work.device = torch, torch.device(config['device'])
        torch.set_num_threads(config.get('cpu_threads', 2))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision(config.get('matmul_precision', 'high'))
        if work.device.type == 'cuda':
            torch.cuda.set_device(work.device)
            torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            device=str(work.device), cpu_threads=torch.get_num_threads(),
            matmul_precision=torch.get_float32_matmul_precision(),
            gpu=torch.cuda.get_device_name(work.device) if work.device.type == 'cuda' else 'not_used')
        write(work.run/'environment.json', work.environment)
        panel_path = work.checked(config['panel_path'], 'Frozen natural token positions')
        panel = json.loads(panel_path.read_text(encoding='utf-8'))
        rows = panel['rows']
        assert rows and all(len(row['tokens']) == 128 for row in rows)
        for index, row in enumerate(rows):
            assert 0 <= row['position'] < len(row['tokens'])
            assert row['next_token_id'] is not None
            if row['position']+1 < len(row['tokens']):
                assert row['next_token_id'] == row['tokens'][row['position']+1]
            else:
                stream_path = row.get('token_path', panel.get('token_path'))
                assert stream_path is not None and 'packed_position' in row
                stream = np.memmap(stream_path, mode='r', dtype='<u2')
                packed = row['packed_position']
                assert packed % 128 == row['position'] and packed+1 < len(stream)
                np.testing.assert_array_equal(stream[packed-127:packed+1], row['tokens'])
                assert int(stream[packed+1]) == row['next_token_id']
                del stream
        write(work.run/'panel.json', panel)
        model_directory = Path(config['model_local_dir'])
        for name in ['config.json', 'model.safetensors', 'tokenizer.json']:
            work.checked(model_directory/name, 'Pinned GPT2-medium '+config['model_revision'])
        for name in ['top_k.py', 'matryoshka_batch_top_k.py']:
            work.checked(Path(config['dictionary_source_dir'])/'dictionary_learning/trainers'/name,
                         'Original SAE encoder implementation', 'MIT')
        model = transformers.AutoModelForCausalLM.from_pretrained(model_directory, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        assert model.config.model_type == 'gpt2' and model.config.hidden_size == 1024
        hook_name = config.get('hook_module_path', 'transformer.h.11')
        assert hook_name == 'transformer.h.11'
        contract = HookPointContract(hook_name, 11, 'resid_post', 1024)
        module = model.get_submodule(hook_name)
        batch_size = int(config['batch_size'])
        target_seed = int(config.get('target_seed', 3))
        maps, checkpoints, checkpoint_hashes, ridge_hashes = {}, {}, {}, {}
        for mechanism in config['mechanisms']:
            path = work.checked(Path(config['target_map_run'])/(mechanism+'_target_group.json'))
            maps[mechanism] = json.loads(path.read_text())
            assert maps[mechanism]['target_seed'] == target_seed
            ridge_path = work.checked(maps[mechanism]['raw_ridge_path'], 'Frozen signed source1 readout')
            ridge_hashes[mechanism] = sha256(ridge_path)
            for method in ['source1', 'source2', 'graph', 'PW_selected']:
                assert set(['whole', 'H1', 'H2']) <= set(maps[mechanism]['masks'][method])
            for seed in [1, 2, target_seed]:
                checkpoint = work.checked(Path(config['checkpoint_directory'])/f'{mechanism}_seed{seed}.pt')
                checkpoints[mechanism, seed] = checkpoint
                checkpoint_hashes[f'{mechanism}_seed{seed}'] = sha256(checkpoint)
        frozen_identity = identity(dict(panel=sha256(panel_path), maps=maps,
            checkpoints=checkpoint_hashes, raw_readouts=ridge_hashes,
            generator=sha256(Path(__file__)), model_revision=config['model_revision'], hook=hook_name,
            precision=torch.get_float32_matmul_precision(), batch_size=batch_size,
            member_limit=config.get('member_limit'),
            requests=REQUESTS, source_code='signed sparse discovery readout for raw; native codes for masks'))
        previous = {}
        if config.get('resume_run'):
            saved = json.loads(work.checked(Path(config['resume_run'])/'response_index.json').read_text())
            assert saved['identity'] == frozen_identity
            previous = {item['key']: item for item in saved['chunks']}
        index = dict(identity=frozen_identity, checkpoint_hashes=checkpoint_hashes,
            requests=REQUESTS, methods=METHODS, chunks=[],
            primary='Clean-probability weighted centered complete-vocabulary logit effect; each request uses its own source1 response energy',
            state_definition='GPT2 transformer output after ln_f at the same specified intervention position',
            top_change_definition='Twenty largest absolute clean-probability-centered logit changes; signed values retained',
            panel_path=str((work.run/'panel.json').resolve()))
        write(work.run/'response_index.json', index)
        chunk_dir = work.run/'chunks'
        chunk_dir.mkdir()

        def forward(indices, delta=None, full_model=False):
            if time.perf_counter()-work.wall_start > config['budget_seconds']:
                raise TimeoutError('Declared operation consumer budget exhausted')
            tokens = torch.tensor([rows[i]['tokens'] for i in indices], dtype=torch.long, device=work.device)
            positions = torch.tensor([rows[i]['position'] for i in indices], device=work.device)
            batch = torch.arange(len(indices), device=work.device)
            captured = {}
            def hook(_module, _args, output):
                hidden = extract_primary_hook_tensor(output, contract)
                captured['hook_hidden'] = hidden[batch, positions].detach()
                if delta is None:
                    return output
                changed = hidden.clone()
                changed[batch, positions] += torch.as_tensor(delta, device=work.device, dtype=hidden.dtype)
                return replace_primary_hook_tensor(output, changed, contract)
            def final_hook(_module, _args, output):
                captured['ln_f'] = output[batch, positions].detach()
            handle = module.register_forward_hook(hook)
            final_handle = model.transformer.ln_f.register_forward_hook(final_hook)
            try:
                with torch.no_grad():
                    if full_model:
                        output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
                        captured['full_logits'] = output.logits[batch, positions].detach()
                    else:
                        model.transformer(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
            finally:
                handle.remove()
                final_handle.remove()
            work.sequence_forwards += len(indices)
            work.token_forwards += int(tokens.numel())
            return captured

        def materialize(key, fields, compute):
            if key in previous:
                item = previous[key]
                path = Path(item['path'])
                assert sha256(path) == item['sha256']
                with np.load(path) as saved:
                    assert str(saved['identity']) == frozen_identity and str(saved['status']) == 'PASS'
                    arrays = {name: saved[name] for name in saved.files}
                item = dict(item, reused_from=config['resume_run'])
            else:
                arrays = compute()
                assert all(np.isfinite(value).all() for value in arrays.values()
                           if np.issubdtype(np.asarray(value).dtype, np.number))
                arrays.update(identity=np.array(frozen_identity), status=np.array('PASS'))
                path = chunk_dir/(key+'.npz')
                temporary = path.with_suffix('.partial.npz')
                np.savez_compressed(temporary, **arrays)
                temporary.replace(path)
                item = dict(key=key, path=str(path.resolve()), sha256=sha256(path), **fields)
            index['chunks'].append(item)
            write(work.run/'response_index.json', index)
            work.record(kind='operation_chunk', task=fields['mechanism'], row_id=fields['start'],
                mode=fields['kind'], method=fields['method'], operation=fields['operation'],
                target_seed=target_seed, component=key, path=str(path.resolve()),
                positions=fields['count'], reused=key in previous)
            work.progress('OPERATION_CHUNK', key=key, completed_chunks=len(index['chunks']),
                          sequence_forwards_actual=work.sequence_forwards)
            return arrays

        # baseline 在全部方法之间共用，每个 chunk 保存真实 hook 与最终状态。
        baseline = np.empty((len(rows), 1024), np.float32)
        hidden = np.empty_like(baseline)
        for start in range(0, len(rows), batch_size):
            selected = np.arange(start, min(start+batch_size, len(rows)))
            def capture(selected=selected):
                values = forward(selected)
                return dict(row_indices=selected, ln_f=values['ln_f'].cpu().numpy(),
                            hook_hidden=values['hook_hidden'].cpu().numpy())
            values = materialize(f'baseline_{start:06d}', dict(mechanism='common', method='baseline',
                operation='none', kind='baseline', start=start, count=len(selected)), capture)
            np.testing.assert_array_equal(values['row_indices'], selected)
            baseline[selected], hidden[selected] = values['ln_f'], values['hook_hidden']
        probe = np.arange(min(batch_size, len(rows)))
        replay = forward(probe, np.zeros_like(hidden[probe]), full_model=True)
        np.testing.assert_array_equal(replay['ln_f'].cpu().numpy(), baseline[probe])
        np.testing.assert_array_equal(replay['hook_hidden'].cpu().numpy(), hidden[probe])
        with torch.no_grad():
            projected = model.lm_head(replay['ln_f'])
        write(work.run/'replay_check.json', dict(noop_ln_f_exact=True, noop_hook_exact=True,
            selected_head_vs_full_logits_max_error=float((projected-replay['full_logits']).abs().max()),
            ln_f_state_sufficient=True, positions=probe.tolist()))
        work.checks.update(noop_ln_f_exact=True, noop_hook_exact=True, causal_attention_includes_eos=True)
        del replay, projected
        for mechanism in config['mechanisms']:
            group = maps[mechanism]
            selected_rows = np.array([i for i, row in enumerate(rows)
                                     if row.get('mechanism', mechanism) == mechanism], dtype=np.int64)
            assert len(selected_rows)
            saes, decoders, codes = {}, {}, {}
            for seed in [1, 2, target_seed]:
                state = torch.load(checkpoints[mechanism, seed], map_location=work.device, weights_only=True)
                assert int(state['k']) == 64
                sae = (AutoEncoderTopK(1024, 8192, 64) if mechanism == 'topk' else
                    MatryoshkaBatchTopKSAE(1024, 8192, 64, state['group_sizes'].cpu().tolist()))
                sae = sae.to(work.device)
                sae.load_state_dict(state)
                sae.eval().requires_grad_(False)
                saes[seed] = sae
                decoders[seed] = sae.decoder.weight.T if mechanism == 'topk' else sae.W_dec
                with torch.no_grad():
                    codes[seed] = sae.encode(torch.tensor(hidden[selected_rows], device=work.device))
                del state
            ridge_path = Path(group['raw_ridge_path'])
            with np.load(ridge_path) as fit:
                source_ids, target_ids = fit['source_ids'], fit['target_ids']
                np.testing.assert_array_equal(source_ids, np.arange(8192))
                np.testing.assert_array_equal(target_ids, np.arange(8192))
                valid = fit['c_st'] >= 0
                rr = np.broadcast_to(np.arange(8192)[:, None], fit['c_st'].shape)[valid]
                relation = sparse.csr_matrix((fit['r_st'][valid], (rr, fit['c_st'][valid])), shape=(8192, 8192))
            predicted = torch.tensor(np.asarray(relation@codes[target_seed].cpu().numpy().T).T,
                                     device=work.device, dtype=torch.float32)
            jobs = []
            for method in METHODS:
                masks = group['masks']['source1' if method == 'raw' else method]
                for request in REQUESTS:
                    members, fraction = request_members(masks, request, 8192)
                    jobs.append(dict(method=method, request=request, members=members,
                                     fraction=fraction, reference='source1__'+request, kind='group'))
            source_members = np.asarray(group['masks']['source1']['whole'], dtype=np.int64)
            matched = np.asarray(group['PW_source1_members'], dtype=np.int64)
            assert len(source_members) == len(matched) and len(np.unique(matched)) == len(matched)
            limit = config.get('member_limit', len(source_members))
            for source_member, target_member in zip(source_members[:limit], matched[:limit]):
                request = f'member_{source_member}'
                for method, member in [('source1', source_member), ('PW_source1', target_member)]:
                    jobs.append(dict(method=method, request=request, members=np.array([member]), fraction=1.0,
                                     reference='source1__'+request, kind='member'))
            references = {}
            for job in jobs:
                method, request = job['method'], job['request']
                seed = 1 if method in ['source1', 'raw'] else 2 if method == 'source2' else target_seed
                member_ids = torch.tensor(job['members'], device=work.device)
                with torch.no_grad():
                    z = predicted if method == 'raw' else codes[seed]
                    delta = -job['fraction']*(z[:, member_ids]@decoders[seed][member_ids])
                own_reference = method == 'source1'
                result_states = np.empty((len(selected_rows), 1024), np.float32)
                for offset in range(0, len(selected_rows), batch_size):
                    stop = min(offset+batch_size, len(selected_rows))
                    indices = selected_rows[offset:stop]
                    def measure(offset=offset, stop=stop, indices=indices):
                        actual = forward(indices, delta[offset:stop])
                        final = actual['ln_f']
                        reference = final if own_reference else torch.tensor(
                            references[job['reference']][offset:stop], device=work.device)
                        with torch.no_grad():
                            logits = model.lm_head(final)
                            clean_logits = model.lm_head(torch.tensor(baseline[indices], device=work.device))
                            source_logits = model.lm_head(reference)
                            statistics = response_statistics(torch, logits, clean_logits, source_logits,
                                torch.tensor([rows[i]['next_token_id'] for i in indices], device=work.device), 20)
                        return dict(row_indices=indices, ln_f=final.cpu().numpy(),
                            edit_norm=delta[offset:stop].norm(dim=-1).cpu().numpy(),
                            member_ids=job['members'], fraction=np.array(job['fraction']), **statistics)
                    key = f'{mechanism}__{method}__{request}__{offset:06d}'
                    values = materialize(key, dict(mechanism=mechanism, method=method, operation=request,
                        kind=job['kind'], start=offset, count=len(indices), reference=job['reference']), measure)
                    np.testing.assert_array_equal(values['row_indices'], indices)
                    result_states[offset:stop] = values['ln_f']
                if own_reference:
                    references['source1__'+request] = result_states
            del saes, decoders, codes, predicted, relation, references, delta
        work.checks.update(complete_frozen_requests=True, source1_member_references=True,
            target_natural_codes=True, raw_signed_readout=True, full_vocabulary_primary=True)
        write(work.run/'response_index.json', index)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
