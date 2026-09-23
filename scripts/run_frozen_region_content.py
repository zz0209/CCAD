from pathlib import Path
import argparse
import gc
import json
import platform
import sys
import time
import traceback

import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, ROOT, write
from train_grammar_material_support import load_config
from ccad.artifacts import sha256


METHODS = ['region6', 'region2', 'region4', 'union6',
           'union_minus_region6', 'cached64_matched', 'raw']
ANSWERS = [' himself', ' herself', ' themselves']


def region_members(structure, proposal, objective, target):
    rows = [r for r in structure['rows'] if r['objective'] == objective
            and r['target'] == target and r['family'] == 'shared_path_full']
    def members(kind, operation):
        matches = [r for r in rows if r['kind'] == kind and str(r['operation']) == operation]
        assert len(matches) == 1, (objective, target, kind, operation)
        return np.asarray(sorted(matches[0]['members']), dtype=np.int64)
    result = {f'region{k}': members('region', str(k)) for k in [6, 2, 4]}
    result['union6'] = members('union', '6')
    result['union_minus_region6'] = np.setdiff1d(result['union6'], result['region6'])
    with np.load(proposal, allow_pickle=False) as saved:
        score = saved['cached64_full_score']
    assert score.shape == (8192, 3) and np.isfinite(score).all()
    order = np.lexsort((np.arange(len(score)), -np.minimum(score[:, 1], score[:, 2])))
    result['cached64_matched'] = np.sort(order[:len(result['region6'])]).astype(np.int64)
    result['raw'] = np.empty(0, dtype=np.int64)
    for ids in result.values():
        assert len(ids) == len(np.unique(ids)) and np.all((ids >= 0) & (ids < 8192))
    return result


def execute(work):
    cfg = work.cfg
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    work.torch, work.device = torch, torch.device(cfg['device'])
    torch.cuda.set_device(work.device)
    torch.cuda.reset_peak_memory_stats(work.device)
    work.environment = dict(python=sys.executable, python_version=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, transformers=transformers.__version__,
        device=str(work.device), gpu=torch.cuda.get_device_name(work.device),
        cpu_threads=torch.get_num_threads(), dtype='float32', matmul_precision='highest',
        attention='eager', hook='transformer.h.11', model_layers=24)
    panel_path = work.checked(ROOT / cfg['panel'])
    assert sha256(panel_path) == cfg['panel_sha256']
    panel = json.loads(panel_path.read_text(encoding='utf-8'))
    rows = panel['rows']
    selected = [i for i, r in enumerate(rows) if r['split'] == cfg['split']]
    if cfg.get('lexical_limit'):
        lexical = list(dict.fromkeys(rows[i]['lexical_id'] for i in selected))[:cfg['lexical_limit']]
        selected = [i for i in selected if rows[i]['lexical_id'] in lexical]
    selected = np.asarray(selected, dtype=np.int64)
    assert len(selected) > 0
    donors = np.asarray([[rows[i]['donor_first'], rows[i]['donor_second']] for i in selected], dtype=np.int64)
    assert np.all((donors >= 0) & (donors < len(rows)))
    for i, pair in zip(selected, donors):
        for donor in pair:
            assert rows[i]['split'] == rows[donor]['split']
            assert rows[i]['lexical_id'] == rows[donor]['lexical_id']
    needed = np.unique(np.concatenate([selected, donors.ravel()]))
    local = {int(r): i for i, r in enumerate(needed)}
    recipient_local = np.asarray([local[int(i)] for i in selected])
    donor_local = np.asarray([[local[int(i)] for i in pair] for pair in donors])
    training = ROOT / cfg['training_run']
    training_cfg = json.loads(work.checked(training / 'config.resolved.json').read_text())
    snapshots = json.loads(work.checked(training / 'checkpoints.json').read_text())['checkpoints']
    assert training_cfg['hook_module_path'] == 'transformer.h.11'
    sys.path.extend([training_cfg['dictionary_source_dir'], training_cfg['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
    for name in ['dictionary_learning/trainers/top_k.py',
                 'dictionary_learning/trainers/matryoshka_batch_top_k.py', 'LICENSE']:
        work.checked(Path(training_cfg['dictionary_source_dir']) / name)
    model_path = Path(training_cfg['model_local_dir'])
    for name in ['config.json', 'tokenizer.json', 'model.safetensors']:
        work.checked(model_path / name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    answer_ids = [tokenizer.encode(answer, add_special_tokens=False) for answer in ANSWERS]
    assert all(len(ids) == 1 for ids in answer_ids)
    answer_ids = np.asarray([ids[0] for ids in answer_ids], dtype=np.int64)
    tokens = {}
    for i in needed:
        plain = tokenizer.encode(rows[i]['text'], add_special_tokens=False)
        for answer, aid in zip(ANSWERS, answer_ids):
            assert tokenizer.encode(rows[i]['text'] + answer, add_special_tokens=False) == plain + [int(aid)]
        tokens[int(i)] = ([tokenizer.eos_token_id] if cfg['add_bos'] else []) + plain
    max_length = max(map(len, tokens.values()))
    assert max_length <= cfg['max_length']
    write(work.run / 'panel.json', dict(rows=[dict(rows[i], original_index=int(i), tokens=tokens[int(i)])
        for i in needed], selected_row_indices=selected.tolist(), donor_indices=donors.tolist(),
        answer_strings=ANSWERS, answer_ids=answer_ids.tolist(), original_panel_sha256=sha256(panel_path)))
    identity = dict(panel_sha256=sha256(panel_path), selected_row_indices=selected.tolist(),
        needed_row_indices=needed.tolist(), model_revision=cfg['model_revision'],
        add_bos=cfg['add_bos'], batch_size=cfg['batch_size'], max_length=max_length,
        answer_ids=answer_ids.tolist(), checkpoint_step=cfg['checkpoint_step'])
    previous = None
    if cfg.get('resume_run'):
        previous = json.loads(work.checked(Path(cfg['resume_run']) / 'response_index.json').read_text())
        assert previous['identity'] == identity, 'Resume input identity differs'
    index = dict(schema_version='frozen-region-content-v1', identity=identity, blocks=[], baseline=None)
    index_path = work.run / 'response_index.json'
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
    model.requires_grad_(False)
    model.config.use_cache = False
    assert model.config.n_layer == 24
    module = model.transformer.h[11]

    def budget():
        assert torch.cuda.max_memory_allocated(work.device) <= cfg['maximum_cuda_bytes'], 'CUDA budget exceeded'
        if time.perf_counter() - work.wall_start > cfg['budget_seconds']:
            raise TimeoutError('Declared content-transfer driver budget exceeded')

    def forward(indices, delta=None):
        ids = torch.full((len(indices), max_length), tokenizer.eos_token_id,
                         dtype=torch.long, device=work.device)
        mask = torch.zeros_like(ids)
        positions = []
        for k, i in enumerate(indices):
            sequence = tokens[int(i)]
            ids[k, :len(sequence)] = torch.tensor(sequence, device=work.device)
            mask[k, :len(sequence)] = 1
            positions.append(len(sequence) - 1)
        idx = torch.arange(len(indices), device=work.device)
        pos = torch.tensor(positions, device=work.device)
        cache = {}
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            cache['hidden'] = h[idx, pos].detach().clone()
            if delta is None:
                return output
            edited = h.clone()
            edited[idx, pos] += delta
            return (edited,) + output[1:] if isinstance(output, tuple) else edited
        handle = module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[idx, pos]
                logprobs = logits.double().log_softmax(-1)
                result = dict(logits=logits[:, answer_ids].cpu().numpy(),
                    logprobs=logprobs[:, answer_ids].cpu().numpy(),
                    full_argmax=logits.argmax(-1).cpu().numpy(), hidden=cache['hidden'].cpu().numpy())
        finally:
            handle.remove()
        work.sequence_forwards += len(indices)
        work.token_forwards += ids.numel()
        budget()
        return result

    def checked_output(spec):
        path = work.checked(spec['path'], 'Reusable completed content-transfer block')
        assert sha256(path) == spec['sha256']
        return path

    if previous and previous['baseline']:
        spec = previous['baseline']
        path = checked_output(spec)
        with np.load(path, allow_pickle=False) as saved:
            baseline = {key: saved[key] for key in saved.files}
        assert np.array_equal(baseline['row_indices'], needed)
        index['baseline'] = dict(spec, reused_from=str(cfg['resume_run']))
    else:
        chunks = []
        for start in range(0, len(needed), cfg['batch_size']):
            chunks.append(forward(needed[start:start + cfg['batch_size']]))
            work.progress('baseline', rows=min(start + cfg['batch_size'], len(needed)), total=len(needed))
        baseline = {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}
        baseline['row_indices'] = needed
        path = work.run / 'baseline.npz'
        np.savez_compressed(path, **baseline)
        index['baseline'] = dict(path=str(path), sha256=sha256(path), reused_from=None)
    write(index_path, index)
    probe = needed[:cfg['batch_size']]
    replay = forward(probe, torch.zeros((len(probe), 1024), device=work.device))
    noop_error = float(np.max(np.abs(replay['logits'] - baseline['logits'][:len(probe)])))
    work.checks['same_batch_noop'] = noop_error <= cfg['noop_atol']
    work.environment['noop_maximum_logit_error'] = noop_error
    assert work.checks['same_batch_noop'], noop_error
    structure_path = work.checked(ROOT / cfg['region_run'] / 'STRUCTURE_FREEZE.json')
    structure = json.loads(structure_path.read_text())
    hidden = torch.as_tensor(baseline['hidden'], device=work.device)
    prior_blocks = {} if previous is None else {
        (b['objective'], b['target_seed'], b['method']): b for b in previous['blocks']}
    for objective in cfg['objectives']:
        for target in cfg['target_seeds']:
            snap = next(s for s in snapshots if s['objective'] == objective
                        and s['seed'] == target and s['step'] == cfg['checkpoint_step'])
            checkpoint = work.checked(snap['path'])
            assert sha256(checkpoint) == snap['sha256']
            proposal = work.checked(ROOT / cfg['region_run'] / f'{objective}_s{target-10}_t{target}_proposals.npz')
            masks = region_members(structure, proposal, objective, target)
            state = torch.load(checkpoint, map_location=work.device, weights_only=True)
            sae = (AutoEncoderTopK(1024, 8192, 64) if objective == 'topk' else
                MatryoshkaBatchTopKSAE(1024, 8192, 64, state['group_sizes'].cpu().tolist())).to(work.device)
            sae.load_state_dict(state)
            sae.eval().requires_grad_(False)
            decoder = sae.decoder.weight.T if objective == 'topk' else sae.W_dec
            with torch.no_grad():
                codes = torch.cat([sae.encode(hidden[start:start + cfg['code_batch_size']])
                                   for start in range(0, len(hidden), cfg['code_batch_size'])])
            self_error = float(((codes - codes) @ decoder).abs().max())
            assert self_error == 0
            work.checks[f'{objective}_{target}_selfswap_zero'] = self_error == 0
            for method in METHODS:
                members = masks[method]
                block_identity = dict(checkpoint_sha256=snap['sha256'], structure_sha256=sha256(structure_path),
                    proposal_sha256=sha256(proposal), member_ids=members.tolist())
                key = (objective, target, method)
                shared_raw = next((b for b in index['blocks'] if b['method'] == 'raw'), None)
                if method == 'raw' and shared_raw is not None:
                    block_path = Path(shared_raw['path'])
                    block = dict(shared_raw, objective=objective, target_seed=target,
                        identity=block_identity, reused_from=str(work.run),
                        reused_block=dict(objective=shared_raw['objective'], target_seed=shared_raw['target_seed']))
                elif key in prior_blocks:
                    old = prior_blocks[key]
                    assert old['identity'] == block_identity
                    block_path = checked_output(old)
                    with np.load(block_path, allow_pickle=False) as saved:
                        assert np.array_equal(saved['row_indices'], selected)
                        assert np.array_equal(saved['donor_indices'], donors)
                        assert saved['logits'].shape == (len(selected), 2, 3)
                    block = dict(old, reused_from=str(cfg['resume_run']))
                else:
                    outputs = {name: [] for name in ['logits', 'logprobs', 'full_argmax', 'delta_norm']}
                    for axis in range(2):
                        with torch.no_grad():
                            if method == 'raw':
                                delta = hidden[donor_local[:, axis]] - hidden[recipient_local]
                            else:
                                delta = (codes[donor_local[:, axis]][:, members] - codes[recipient_local][:, members]) @ decoder[members]
                        pieces = []
                        for start in range(0, len(selected), cfg['batch_size']):
                            pieces.append(forward(selected[start:start + cfg['batch_size']],
                                delta[start:start + cfg['batch_size']]))
                            if start % (cfg['batch_size'] * 8) == 0:
                                work.progress('intervention', objective=objective, target=target, method=method,
                                    donor_axis=axis, rows=min(start + cfg['batch_size'], len(selected)), total=len(selected))
                        for name in ['logits', 'logprobs', 'full_argmax']:
                            outputs[name].append(np.concatenate([p[name] for p in pieces]))
                        outputs['delta_norm'].append(delta.double().norm(dim=1).cpu().numpy())
                    arrays = {name: np.stack(values, axis=1) for name, values in outputs.items()}
                    assert all(np.isfinite(a).all() for a in arrays.values())
                    block_path = work.run / f'{objective}_t{target}_{method}.npz'
                    np.savez_compressed(block_path, **arrays, row_indices=selected, donor_indices=donors,
                                        member_ids=members, answer_ids=answer_ids)
                    block = dict(objective=objective, target_seed=target, method=method,
                        path=str(block_path), sha256=sha256(block_path), identity=block_identity, reused_from=None)
                index['blocks'].append(block)
                write(index_path, index)
                work.record(kind='content_transfer_block', task='reflexive_content', component=block_path.name,
                    row_id=0, mode=objective, method=method, target_seed=target, operation='two_donors',
                    split=cfg['split'], rows=len(selected), members=len(members), reused=block['reused_from'] is not None)
                work.progress('method_complete', objective=objective, target=target, method=method,
                              completed_blocks=len(index['blocks']))
            del sae, state, decoder, codes
            gc.collect()
            torch.cuda.empty_cache()
    work.checks['all_methods_complete'] = len(index['blocks']) == len(METHODS) * len(cfg['objectives']) * len(cfg['target_seeds'])
    work.environment['actual_output_bytes'] = sum(p.stat().st_size for p in work.run.rglob('*') if p.is_file())
    work.checks['storage_budget'] = work.environment['actual_output_bytes'] <= cfg['maximum_new_bytes']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = MultisiteWork(cfg, args.config, ['scripts/run_frozen_region_content.py',
        'scripts/prepare_region_content_panel.py', 'scripts/train_grammar_material_support.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py'])
    error = None
    try:
        execute(work)
    except Exception:
        error = traceback.format_exc()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
