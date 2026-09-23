import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, write
from ccad.artifacts import sha256
from train_grammar_material_support import load_config


PARTS = ('verb', 'number', 'gender')
QUERIES = {'verb': [1, 0, 0], 'number': [0, 1, 0], 'gender': [0, 0, 1],
           'verb_number': [1, 1, 0], 'verb_gender': [1, 0, 1],
           'number_gender': [0, 1, 1], 'full': [1, 1, 1]}


def evaluate(config_path):
    cfg = load_config(config_path)
    work = MultisiteWork(cfg, config_path, ['scripts/evaluate_incremental_function_blocks.py',
        'src/ccad/incremental_function_blocks.py', 'scripts/incremental_raw_operator.py',
        'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
    error = None
    try:
        sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        from ccad.incremental_function_blocks import load_checkpoint, encode, group_delete
        from incremental_raw_operator import fit_stage1, fit_stage2, load_operators, predict
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats()
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            dtype='float32', matmul_precision='highest', autocast=False)
        parent = Path(cfg['source_run'])
        panel_path = work.checked(cfg['evaluation_panel'])
        if cfg.get('evaluation_panel_sha256'):
            assert sha256(panel_path) == cfg['evaluation_panel_sha256']
        panel = json.loads(panel_path.read_text(encoding='utf-8'))
        rows = [r for task in cfg['tasks'] for r in panel['rows']
                if r['task'] == task and r['split'] == cfg['evaluation_split']]
        rows = [r for task in cfg['tasks'] for r in [r for r in rows if r['task'] == task][:cfg['eval_pairs_per_task']]]
        assert len(rows) == len(cfg['tasks'])*cfg['eval_pairs_per_task']
        write(work.run/'panel.json', dict(rows=rows, queries=QUERIES, query_order=list(QUERIES)))
        for filename in ('model.safetensors', 'config.json', 'tokenizer.json'):
            work.checked(Path(cfg['model_local_dir'])/filename)
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        tok = transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'], local_files_only=True)
        dim = model.config.hidden_size
        assert dim == 1024
        batch_size = cfg.get('batch_pairs', 16)

        @torch.no_grad()
        def forward(rr, delta=None):
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Incremental functional evaluation exceeded budget')
            length = max(len(r[k]) for r in rr for k in ('good', 'bad'))
            ids = torch.full((2*len(rr), length), tok.eos_token_id, device=work.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(rr):
                for j, key in enumerate(('good', 'bad')):
                    ids[2*i+j, :len(row[key])] = torch.tensor(row[key], device=work.device)
                    mask[2*i+j, :len(row[key])] = 1
            positions = torch.tensor([r['position'] for r in rr], device=work.device).repeat_interleave(2)
            captured = []
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                ix = torch.arange(len(h), device=work.device)
                x = h[ix, positions]
                assert float((x[::2]-x[1::2]).abs().max()) < .001
                captured.append(x[::2].detach().cpu().numpy())
                if delta is None:
                    return output
                edited = h.clone()
                edited[ix, positions] += torch.as_tensor(delta, device=work.device).repeat_interleave(2, 0)
                return (edited, *output[1:]) if isinstance(output, tuple) else edited
            handle = model.get_submodule(cfg['hook_module_path']).register_forward_hook(hook)
            try:
                hidden = model.transformer(ids, attention_mask=mask, use_cache=False).last_hidden_state
            finally:
                handle.remove()
            logits = model.get_output_embeddings()(hidden[:, :-1])
            scores = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            total = (scores*mask[:, 1:]).double().sum(1).reshape(-1, 2)
            work.sequence_forwards += len(ids)
            work.token_forwards += int(mask.sum())
            return (total[:, 0]-total[:, 1]).cpu().numpy(), captured[0]

        responses, methods, quality = {}, {}, []
        vector_dir = work.run/'vectors'
        vector_dir.mkdir()
        if cfg.get('resume_run'):
            prior = Path(cfg['resume_run'])
            prior_cfg = json.loads(work.checked(prior/'config.resolved.json').read_text())
            for key in ('model_revision', 'source_checkpoint', 'training_run', 'evaluation_panel',
                        'evaluation_split', 'eval_pairs_per_task', 'target_seeds', 'arrivals', 'cache_run'):
                assert prior_cfg[key] == cfg[key], key
            prior_panel = json.loads(work.checked(prior/'panel.json').read_text())
            assert prior_panel['rows'] == rows and prior_panel['queries'] == QUERIES
            state_path = work.checked(prior/'current_states.npz')
            with np.load(state_path) as saved:
                hidden, clean = saved['hidden'].copy(), saved['clean'].copy()
            shutil.copy2(state_path, work.run/'current_states.npz')
            with np.load(work.checked(prior/'responses.npz')) as saved:
                responses = {key: saved[key].copy() for key in saved.files}
            methods = json.loads(work.checked(prior/'methods.json').read_text())
            quality = json.loads(work.checked(prior/'natural_quality.json').read_text())
            shutil.copy2(work.checked(prior/'metrics.raw.jsonl'), work.run/'metrics.raw.jsonl')
            for key, value in methods.items():
                vector_path = work.checked(value['vectors'])
                shutil.copy2(vector_path, vector_dir/(key+'.npz'))
                value['vectors'] = (vector_dir/(key+'.npz')).as_posix()
            write(work.run/'methods.json', methods)
            write(work.run/'natural_quality.json', quality)
            np.savez_compressed(work.run/'responses.npz', **responses)
            work.checks['resumed_inputs_and_panel_identical'] = True
        else:
            hidden, clean = [], []
            for off in range(0, len(rows), batch_size):
                margins, x = forward(rows[off:off+batch_size])
                clean.extend(margins)
                hidden.extend(x)
            hidden, clean = np.asarray(hidden, np.float32), np.asarray(clean, np.float64)
            np.savez_compressed(work.run/'current_states.npz', hidden=hidden, clean=clean)
        count = min(batch_size, len(rows))
        repeat, _ = forward(rows[:count], np.zeros_like(hidden[:count]))
        np.testing.assert_array_equal(repeat, clean[:count])
        work.checks['same_batch_zero_update_exact'] = True
        state = torch.load(work.checked(cfg['source_checkpoint']), weights_only=True, map_location=work.device)
        source = AutoEncoderTopK(dim, len(state['encoder.weight']), int(state['k'])).to(work.device)
        source.load_state_dict(state)
        source.eval().requires_grad_(False)
        gates = torch.as_tensor(np.load(work.checked(parent/'topk_s1_source.npz'))['gate'], device=work.device)
        assert tuple(gates.shape) == (8192, 3) and bool((gates.sum(1) <= 1).all())
        x = torch.as_tensor(hidden, device=work.device)
        with torch.no_grad():
            z = source.encode(x)
            source_delta = torch.stack([-(z*gates[:, i])@source.decoder.weight.T for i in range(3)]).cpu().numpy()
        responses['clean'] = clean

        def execute(key, delta, metadata, valid_parts=PARTS):
            assert delta.shape == (3, len(rows), dim) and np.isfinite(delta).all()
            output = np.full((len(QUERIES), len(rows)), np.nan, np.float64)
            valid = []
            for qi, (name, request) in enumerate(QUERIES.items()):
                active = [p for p, value in zip(PARTS, request) if value]
                enabled = set(active) <= set(valid_parts)
                valid.append(enabled)
                if not enabled:
                    continue
                total = np.einsum('p,pnd->nd', np.asarray(request, np.float32), delta)
                for off in range(0, len(rows), batch_size):
                    output[qi, off:off+batch_size], _ = forward(rows[off:off+batch_size], total[off:off+batch_size])
                with (work.run/'metrics.raw.jsonl').open('a', encoding='utf-8') as stream:
                    for row, value in zip(rows, output[qi]):
                        stream.write(json.dumps(dict(kind='incremental_function', task=row['task'],
                            row_id=row['row_id'], component=row['task']+':'+str(row['row_id']),
                            mode=name, operation=name, method=key, seed=1,
                            target_seed=metadata['target_seed'], split=cfg['evaluation_split'],
                            margin=float(value), accuracy=bool(value > 0)))+'\n')
                work.progress('QUERY', method=key, query=name, pairs=len(rows))
            vector_path = vector_dir/(key+'.npz')
            np.savez_compressed(vector_path, delta=delta)
            methods[key] = dict(metadata, valid_queries=valid, vectors=vector_path.as_posix())
            responses[key] = output
            np.savez_compressed(work.run/'responses.npz', **responses)
            write(work.run/'methods.json', methods)

        if 'source' not in responses:
            execute('source', source_delta, dict(method='source', target_seed=None, stage=0, arrival=None))
        natural = np.load(work.checked(cfg['natural_states']))['hidden'][-1024:].astype(np.float32)
        variance = np.square(natural-natural.mean(0)).sum()
        index = json.loads(work.checked(Path(cfg['training_run'])/'final_checkpoints.json').read_text())
        entries = index['checkpoints']
        entries = [entry for entry in entries if entry['target_seed'] in cfg['target_seeds']
                   and entry['arrival'] in cfg['arrivals']]
        for entry in entries:
            key = f"s{entry['target_seed']}__{entry['arrival']}__{entry['stage']}__{entry['method']}"
            if key in methods:
                assert all(methods[key][name] == entry[name] for name in ('path', 'method', 'stage', 'target_seed', 'arrival'))
                continue
            path = work.checked(entry['path'])
            ae, groups, metadata = load_checkpoint(path, work.device)
            ae.eval().requires_grad_(False)
            allocation = metadata['allocation']
            valid_parts = [p for p in PARTS if p in groups]
            with torch.no_grad():
                delta = np.stack([group_delete(ae, x, groups, allocation, [part]).cpu().numpy()
                                  if part in groups else np.zeros_like(hidden) for part in PARTS])
                nt = torch.as_tensor(natural, device=work.device)
                codes = encode(ae, nt, groups, allocation)
                rec = ae.decode(codes)
                q = dict(method=entry['method'], target_seed=entry['target_seed'], arrival=entry['arrival'], stage=entry['stage'],
                    fve=float(1-(rec-nt).square().sum()/variance), l0=float((codes > 0).sum(1).float().mean()),
                    alive=int((codes > 0).any(0).sum()), decoder_norm_mean=float(ae.decoder.weight.norm(dim=0).mean()))
            key = f"s{entry['target_seed']}__{entry['arrival']}__{entry['stage']}__{entry['method']}"
            execute(key, delta, entry, valid_parts)
            quality.append(q)
            write(work.run/'natural_quality.json', quality)
            del ae
        for seed in cfg['target_seeds']:
            if f'native_s{seed}' in methods:
                continue
            state = torch.load(work.checked(cfg['target_checkpoint_template'].format(seed=seed)), weights_only=True, map_location=work.device)
            ae = AutoEncoderTopK(dim, len(state['encoder.weight']), int(state['k'])).to(work.device)
            ae.load_state_dict(state)
            ae.eval().requires_grad_(False)
            mapping_path = cfg.get('native_maps', {}).get(str(seed), str(parent/f'topk_s1_t{seed}_map.npz'))
            mapping = np.load(work.checked(mapping_path))
            ids = torch.as_tensor(mapping['target_members'], device=work.device)
            gate = torch.as_tensor(mapping['partition64'], device=work.device)
            assert bool((gate.sum(1) <= 1.000001).all())
            with torch.no_grad():
                codes = ae.encode(x)[:, ids]
                delta = torch.stack([-(codes*gate[:, i])@ae.decoder.weight[:, ids].T for i in range(3)]).cpu().numpy()
                nc = ae.encode(torch.as_tensor(natural, device=work.device))
                rec = ae.decode(nc).cpu().numpy()
            execute(f'native_s{seed}', delta, dict(method='native', target_seed=seed, stage=0, arrival=None))
            quality.append(dict(method='original', target_seed=seed, arrival=None, stage=0,
                fve=float(1-np.square(rec-natural).sum()/variance), l0=float((nc > 0).sum(1).float().mean()),
                alive=int((nc > 0).any(0).sum()), decoder_norm_mean=float(ae.decoder.weight.norm(dim=0).mean())))
            write(work.run/'natural_quality.json', quality)
            del ae
        cache_index = json.loads(work.checked(Path(cfg['cache_run'])/'cache_index.json').read_text())
        for arrival in cfg['arrivals']:
            if 'raw__'+arrival in methods:
                continue
            raw_dir = work.run/'raw'/arrival
            raw_dir.mkdir(parents=True)
            stage1_path, stage2_path = raw_dir/'stage1.npz', raw_dir/'stage2.npz'
            fit_stage1(work.checked(cache_index['arrivals'][arrival]['stage1']), stage1_path)
            fit_stage2(work.checked(cache_index['arrivals'][arrival]['stage2']), stage1_path, stage2_path)
            operators = load_operators(stage2_path)
            delta = np.stack([predict(operators, hidden, part) for part in PARTS]).astype(np.float32)
            execute('raw__'+arrival, delta, dict(method='raw', target_seed=None, stage=2, arrival=arrival))
        work.checks.update(actual_model_interventions=True, saved_all_partial_results=True)
        return work.finish()
    except Exception as exc:
        error = traceback.format_exc()
        work.finish(error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    raise SystemExit(evaluate(parser.parse_args().config))
