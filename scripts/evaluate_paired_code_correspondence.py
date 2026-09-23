import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import transformers

from ccad.artifacts import sha256
from run_causalgym_multisite import MultisiteWork, write
from run_shift_transfer import input_member_delta
from train_grammar_material_support import load_config


PARTS = ('verb', 'number', 'gender')


def request_table(source_part, subsets):
    groups = np.eye(3, dtype=np.float32)[:, source_part]
    requests = []

    def add(name, family, steps, diagnostic=False):
        requests.append(dict(name=name, family=family, steps=[q.tolist() for q in steps],
                             diagnostic=diagnostic))

    for index, q in enumerate(subsets):
        add(f'subset_{index:02d}', 'subset', [q])
    for first, second in ((0, 1), (0, 2), (1, 2)):
        add(f'{PARTS[first]}_{PARTS[second]}', 'pair', [groups[first]+groups[second]])
    for index, name in enumerate(PARTS):
        for strength in (.25, .5, .75):
            add(f'{name}_{strength:g}', 'strength', [groups[index]*strength])
    for first in range(3):
        for second in range(3):
            if first != second:
                add(f'{PARTS[first]}_half_then_{PARTS[second]}', 'sequential',
                    [.5*groups[first], groups[second]])
    for index, name in enumerate(PARTS):
        add(name, 'trained_group', [groups[index]], True)
    assert len(requests) == 29
    for first in range(3):
        for second in range(3):
            if first != second:
                add(f'{PARTS[first]}_half_with_{PARTS[second]}', 'simultaneous_diagnostic',
                    [.5*groups[first]+groups[second]], True)
    return requests


def evaluate(config_path):
    cfg = load_config(config_path)
    work = MultisiteWork(cfg, config_path, ['scripts/evaluate_paired_code_correspondence.py',
        'scripts/run_shift_transfer.py', 'src/ccad/incremental_function_blocks.py',
        'scripts/run_causalgym_multisite.py', 'scripts/train_grammar_material_support.py',
        'src/ccad/artifacts.py'])
    try:
        sys.path[:0] = [cfg['dictionary_overlay_dir'], cfg['dictionary_source_dir']]
        import ccad.incremental_function_blocks as api
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            dtype='float32', matmul_precision='highest', autocast=False, cpu_threads=2)
        training = Path(cfg['training_run'])
        with np.load(work.checked(training/'evaluation_subsets.npz')) as saved:
            members, parts, subsets = saved['source_members'], saved['source_part'], saved['requests']
        assert members.shape == parts.shape == (192,) and np.all(np.diff(members) > 0)
        assert np.array_equal(np.bincount(parts), [64, 64, 64]) and subsets.shape == (8, 192)
        rng = np.random.default_rng(20260924)
        expected = np.zeros_like(subsets)
        for q in expected:
            for pi in range(3):
                q[rng.choice(np.flatnonzero(parts == pi), 32, replace=False)] = 1
        np.testing.assert_array_equal(subsets, expected)
        requests = request_table(parts, subsets)
        panel_path = work.checked(cfg['evaluation_panel'])
        panel = json.loads(panel_path.read_text(encoding='utf-8'))
        rows = [row for task in cfg['tasks'] for row in
                [r for r in panel['rows'] if r['task'] == task and r['split'] == cfg['evaluation_split']]
                [:cfg['eval_pairs_per_task']]]
        assert len(rows) == len(cfg['tasks'])*cfg['eval_pairs_per_task']
        for row in rows:
            pos = row['position']
            assert row['good'][:pos+1] == row['bad'][:pos+1]
            assert 0 <= pos < min(len(row['good']), len(row['bad']))
        panel_saved = dict(rows=rows, requests=requests, request_order=[q['name'] for q in requests],
            source_members=members.tolist(), source_part=parts.tolist(), subset_seed=20260924,
            main_request_count=29, simultaneous_diagnostic_count=6,
            evaluation_panel_sha256=sha256(panel_path))
        write(work.run/'panel.json', panel_saved)
        np.savez_compressed(work.run/'requests.npz', source_members=members, source_part=parts,
            q_first=np.asarray([q['steps'][0] for q in requests], np.float32),
            q_second=np.asarray([q['steps'][1] if len(q['steps']) == 2 else np.zeros(192)
                                 for q in requests], np.float32),
            step_count=np.asarray([len(q['steps']) for q in requests]))
        index = dict(methods=[], requests=requests, chunks=[], source_checkpoint=cfg['source_checkpoint'],
                     target_seeds=cfg['target_seeds'])
        prior_chunks = {}
        if cfg.get('resume_run'):
            prior = Path(cfg['resume_run'])
            old_panel = json.loads(work.checked(prior/'panel.json').read_text())
            assert old_panel == panel_saved
            old_cfg = json.loads(work.checked(prior/'config.resolved.json').read_text())
            for key in ('source_checkpoint', 'training_run', 'target_seeds', 'program_checkpoints',
                        'model_revision', 'hook_module_path', 'batch_pairs'):
                assert old_cfg.get(key) == cfg.get(key), key
            old_index = json.loads(work.checked(prior/'response_index.json').read_text())
            prior_chunks = {(item['method'], item['request']): item for item in old_index['chunks']}
        write(work.run/'response_index.json', index)
        for filename in ('model.safetensors', 'config.json', 'tokenizer.json'):
            work.checked(Path(cfg['model_local_dir'])/filename)
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'], local_files_only=True)
        source = api.load_dictionary(work.checked(cfg['source_checkpoint']), work.device).eval().requires_grad_(False)
        source_ids = torch.as_tensor(members, device=work.device)
        with np.load(work.checked(Path(cfg['source_run'])/'topk_s1_source.npz')) as saved:
            original_gate = saved['gate']
        original_members = np.flatnonzero(original_gate.sum(1))
        np.testing.assert_array_equal(original_members, members)
        np.testing.assert_array_equal(original_gate[members].argmax(1), parts)
        source_decoder = source.decoder.weight[:, source_ids].T
        source_spec = dict(sae=source, member_ids=source_ids, decoder=source_decoder,
                           part_ids=torch.as_tensor(parts, device=work.device))
        batch_size = cfg.get('batch_pairs', 16)

        @torch.no_grad()
        def forward(batch, delta=None):
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Paired-code response budget exceeded')
            length = max(len(row[key]) for row in batch for key in ('good', 'bad'))
            ids = torch.full((2*len(batch), length), tokenizer.eos_token_id, device=work.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                for j, key in enumerate(('good', 'bad')):
                    ids[2*i+j, :len(row[key])] = torch.tensor(row[key], device=work.device)
                    mask[2*i+j, :len(row[key])] = 1
            positions = torch.tensor([row['position'] for row in batch], device=work.device).repeat_interleave(2)
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
                edited[ix, positions] += delta.repeat_interleave(2, 0)
                return (edited, *output[1:]) if isinstance(output, tuple) else edited

            handle = model.get_submodule(cfg['hook_module_path']).register_forward_hook(hook)
            try:
                hidden = model.transformer(ids, attention_mask=mask, use_cache=False).last_hidden_state
            finally:
                handle.remove()
            logits = model.get_output_embeddings()(hidden[:, :-1])
            score = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            totals = (score*mask[:, 1:]).double().sum(1).reshape(-1, 2)
            work.sequence_forwards += len(ids)
            work.token_forwards += int(mask.sum())
            return (totals[:, 0]-totals[:, 1]).cpu().numpy(), captured[0]

        baseline_margin, captured = [], []
        for off in range(0, len(rows), batch_size):
            margin, hidden = forward(rows[off:off+batch_size])
            baseline_margin.append(margin)
            captured.append(hidden)
        clean, hidden = np.concatenate(baseline_margin), np.concatenate(captured)
        x0 = torch.tensor(hidden, device=work.device)
        replay, _ = forward(rows[:batch_size], torch.zeros_like(x0[:batch_size]))
        np.testing.assert_array_equal(replay, clean[:batch_size])
        np.savez_compressed(work.run/'baseline.npz', clean=clean, hidden=hidden)
        work.progress('BASELINE', pairs=len(rows), zero_replay_max_error=float(np.max(np.abs(replay-clean[:batch_size]))))
        chunks_dir = work.run/'chunks'
        chunks_dir.mkdir()
        all_margins, all_state_metrics, method_names = [], [], []
        call_counts = dict(operation_source_encoder_calls=0, operation_target_encoder_calls=0,
            measurement_source_encoder_calls=0, measurement_target_encoder_calls=0,
            measurement_source_states=0, measurement_target_states=0)

        @torch.no_grad()
        def apply_request(h, q, kind, ae, relation):
            if kind == 'source':
                call_counts['operation_source_encoder_calls'] += 1
                call_counts['measurement_source_encoder_calls'] += 1
                call_counts['measurement_source_states'] += len(h)
                z = source.encode(h)
                delta = -(z[:, source_ids]*q)@source_decoder
                altered = source.encode(h+delta)
                ideal = z.clone()
                ideal[:, source_ids] *= 1-q
                diagnostics = dict(changed=(z[:, source_ids]*q != 0).sum(1),
                    source_actual_reencode_error_squared=(altered-ideal).double().square().sum(1))
            elif kind in ('physical', 'code'):
                call_counts['operation_target_encoder_calls'] += 1
                call_counts['measurement_source_encoder_calls'] += 1
                call_counts['measurement_target_encoder_calls'] += 1
                call_counts['measurement_source_states'] += len(h)
                call_counts['measurement_target_states'] += len(h)
                z = api.encode(ae, h, {}, 'global')
                mask = relation@q
                assert float(mask.min()) >= -1e-7 and float(mask.max()) <= 1.000002
                delta = -(z*mask)@ae.decoder.weight.T
                hs = h-(source.encode(h)[:, source_ids]*q)@source_decoder
                zs = api.encode(ae, hs, {}, 'global')
                error = zs-(1-mask)*z
                decoded = error@ae.decoder.weight.T
                residual = (ae.decode(zs)-hs)-(ae.decode(z)-h)
                physical = (h+delta)-hs
                diagnostics = dict(changed=(z*mask != 0).sum(1),
                    paired_code_error_squared=error.double().square().sum(1),
                    decoded_code_error_squared=decoded.double().square().sum(1),
                    residual_difference_squared=residual.double().square().sum(1),
                    residual_decoded_cross=-2*(residual.double()*decoded.double()).sum(1),
                    same_incoming_physical_error_squared=physical.double().square().sum(1),
                    identity_max_error=(physical-residual+decoded).abs().amax(1))
            else:
                mode = 'raw_reconstruction' if kind.startswith('readout') else 'input_tangent_budget'
                call_counts['operation_source_encoder_calls'] += 2 if mode == 'raw_reconstruction' else 1
                call_counts['operation_target_encoder_calls'] += 1
                delta, _ = input_member_delta(h, ae, source_spec, q, mode,
                    cfg.get('members_per_source', 2)*len(members), torch.ones(len(h), device=work.device))
                diagnostics = {}
            return h+delta, diagnostics

        def execute(key, kind, ae=None, relation=None, metadata=None):
            method_names.append(key)
            index['methods'].append(dict(key=key, kind=kind, **(metadata or {})))
            margins, metrics = [], []
            for qi, request in enumerate(requests):
                old = prior_chunks.get((key, request['name']))
                if old is not None:
                    path = work.checked(old['path'])
                    assert sha256(path) == old['sha256']
                    record = dict(old, reused_from=cfg['resume_run'])
                else:
                    h = x0.clone()
                    first_delta = torch.zeros_like(h)
                    diagnostics = {}
                    with torch.no_grad():
                        for step, values in enumerate(request['steps']):
                            q = torch.tensor(values, device=work.device)
                            h, stats = apply_request(h, q, kind, ae, relation)
                            if step == 0:
                                first_delta = h-x0
                            for name, value in stats.items():
                                diagnostics[f'step{step+1}__{name}'] = value.cpu().numpy()
                        delta = h-x0
                    output = []
                    for off in range(0, len(rows), batch_size):
                        value, actual = forward(rows[off:off+batch_size], delta[off:off+batch_size])
                        np.testing.assert_array_equal(actual, hidden[off:off+batch_size])
                        output.append(value)
                    path = chunks_dir/f'{key}__q{qi:02d}.npz'
                    np.savez_compressed(path, margins=np.concatenate(output), delta=delta.cpu().numpy(),
                        first_delta=first_delta.cpu().numpy(), **diagnostics)
                    record = dict(method=key, request=request['name'], path=path.as_posix(), sha256=sha256(path))
                with np.load(path) as saved:
                    margins.append(saved['margins'])
                    delta = saved['delta'].astype(np.float64)
                    if key == 'source':
                        source_delta = delta
                    else:
                        source_record = next(item for item in index['chunks']
                            if item['method'] == 'source' and item['request'] == request['name'])
                        with np.load(source_record['path']) as ref:
                            source_delta = ref['delta'].astype(np.float64)
                    metrics.append(np.stack([np.square(delta-source_delta).sum(1),
                        np.square(source_delta).sum(1), np.square(delta).sum(1)]))
                index['chunks'].append(record)
                write(work.run/'response_index.json', index)
                work.progress('REQUEST', method=key, request=request['name'], request_number=qi+1,
                    requests=len(requests), pairs=len(rows), reused=old is not None)
            all_margins.append(np.stack(margins))
            all_state_metrics.append(np.stack(metrics))

        execute('source', 'source', metadata=dict(target_seed=None, source_runtime=True))
        checkpoints = json.loads(work.checked(training/'final_checkpoints.json').read_text())['checkpoints']
        selected = [entry for entry in checkpoints if entry['target_seed'] in cfg['target_seeds']]
        assert len(selected) == 2*len(cfg['target_seeds'])
        for seed in cfg['target_seeds']:
            original_path = work.checked(cfg['target_checkpoint_template'].format(seed=seed))
            ae = api.load_dictionary(original_path, work.device).eval().requires_grad_(False)
            execute(f's{seed}__readout_initial', 'readout_initial', ae,
                    metadata=dict(target_seed=seed, path=str(original_path), source_runtime=True))
            del ae
            for entry in [item for item in selected if item['target_seed'] == seed]:
                path = work.checked(entry['path'])
                assert sha256(path) == entry['sha256']
                saved = torch.load(path, map_location=work.device, weights_only=True)
                np.testing.assert_array_equal(saved['source_members'].cpu().numpy(), members)
                np.testing.assert_array_equal(saved['source_part'].cpu().numpy(), parts)
                state = saved['dictionary']
                ae = AutoEncoderTopK(1024, 8192, int(state['k'])).to(work.device)
                ae.load_state_dict(state)
                ae.eval().requires_grad_(False)
                relation = saved['A']
                assert relation.shape == (8192, 192)
                assert float(relation.min()) >= -1e-7 and float(relation.sum(1).max()) <= 1.000002
                execute(f's{seed}__{entry["method"]}', entry['method'], ae, relation,
                    metadata=dict(target_seed=seed, path=str(path), sha256=entry['sha256'], source_runtime=False,
                        measurement_source_calls='One source encoder per request step for diagnostics only',
                        diagnostic_only='Paired identity reads source; deployment delta uses target encode, A and target decoder'))
                del ae, saved, relation
            program_path = cfg.get('program_checkpoints', {}).get(str(seed))
            if program_path is not None:
                path = work.checked(program_path)
                saved = torch.load(path, map_location=work.device, weights_only=True)
                state = saved['dictionary']
                ae = AutoEncoderTopK(1024, 8192, int(state['k'])).to(work.device)
                ae.load_state_dict(state)
                ae.eval().requires_grad_(False)
                metadata = dict(target_seed=seed, path=str(path), sha256=sha256(path),
                    source_runtime=True, historical_training=True,
                    gain_used=False, gain_reason='Original program variant evaluates tangent; gain applies only to gain variant',
                    execution='Original input_tangent_budget with arbitrary192 source coefficients')
                execute(f's{seed}__program', 'program', ae, metadata=metadata)
                execute(f's{seed}__readout_program', 'readout_program', ae, metadata=metadata)
                del ae, saved
        margins = np.stack(all_margins)
        np.savez_compressed(work.run/'responses.npz', margins=margins, clean=clean,
            method_names=np.asarray(method_names), request_names=np.asarray([q['name'] for q in requests]))
        for seed in cfg['target_seeds']:
            selected_names = ['source']+[name for name in method_names if name.startswith(f's{seed}__')]
            ids = [method_names.index(name) for name in selected_names]
            np.savez_compressed(work.run/f'responses_s{seed}.npz', margins=margins[ids], clean=clean,
                method_names=np.asarray([name.split('__', 1)[-1] for name in selected_names]),
                request_names=np.asarray([q['name'] for q in requests]), target_seed=seed)
        np.savez_compressed(work.run/'state_metrics.npz', values=np.stack(all_state_metrics),
            metric_names=np.asarray(['local_delta_error_squared', 'source_delta_energy', 'target_delta_energy']),
            method_names=np.asarray(method_names), request_names=np.asarray([q['name'] for q in requests]))
        write(work.run/'execution_calls.json', dict(**call_counts,
            scope='Current run actual calls; reused chunks excluded. Measurement calls are included in driver time.',
            deployment='Physical/code delta uses one target encoder per step and zero source calls; its diagnostics read source.',
            sequence_forwards=work.sequence_forwards, token_forwards=work.token_forwards))
        with (work.run/'metrics.raw.jsonl').open('w', encoding='utf-8') as stream:
            for mi, method in enumerate(method_names):
                for qi, request in enumerate(requests):
                    for ri, row in enumerate(rows):
                        stream.write(json.dumps(dict(kind='paired_code_response', task=row['task'],
                            row_id=row['row_id'], component=row['task']+':'+str(row['row_id']),
                            method=method, mode=request['name'], operation=request['name'],
                            request=request['name'], family=request['family'], seed=1,
                            target_seed=next(item['target_seed'] for item in index['methods'] if item['key'] == method),
                            split=cfg['evaluation_split'], margin=float(margins[mi, qi, ri]),
                            correct=bool(margins[mi, qi, ri] > 0)))+'\n')
        work.checks.update(actual_model_interventions=True, source_reencoded_each_step=True,
            target_reencoded_each_step=True, frozen_subset_identity=True, zero_replay_exact=True)
        return work.finish()
    except Exception:
        work.finish(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    raise SystemExit(evaluate(parser.parse_args().config))
