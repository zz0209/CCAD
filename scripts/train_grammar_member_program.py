import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, ROOT, write
from run_shift_transfer import input_member_delta
from ccad.artifacts import sha256


def load_target_parameters(target, initial, trained, group=None):
    assert group in (None, 'encoder', 'decoder')
    target.load_state_dict(initial)
    if group is None:
        target.load_state_dict(trained)
    else:
        keys = ('encoder.weight', 'encoder.bias', 'b_dec') if group == 'encoder' else ('decoder.weight',)
        state = dict(initial)
        state.update({key: trained[key] for key in keys})
        target.load_state_dict(state)


def freeze_decoder_for_training(target):
    target.requires_grad_(False)
    target.encoder.requires_grad_(True)
    assert {name for name, value in target.named_parameters() if value.requires_grad} == {'encoder.weight', 'encoder.bias'}


def balanced_source_schedule(seeds, steps, training_seed, endpoint_count=4):
    seeds = np.asarray(seeds, dtype=np.int64)
    assert len(seeds) in (1, 2) and len(set(seeds.tolist())) == len(seeds)
    rng = np.random.default_rng(training_seed+30000)
    schedule = np.empty(steps, dtype=np.int64)
    endpoint_positions = np.arange(0, steps, 2)
    endpoint_groups = [endpoint_positions[i::endpoint_count] for i in range(endpoint_count)]
    # 每类 endpoint 分别平衡；小样本中的不足一组请求按 endpoint 总量平衡。
    groups = endpoint_groups if all(len(group) % len(seeds) == 0 for group in endpoint_groups) else [endpoint_positions]
    for positions in [*groups, np.arange(1, steps, 2)]:
        source_ids = np.resize(seeds, len(positions))
        rng.shuffle(source_ids)
        schedule[positions] = source_ids
    return schedule


def grouped_source_members(gate):
    assert gate.ndim == 2 and gate.shape[1] == 3
    members = gate.sum(1).nonzero().flatten()
    parts = gate[members].argmax(1)
    assert bool(((gate == 0) | (gate == 1)).all()) and bool((gate.sum(1) <= 1).all())
    assert len(members) == 192 and torch.equal(torch.bincount(parts, minlength=3), torch.full((3,), 64, device=gate.device))
    return members, parts


def check_training_sources(c, output):
    seeds = c['training_source_seeds']
    definitions, identity = {}, {}
    request = torch.tensor([.125, .625, 1.])
    for seed in seeds:
        path = Path(c['source_run'])/f'topk_s{seed}_source.npz'
        gate = torch.from_numpy(np.load(path)['gate']).float()
        members, parts = grouped_source_members(gate)
        active_q = request[parts]
        torch.testing.assert_close(active_q, gate[members]@request, rtol=0, atol=0)
        # 从真实 gate 同时按组和按成员计算请求，检查各源自身的坐标顺序。
        values = torch.arange(1, len(members)+1, dtype=torch.float32)
        member_sum = (values*active_q).sum()
        group_sum = sum(request[p]*values[parts == p].sum() for p in range(3))
        torch.testing.assert_close(member_sum, group_sum, rtol=0, atol=0)
        definitions[seed] = (members, parts)
        identity[str(seed)] = dict(gate=str(path.resolve()), gate_sha256=sha256(path), members=members.tolist(),
                                   part_ids=parts.tolist(), group_counts=torch.bincount(parts).tolist())
    schedule_checks = {}
    for steps in [8, 512]:
        schedule = balanced_source_schedule(seeds, steps, c['training_seed'])
        assert np.array_equal(schedule, balanced_source_schedule(seeds, steps, c['training_seed']))
        counts = {kind: {str(seed): int((schedule[parity::2] == seed).sum()) for seed in seeds}
                  for kind, parity in [('endpoint', 0), ('continuous', 1)]}
        assert all(max(v.values())-min(v.values()) <= 1 for v in counts.values())
        if steps == 512:
            for endpoint in range(4):
                selected = schedule[np.arange(2*endpoint, steps, 8)]
                assert len({int((selected == seed).sum()) for seed in seeds}) == 1
        fit_count = len(c['tasks'])*(8 if steps == 8 else c['fit_pairs_per_task'])
        rng = np.random.default_rng(c['training_seed'])
        rng.random((steps, 3))
        order = rng.permutation(fit_count)
        row_batches = np.array([[order[(step*c['batch_pairs']+j) % fit_count]
            for j in range(c['batch_pairs'])] for step in range(steps)])
        counts['unique_rows'] = {str(seed): len(np.unique(row_batches[schedule == seed])) for seed in seeds}
        counts['available_rows'] = fit_count
        schedule_checks[str(steps)] = counts
    ordering = {f'{a}_{b}': dict(member_ids_differ=int((definitions[a][0] != definitions[b][0]).sum()),
        part_order_differs=int((definitions[a][1] != definitions[b][1]).sum()))
        for i, a in enumerate(seeds) for b in seeds[i+1:]}
    result = dict(status='PASS', written_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha256(Path(__file__)), config_sha256=sha256(Path(c['_check_config_path'])),
                  source_definitions=identity, actual_source_ordering=ordering,
                  balanced_request_types=schedule_checks, request_indexing_exact=True,
                  single_source_schedule_constant=bool((balanced_source_schedule([seeds[0]], 512, c['training_seed']) == seeds[0]).all()),
                  deterministic_resume_schedule=True, gpu_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    print(json.dumps(dict(status=result['status'], output=str(output),
        actual_source_ordering=ordering, schedule_checks=schedule_checks)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--target-seed', type=int)
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--heldout-part', choices=['verb', 'number', 'gender'])
    parser.add_argument('--member-requests', action='store_true')
    parser.add_argument('--evaluate-from', type=Path)
    parser.add_argument('--isolated-evaluation', action='store_true')
    parser.add_argument('--finite-columns', action='store_true')
    parser.add_argument('--adapt-part', choices=['verb', 'number', 'gender'])
    parser.add_argument('--new-fit-pairs', type=int)
    parser.add_argument('--adapt-smoke', action='store_true')
    parser.add_argument('--part-columns', action='store_true')
    parser.add_argument('--evaluation-pairs', type=int)
    parser.add_argument('--program-smoke', action='store_true')
    parser.add_argument('--check-training-sources', type=Path)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if args.check_training_sources is not None:
        c['_check_config_path'] = str(args.config)
        return check_training_sources(c, args.check_training_sources)
    if args.member_requests:
        c['member_requests'] = True
    if args.heldout_part is not None:
        c['heldout_part'] = args.heldout_part
    part_names = ['verb', 'number', 'gender']
    heldout = part_names.index(c['heldout_part']) if c.get('heldout_part') else None
    training_parts = [i for i in range(3) if i != heldout]
    if args.adapt_part is not None:
        assert heldout is None and args.new_fit_pairs is not None
        c['adapt_part'], c['new_fit_pairs'] = args.adapt_part, args.new_fit_pairs
        c['run_id_template'] = c['run_id_template'].replace('{adapt_part}', args.adapt_part).replace('{new_fit_pairs}', str(args.new_fit_pairs))
    if args.source_seed is not None:
        assert args.source_seed in c['source_seeds']
        c['source_seed'] = args.source_seed
    source_seed = c.get('source_seed', 1)
    if 'source_checkpoint_template' in c:
        c['source_checkpoint'] = c['source_checkpoint_template'].format(seed=source_seed)
    if args.target_seed is not None:
        assert args.target_seed in c['target_seeds']
        c['target_seed'] = args.target_seed
        c['seeds'] = [source_seed, args.target_seed]
        c['run_id'] = c['run_id_template'].format(seed=args.target_seed, source_seed=source_seed,
                                                heldout=c.get('heldout_part', 'none'))
        c['target_checkpoint'] = c['target_checkpoint_template'].format(seed=args.target_seed)
        if str(args.target_seed) in c.get('resume_by_seed', {}):
            c['resume_checkpoints'] = c['resume_by_seed'][str(args.target_seed)]
    if c.get('exclude_self_transfer'):
        assert source_seed != c['target_seed']
    if 'evaluation_by_target' in c:
        c['evaluation_checkpoints'] = c['evaluation_by_target'][str(c['target_seed'])]
    if c.get('member_requests'):
        c['run_id'] += '_MEMBER'
        c['variants'] = ['program']
    if args.evaluate_from is not None:
        saved_config = json.loads((args.evaluate_from/'config.resolved.json').read_text())
        assert saved_config['target_seed'] == c['target_seed']
        assert saved_config.get('source_seed', 1) == source_seed
        assert saved_config.get('heldout_part') == c.get('heldout_part')
        c['evaluation_parent'] = args.evaluate_from.as_posix()
        c['evaluation_checkpoints'] = {name: dict(path=(args.evaluate_from/f'{name}_step512.pt').as_posix(), execution='tangent')
                                       for name in saved_config['variants']}
        c['evaluation_checkpoints']['readout_program'] = dict(
            path=(args.evaluate_from/'program_step512.pt').as_posix(), execution='readout')
        c['variants'], c['steps'] = [], 0
        c['run_id'] += '_REPLAY'
    if args.isolated_evaluation:
        assert heldout is not None and args.evaluate_from is not None
        c['isolated_evaluation'] = True
        c['run_id'] += '_ISOLATED'
    if args.finite_columns:
        c['native_operation'] = 'input_finite_budget'
        c['eval_batch_pairs'] = min(c['eval_batch_pairs'], 4)
        c['run_id'] += '_FINITE'
    if c.get('adapt_part'):
        assert c['steps'] > 0 and not c.get('evaluation_checkpoints')
        prior_run = Path(c['prior_run_template'].format(adapt_part=c['adapt_part'], seed=c['target_seed']))
        prior_config = json.loads((prior_run/'config.resolved.json').read_text())
        assert json.loads((prior_run/'status.json').read_text())['status'] == 'PASS'
        assert prior_config['heldout_part'] == c['adapt_part']
        assert prior_config['target_seed'] == c['target_seed'] and prior_config.get('source_seed', 1) == source_seed
        c['prior_checkpoint'] = (prior_run/'program_step512.pt').as_posix()
        c['prior_training_steps'] = prior_config['steps']
    if args.part_columns:
        c['native_operation'] = 'input_part_budget'
        c['run_id'] += '_PART_COLUMNS'
    if args.evaluation_pairs is not None:
        c['eval_pairs_per_task'] = args.evaluation_pairs
        c['run_id'] += f'_EVAL{args.evaluation_pairs}'
    if args.adapt_smoke:
        assert c.get('adapt_part')
        c.update(steps=8, checkpoint_every=8, fit_pairs_per_task=8,
                 eval_pairs_per_task=2, learning_curve_steps=[4], budget_seconds=240)
        c['run_id'] += '_SMOKE_V2'
    if args.program_smoke:
        assert c['variants'] == ['program']
        c.update(steps=8, checkpoint_every=8, fit_pairs_per_task=8,
                 eval_pairs_per_task=2, budget_seconds=240)
        c['run_id'] += '_SMOKE'
    multiple_sources = 'training_source_seeds' in c
    training_source_seeds = c.get('training_source_seeds', [source_seed])
    if multiple_sources:
        assert len(training_source_seeds) in (1, 2) and len(set(training_source_seeds)) == len(training_source_seeds)
        assert c['variants'] == ['program'] and c['steps'] > 0
        assert all(not c.get(key) for key in ['heldout_part', 'member_requests', 'adapt_part',
            'fixed_control', 'isolated_evaluation', 'freeze_target_decoder', 'evaluation_checkpoints'])
        assert c.get('native_operation', 'input_tangent_budget') == 'input_tangent_budget'
        assert 'source_checkpoint_template' in c and args.evaluate_from is None
        assert all(isinstance(seed, int) and seed in c['source_seeds'] for seed in training_source_seeds)
        if c.get('exclude_self_transfer'):
            assert c['target_seed'] not in training_source_seeds
    freeze_decoder = c.get('freeze_target_decoder', False)
    assert isinstance(freeze_decoder, bool)
    assert not freeze_decoder or set(c['variants']) <= {'program'}
    w = MultisiteWork(c, args.config, [
        'scripts/train_grammar_member_program.py', 'scripts/run_shift_transfer.py',
        'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/artifacts.py', 'src/ccad/activation_contract.py', 'src/ccad/request_capacity.py'])
    handle, error = None, None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        w.torch, w.device = torch, torch.device(c['device'])
        torch.cuda.set_device(w.device)
        torch.cuda.reset_peak_memory_stats()
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK

        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py',
                  'Pinned dictionary_learning TopK implementation', 'MIT')
        w.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                             transformers=transformers.__version__, threads=2,
                             model=c['model_revision'], hook=c['hook_module_path'])
        for filename in ['model.safetensors', 'config.json', 'tokenizer.json']:
            w.checked(Path(c['model_local_dir'])/filename)
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        tok = transformers.AutoTokenizer.from_pretrained(c['model_local_dir'], local_files_only=True)
        dim = model.config.hidden_size
        source_state = torch.load(w.checked(c['source_checkpoint']), map_location=w.device, weights_only=True)
        initial = torch.load(w.checked(c['target_checkpoint']), map_location=w.device, weights_only=True)
        source_ae = AutoEncoderTopK(dim, source_state['encoder.weight'].shape[0], int(source_state['k'])).to(w.device)
        source_ae.load_state_dict(source_state)
        source_ae.eval().requires_grad_(False)
        target = AutoEncoderTopK(dim, initial['encoder.weight'].shape[0], int(initial['k'])).to(w.device)
        target.load_state_dict(initial)
        parent = Path(c['source_run'])
        prior = torch.load(w.checked(c['prior_checkpoint']), map_location=w.device, weights_only=True)['dictionary'] if c.get('adapt_part') else None
        gate = torch.tensor(np.load(w.checked(parent/f'topk_s{source_seed}_source.npz'))['gate'], device=w.device)
        members = gate.sum(1).nonzero().flatten()
        part_ids = gate[members].argmax(1)
        assert len(members) == 192 and bool((gate.sum(1) <= 1).all())
        source = dict(sae=source_ae, member_ids=members,
                      part_ids=part_ids,
                      decoder=source_ae.decoder.weight[:, members].T)
        training_indices = torch.tensor([i for i, part in enumerate(part_ids.tolist())
                                         if part in training_parts], device=w.device)
        training_members = members[training_indices]
        training_source = dict(sae=source_ae, member_ids=training_members,
                               part_ids=part_ids[training_indices],
                               decoder=source_ae.decoder.weight[:, training_members].T)
        assert len(training_members) == 64*len(training_parts)
        source_pool = {source_seed: training_source}
        source_states = {source_seed: source_state}
        for seed in training_source_seeds:
            if seed == source_seed:
                continue
            state = torch.load(w.checked(c['source_checkpoint_template'].format(seed=seed)),
                               map_location=w.device, weights_only=True)
            ae = AutoEncoderTopK(dim, len(state['encoder.weight']), int(state['k'])).to(w.device)
            ae.load_state_dict(state)
            ae.eval().requires_grad_(False)
            extra_gate = torch.tensor(np.load(w.checked(parent/f'topk_s{seed}_source.npz'))['gate'], device=w.device)
            extra_members, extra_parts = grouped_source_members(extra_gate)
            source_pool[seed] = dict(sae=ae, member_ids=extra_members, part_ids=extra_parts,
                                     decoder=ae.decoder.weight[:, extra_members].T)
            source_states[seed] = state
        active_training_seed = training_source_seeds[0]
        active_training_source = source_pool[active_training_seed]
        source_identity = {}
        if multiple_sources:
            for seed in training_source_seeds:
                sp = source_pool[seed]
                checkpoint_path = str(Path(c['source_checkpoint_template'].format(seed=seed)).resolve())
                gate_path = str((parent/f'topk_s{seed}_source.npz').resolve())
                source_identity[str(seed)] = dict(
                    checkpoint=next(item for item in w.inputs if item['path'] == checkpoint_path),
                    gate=next(item for item in w.inputs if item['path'] == gate_path),
                    members=sp['member_ids'].cpu().tolist(), part_ids=sp['part_ids'].cpu().tolist())
            write(w.run/'training_sources.json', dict(evaluation_source=source_seed, sources=source_identity))
        evaluation_indices = (part_ids == heldout).nonzero().flatten() if c.get('isolated_evaluation') else torch.arange(len(members), device=w.device)
        evaluation_source = dict(sae=source_ae, member_ids=members[evaluation_indices],
                                 part_ids=part_ids[evaluation_indices],
                                 decoder=source['decoder'][evaluation_indices])
        old = np.load(w.checked(parent/f'topk_s{source_seed}_t{c["target_seed"]}_map.npz')) if c.get('fixed_control') else None
        fixed_ids = torch.tensor(old['target_members'], device=w.device) if old is not None else None
        fixed_gate = torch.tensor(old['partition64'], device=w.device) if old is not None else None
        natural = torch.tensor(np.load(w.checked(c['natural_states']))['hidden'], device=w.device)
        natural_fit, natural_eval = natural[:-1024], natural[-1024:]
        panel = json.loads(w.checked(c['panel']).read_text())
        fit_tasks = [c['tasks'][i] for i in training_parts]
        fit_counts = {task: c['new_fit_pairs'] if c.get('adapt_part') == part_names[c['tasks'].index(task)] else c['fit_pairs_per_task'] for task in fit_tasks}
        fit = [r for task in fit_tasks for r in [r for r in panel['rows'] if r['task'] == task and r['split'] == 'fit'][:fit_counts[task]]]
        evaluation = json.loads(w.checked(c['evaluation_panel']).read_text()) if c.get('evaluation_panel') else panel
        evaluation_split = c.get('evaluation_split', 'development')
        rows = [r for task in c['tasks'] for r in [r for r in evaluation['rows'] if r['task'] == task and r['split'] == evaluation_split][:c['eval_pairs_per_task']]]
        assert len(fit) == sum(fit_counts.values())
        assert len(rows) == len(c['tasks'])*c['eval_pairs_per_task']
        assert not {r[key] for r in fit for key in ['sentence_good', 'sentence_bad']} & {r[key] for r in rows for key in ['sentence_good', 'sentence_bad']}
        queries = c['queries']
        write(w.run/'panel.json', dict(fit=fit, rows=rows, queries=queries,
            query_order=list(queries), scope=c['scope']))
        fitting = False
        mode, q = 'none', torch.ones(len(members), device=w.device)
        request = torch.ones(3, device=w.device)
        gain = torch.nn.Parameter(torch.ones(len(members), device=w.device))
        positions, cache = None, {}
        execution_counts = None
        source_access = []
        acquisition_phase = 'normalization'
        source_call_counts = {}

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            idx = torch.arange(len(h), device=w.device)
            x = h[idx, positions]
            cache['input'] = x.detach()
            if mode == 'none':
                return output
            active_source = active_training_source if fitting else evaluation_source
            active_members = active_source['member_ids']
            active_indices = training_indices if fitting else evaluation_indices
            active_q = request[active_source['part_ids']] if multiple_sources and fitting else q[active_indices]
            if mode == 'source':
                delta = -(active_source['sae'].encode(x)[:, active_members]*active_q)@active_source['decoder']
            elif mode == 'fixed':
                delta = -(target.encode(x)[:, fixed_ids]*(fixed_gate@request))@target.decoder.weight[:, fixed_ids].T
            else:
                sp = active_source
                if mode == 'gain':
                    sp = dict(active_source, transport_basis=-(target.encoder.weight@active_source['decoder'].T)*gain[active_indices])
                operation = {'readout': 'raw_reconstruction', 'member_columns': 'input_tangent_budget',
                             'part_member_support': 'input_part_member_support_budget'}.get(mode, c.get('native_operation', 'input_tangent_budget'))
                delta, counts = input_member_delta(x, target, sp, active_q, operation,
                    c['members_per_source']*len(active_members), torch.ones(len(x), device=w.device))
                if execution_counts is not None:
                    execution_counts.append(dict(operation=operation,**counts))
                if operation != 'raw_reconstruction':
                    assert counts['minimum_final_code'] >= -1e-5
            hh = h.clone()
            hh[idx, positions] = x+delta
            cache['delta'] = delta.detach()
            return (hh, *output[1:]) if isinstance(output, tuple) else hh

        handle = model.get_submodule(c['hook_module_path']).register_forward_hook(hook)

        def forward(rr):
            nonlocal positions
            if multiple_sources and fitting and mode == 'source':
                counts = source_call_counts.setdefault(acquisition_phase, {}).setdefault(str(active_training_seed), dict(calls=0, pairs=0))
                counts['calls'] += 1
                counts['pairs'] += len(rr)
            if fitting and mode == 'source' and c.get('adapt_part'):
                active_parts = [part_names[i] for i in range(3) if bool((q[part_ids == i] != 0).any())]
                source_access.append(dict(phase=acquisition_phase, parts=active_parts,
                    rows=[dict(task=r['task'], row_id=r['row_id']) for r in rr]))
            length = max(len(r[key]) for r in rr for key in ['good', 'bad'])
            ids = torch.full((2*len(rr), length), tok.eos_token_id, device=w.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(rr):
                for j, key in enumerate(['good', 'bad']):
                    tokens = row[key]
                    ids[2*i+j, :len(tokens)] = torch.tensor(tokens, device=w.device)
                    mask[2*i+j, :len(tokens)] = 1
            positions = torch.tensor([r['position'] for r in rr], device=w.device).repeat_interleave(2)
            hidden = model.transformer(ids, attention_mask=mask, use_cache=False).last_hidden_state
            logits = model.get_output_embeddings()(hidden[:, :-1])
            per_token = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            total = (per_token*mask[:, 1:]).double().sum(1).reshape(-1, 2)
            margins = total[:, 0]-total[:, 1]
            w.sequence_forwards += len(ids)
            w.token_forwards += int(mask.sum())
            assert float((cache['input'][::2]-cache['input'][1::2]).abs().max()) < .001
            if time.perf_counter()-w.wall_start > c['budget_seconds']:
                raise TimeoutError('Declared grammar program budget exceeded')
            return hidden, mask, margins

        def state_mse(a, b, mask):
            return ((a-b).square()*mask[:, :, None]).sum()/(mask.sum()*a.shape[-1])

        values, quality = {}, []

        @torch.no_grad()
        def evaluate(name, execution):
            nonlocal mode, q, request, fitting, execution_counts
            fitting = False
            mode = execution
            result = np.empty((len(queries), len(rows)), dtype=np.float64)
            for qi, (query, vector) in enumerate(queries.items()):
                execution_counts = [] if c.get('record_execution_counts') else None
                request = torch.tensor(vector, device=w.device, dtype=torch.float32)
                q = request[part_ids]
                for off in range(0, len(rows), c['eval_batch_pairs']):
                    rr = rows[off:off+c['eval_batch_pairs']]
                    _, _, margins = forward(rr)
                    result[qi, off:off+len(rr)] = margins.cpu().numpy()
                    for row, value in zip(rr, margins.cpu().tolist()):
                        w.record(kind='grammar_program', task=row['task'], row_id=row['row_id'],
                            component=row['task']+':'+str(row['row_id']), mode=query, operation=query,
                            method=name, seed=source_seed, target_seed=c['target_seed'], split=evaluation_split,
                            margin=value, accuracy=value > 0)
                if execution_counts is not None:
                    write(w.run/f'{name}__{query}__execution.json',execution_counts)
                execution_counts = None
                w.progress('EVALUATION', method=name, query=query)
            values[name] = result
            np.savez_compressed(w.run/'responses.npz', **values)
            z = target.encode(natural_eval)
            rec = target.decode(z)
            quality.append(dict(method=name, fve=float(1-(rec-natural_eval).square().sum()/
                (natural_eval-natural_eval.mean(0)).square().sum()), l0=float((z > 0).sum(1).float().mean())))
            write(w.run/'natural_quality.json', quality)

        evaluate('none', 'none')
        evaluate('source', 'source')
        evaluate('initial', 'tangent')
        evaluate('readout_initial', 'readout')
        if c.get('native_operation', '').startswith('input_part_'):
            evaluate('initial_member', 'member_columns')
        if prior is not None:
            target.load_state_dict(prior)
            evaluate('prior', 'tangent')
            target.load_state_dict(initial)
        if c.get('fixed_control'):
            evaluate('fixed', 'fixed')
        for name, item in c.get('evaluation_checkpoints', {}).items():
            saved = torch.load(w.checked(item['path'].format(target_seed=c['target_seed'],
                               training_run=c.get('checkpoint_run_by_target', {}).get(str(c['target_seed']), ''))),
                               map_location=w.device, weights_only=True)
            parameter_group = c.get('evaluation_parameter_groups', {}).get(name)
            load_target_parameters(target, initial, saved['dictionary'], parameter_group)
            target.requires_grad_(False)
            evaluate(name, item['execution'])
            changed_keys = {'encoder.weight', 'encoder.bias', 'b_dec'} if parameter_group == 'encoder' else {'decoder.weight'}
            expected = saved['dictionary'] if parameter_group is None else {
                key: saved['dictionary'][key] if key in changed_keys else value for key, value in initial.items()}
            assert all(torch.equal(target.state_dict()[key], value) for key, value in expected.items())
        target.load_state_dict(initial)
        endpoint_parts = torch.eye(3, device=w.device)[training_parts]
        endpoints = torch.cat([endpoint_parts, endpoint_parts.sum(0, keepdim=True)])
        fitting = True
        calibration = [fit[int(i)] for i in np.linspace(0, len(fit)-1, min(24, len(fit)), dtype=int)]
        calibration_batches = [(calibration[off:off+c['batch_pairs']], endpoints) for off in range(0, len(calibration), c['batch_pairs'])]
        if c.get('adapt_part'):
            new_index = part_names.index(c['adapt_part'])
            old_endpoints = torch.eye(3, device=w.device)[[i for i in range(3) if i != new_index]]
            old_endpoints = torch.cat([old_endpoints, old_endpoints.sum(0, keepdim=True)])
            calibration_batches = []
            for ti, task in enumerate(fit_tasks):
                task_fit = [r for r in fit if r['task'] == task]
                selected = [task_fit[int(i)] for i in np.linspace(0, len(task_fit)-1, min(8, len(task_fit)), dtype=int)]
                task_endpoints = torch.eye(3, device=w.device)[new_index:new_index+1] if ti == new_index else old_endpoints
                calibration_batches.extend((selected[off:off+c['batch_pairs']], task_endpoints) for off in range(0, len(selected), c['batch_pairs']))
        scales_by_source = {}
        for calibration_seed in training_source_seeds:
            active_training_seed = calibration_seed
            active_training_source = source_pool[calibration_seed]
            energy = []
            with torch.no_grad():
                for rr, calibration_endpoints in calibration_batches:
                    mode = 'none'
                    clean_h, mask, clean_m = forward(rr)
                    for request in calibration_endpoints:
                        q, mode = request[part_ids], 'source'
                        source_h, _, source_m = forward(rr)
                        energy.append([float(state_mse(source_h, clean_h, mask)),
                                       float((source_m-clean_m).square().mean())])
            scales_by_source[calibration_seed] = np.maximum(np.mean(energy, axis=0), 1e-8)
        scales = scales_by_source[training_source_seeds[0]]
        scale_record = dict(hidden=float(scales[0]), response=float(scales[1]))
        if multiple_sources:
            scale_record.update(by_source={str(seed): dict(hidden=float(value[0]), response=float(value[1]))
                for seed, value in scales_by_source.items()}, normalization_contexts=calibration,
                endpoints=endpoints.cpu().tolist())
        write(w.run/'loss_scales.json', scale_record)
        rng = np.random.default_rng(c['training_seed'])
        requests = rng.random((c['steps'], 3)).astype('float32')
        if heldout is not None:
            requests[:, heldout] = 0
        for step in range(0, c['steps'], 2):
            requests[step] = endpoints[(step//2) % len(endpoints)].cpu().numpy()
        order = rng.permutation(len(fit))
        natural_indices = rng.integers(len(natural_fit), size=(c['steps'], c['natural_batch_states']))
        row_batches = np.array([[order[(step*c['batch_pairs']+j) % len(fit)] for j in range(c['batch_pairs'])] for step in range(c['steps'])])
        if c.get('adapt_part'):
            task_rng = np.random.default_rng(c['training_seed']+20000)
            task_order = np.tile(task_rng.permutation(3), (c['steps']+2)//3)[:c['steps']]
            by_task = {task: np.array([i for i, r in enumerate(fit) if r['task'] == task]) for task in fit_tasks}
            row_batches = np.array([task_rng.choice(by_task[c['tasks'][ti]], c['batch_pairs'], replace=False) for ti in task_order])
            for step, ti in enumerate(task_order):
                if ti == new_index:
                    coefficient = 1. if step % 2 == 0 else requests[step, new_index]
                    requests[step] = 0
                    requests[step, new_index] = coefficient
                else:
                    if step % 2 == 0:
                        requests[step] = old_endpoints[(step//2) % len(old_endpoints)].cpu().numpy()
                    requests[step, new_index] = 0
        member_requests = np.array([vector[part_ids.cpu().numpy()] for vector in requests])
        if c.get('member_requests'):
            member_rng = np.random.default_rng(c['training_seed']+10000)
            member_requests[1::2] = 0
            member_requests[np.ix_(np.arange(1, c['steps'], 2), training_indices.cpu().numpy())] = member_rng.random(
                (c['steps']//2, len(training_indices)))
        source_schedule = balanced_source_schedule(training_source_seeds, c['steps'], c['training_seed'])
        extra_schedule = {}
        source_resume = None
        if multiple_sources:
            member_requests = np.array([vector[source_pool[int(seed)]['part_ids'].cpu().numpy()]
                                       for vector, seed in zip(requests, source_schedule)])
            extra_schedule.update(training_source_seeds=np.array(training_source_seeds),
                                  source_seed_by_step=source_schedule)
            for seed in training_source_seeds:
                extra_schedule[f'source_{seed}_members'] = source_pool[seed]['member_ids'].cpu().numpy()
                extra_schedule[f'source_{seed}_parts'] = source_pool[seed]['part_ids'].cpu().numpy()
                extra_schedule[f'source_{seed}_loss_scales'] = scales_by_source[seed]
            coverage = {str(seed): dict(steps=int((source_schedule == seed).sum()),
                endpoint_steps=int((source_schedule[::2] == seed).sum()),
                continuous_steps=int((source_schedule[1::2] == seed).sum()),
                unique_rows=len(np.unique(row_batches[source_schedule == seed])),
                rows=np.unique(row_batches[source_schedule == seed]).tolist()) for seed in training_source_seeds}
            write(w.run/'training_source_coverage.json', dict(fit_pairs=len(fit), sources=coverage,
                schedule_seed=c['training_seed']+30000))
            source_resume = dict(seeds=training_source_seeds, identity=source_identity,
                schedule=torch.from_numpy(source_schedule), requests=torch.from_numpy(requests),
                row_batches=torch.from_numpy(row_batches), natural=torch.from_numpy(natural_indices),
                scales=torch.from_numpy(np.array([scales_by_source[seed] for seed in training_source_seeds])))
        np.savez_compressed(w.run/'training_schedule.npz', requests=requests, rows=order, row_batches=row_batches, natural=natural_indices,
                            source_members=training_members.cpu().numpy(), source_parts=part_ids[training_indices].cpu().numpy(),
                            member_requests=member_requests, **extra_schedule)
        for variant in c['variants']:
            acquisition_phase = variant
            fitting = True
            target.load_state_dict(prior if variant == 'program_warm' else initial)
            is_program = variant.startswith('program')
            target.requires_grad_(is_program or variant == 'whole')
            if freeze_decoder:
                freeze_decoder_for_training(target)
            gain.data.fill_(1.)
            gain.requires_grad_(variant == 'gain')
            parameters = [gain] if variant == 'gain' else list(target.encoder.parameters()) if freeze_decoder else list(target.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=c['gain_lr'] if variant == 'gain' else c['dictionary_lr'], weight_decay=0.)
            assert len(optimizer.state) == 0
            first_step = 0
            if variant in c.get('resume_checkpoints', {}):
                saved = torch.load(w.checked(c['resume_checkpoints'][variant]), map_location=w.device, weights_only=True)
                target.load_state_dict(saved['dictionary'])
                gain.data.copy_(saved['gain'])
                optimizer.load_state_dict(saved['optimizer'])
                first_step = saved['step']
                if multiple_sources:
                    previous = saved['training_sources']
                    assert previous['seeds'] == source_resume['seeds'] and previous['identity'] == source_resume['identity']
                    assert all(torch.equal(previous[key].cpu(), source_resume[key])
                               for key in ['schedule', 'requests', 'row_batches', 'natural', 'scales'])
            for step in range(first_step, c['steps']):
                active_training_seed = int(source_schedule[step])
                active_training_source = source_pool[active_training_seed]
                scales = scales_by_source[active_training_seed]
                request = torch.tensor(requests[step], device=w.device)
                if variant == 'whole':
                    request = endpoints[-1]
                if heldout is not None:
                    assert request[heldout] == 0 and bool((part_ids[training_indices] != heldout).all())
                q = request[part_ids]
                if c.get('member_requests') and variant == 'program':
                    q = torch.tensor(member_requests[step], device=w.device, dtype=torch.float32)
                    if heldout is not None:
                        assert bool((q[part_ids == heldout] == 0).all())
                rr = [fit[i] for i in row_batches[step]]
                mode = 'source'
                with torch.no_grad():
                    th, mask, tm = forward(rr)
                mode = 'gain' if variant == 'gain' else 'tangent'
                sh, _, sm = forward(rr)
                hidden_loss = state_mse(sh, th, mask)/scales[0]
                response_loss = (sm-tm).square().mean()/scales[1]
                natural_h = natural_fit[natural_indices[step]]
                reconstruction = (target(natural_h)-natural_h).square().mean()/natural_h.square().mean()
                loss = (1-c['response_weight'])*hidden_loss+c['response_weight']*response_loss+c['reconstruction_weight']*reconstruction
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                with torch.no_grad():
                    if target.decoder.weight.requires_grad and (is_program or variant == 'whole'):
                        target.decoder.weight.div_(target.decoder.weight.norm(dim=0).clamp_min(1e-10))
                    gain.clamp_(min=0)
                assert bool(torch.isfinite(loss))
                if (step+1) % c['checkpoint_every'] == 0 or step+1 == c['steps']:
                    torch.save(dict(dictionary=target.state_dict(), gain=gain.detach(),
                        optimizer=optimizer.state_dict(), step=step+1,
                        **({'training_sources': source_resume} if multiple_sources else {})), w.run/f'{variant}_step{step+1}.pt')
                    w.progress('TRAINING', method=variant, step=step+1, loss=float(loss.detach()),
                               response=float(response_loss.detach()), reconstruction=float(reconstruction.detach()),
                               peak_cuda_bytes=torch.cuda.max_memory_allocated())
                if step+1 in c.get('learning_curve_steps', []):
                    evaluate(f'{variant}_step{step+1}', 'tangent')
                    fitting = True
            if freeze_decoder:
                state = target.state_dict()
                unchanged = all(torch.equal(state[key], initial[key]) for key in ('decoder.weight', 'b_dec'))
                updated = any(not torch.equal(state[key], initial[key]) for key in ('encoder.weight', 'encoder.bias'))
                w.checks[f'decoder_frozen_{variant}'] = unchanged
                w.checks[f'encoder_updated_{variant}'] = updated
                assert unchanged and updated
            evaluate(variant, 'gain' if variant == 'gain' else 'tangent')
            if variant == 'program':
                evaluate('readout_program', 'readout')
        w.checks['source_frozen'] = all(
            not any(parameter.requires_grad for parameter in source_pool[seed]['sae'].parameters()) and
            all(torch.equal(source_pool[seed]['sae'].state_dict()[key], value) for key, value in state.items())
            for seed, state in source_states.items())
        w.checks['base_model_frozen'] = all(not p.requires_grad for p in model.parameters())
        if multiple_sources:
            write(w.run/'training_source_calls.json', source_call_counts)
            w.checks['training_source_calls_match_steps'] = sum(
                value['calls'] for value in source_call_counts.get('program', {}).values()) == c['steps']-first_step
        if c.get('adapt_part'):
            new_task = c['tasks'][new_index]
            new_rows = {(r['task'], r['row_id']) for access in source_access if c['adapt_part'] in access['parts'] for r in access['rows']}
            assert len(new_rows) <= c['new_fit_pairs'] and all(task == new_task for task, _ in new_rows)
            write(w.run/'source_response_access.json', dict(calls=source_access,
                new_context_count=len(new_rows), pair_response_evaluations=sum(len(a['rows']) for a in source_access),
                new_pair_response_evaluations=sum(len(a['rows']) for a in source_access if c['adapt_part'] in a['parts'])))
        write(w.run/'method_summary.json', dict(quality=quality, source_members=members.cpu().tolist(),
            source_parts=part_ids.cpu().tolist(), same_rule='run_shift_transfer.input_member_delta',
            heldout_part=c.get('heldout_part'), training_tasks=fit_tasks,
            training_source_members=training_members.cpu().tolist(),
            evaluation_source_members=members[evaluation_indices].cpu().tolist(),
            source_seed=source_seed, target_seed=c['target_seed'],
            training_source_seeds=training_source_seeds,
            training_source_identity=source_identity,
            training_variants=c['variants'], requested_steps=c['steps'],
            adaptation_part=c.get('adapt_part'), fitting_pairs_by_task=fit_counts,
            prior_training_steps=c.get('prior_training_steps'),
            fresh_optimizer_for_adaptation=bool(c.get('adapt_part')),
            frozen_evaluation_methods=list(c.get('evaluation_checkpoints', {}))))
    except Exception:
        error = traceback.format_exc()
    finally:
        if handle is not None:
            handle.remove()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
