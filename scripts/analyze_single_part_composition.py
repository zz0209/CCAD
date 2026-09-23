import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 2e-5
METHODS = ['single_program', 'mixed_program', 'initial', 'readout_initial',
           'single_readout_program', 'mixed_readout_program', 'source_additive',
           'source', 'none']
FAMILIES = {
    'primary_binary4': ['full', 'verb_number', 'verb_gender', 'number_gender'],
    'strength4': ['participation_a', 'participation_b', 'participation_c', 'half'],
    'parts3': ['verb', 'number', 'gender'],
}
FAMILIES['all_unseen8'] = FAMILIES['primary_binary4'] + FAMILIES['strength4']
METRICS = ['response_nrmse', 'accuracy', 'source_answer_agreement',
           'source_effect_sign_agreement', 'interaction_rms',
           'interaction_reconstruction_rmse']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}


def maximum_difference(left, right, label):
    assert left.shape == right.shape, (label, left.shape, right.shape)
    difference = float(np.max(np.abs(left - right)))
    assert difference <= TOLERANCE, (label, difference, TOLERANCE)
    return difference


def correct_measurement_identity(run, panel, config, schedule, requests, measured):
    # 以真实词元和干预位置识别句对，跨阶段保留相同测量的共享身份。
    scales = read_json(run / 'loss_scales.json')
    seed = config.get('source_seed', 1)
    assert config.get('training_source_seeds', [seed]) == [seed], run
    assert not config.get('member_requests') and not config.get('adapt_part'), run
    parts = schedule['source_parts']
    assert parts.ndim == 1 and set(parts.tolist()) == {0, 1, 2}, run
    corrected = {phase: set() for phase in ['normalization', 'program', 'evaluation']}
    original_keys = {phase: set() for phase in corrected}

    def add(phase, row, vector):
        member_request = tuple(np.asarray(vector, dtype=np.float32)[parts].tolist())
        corrected[phase].add((seed, tuple(row['good']), tuple(row['bad']),
                              row['position'], member_request))
        original_keys[phase].add((seed, row['task'], row['row_id'], member_request))

    for row in scales['normalization_contexts']:
        for vector in scales['endpoints']:
            add('normalization', row, vector)
    for vector, batch in zip(requests, schedule['row_batches']):
        for index in batch:
            add('program', panel['fit'][int(index)], vector)
    for row in panel['rows']:
        for vector in panel['queries'].values():
            add('evaluation', row, vector)
    fit = corrected['normalization'] | corrected['program']
    old_fit = original_keys['normalization'] | original_keys['program']
    old_total = old_fit | original_keys['evaluation']
    assert len(fit) == len(old_fit) == measured['distinct_fitting_pair_requests'], run
    assert len(old_total) == measured['distinct_pair_requests_total'], run
    for phase in corrected:
        assert len(original_keys[phase]) == measured['distinct_pair_requests_by_phase'][phase], (run, phase)
    total = fit | corrected['evaluation']
    corrected_content = dict(measured)
    corrected_content.update(distinct_fitting_pair_requests=len(fit),
                             distinct_pair_requests_total=len(total),
                             distinct_pair_requests_by_phase={**measured['distinct_pair_requests_by_phase'],
                                 **{phase: len(values) for phase, values in corrected.items()}},
                             distinct_measurement_unit='source seed, good tokens, bad tokens, intervention position, exact source member request')
    correction = {
        'reason': 'The original task/row_id key was reused between fit and confirmation panels. Distinct source measurements are recomputed from token content and intervention position; original run files and actual call counts remain unchanged.',
        'original_total': measured['distinct_pair_requests_total'], 'corrected_total': len(total),
        'original_fitting': measured['distinct_fitting_pair_requests'], 'corrected_fitting': len(fit),
        'evaluation': len(corrected['evaluation']), 'fit_evaluation_overlap': len(fit & corrected['evaluation']),
        'original_key_fit_evaluation_overlap': len(old_fit & original_keys['evaluation']),
        'cache_validation': 'Cache validation repeats a program measurement and contributes no distinct measurement beyond the saved training requests.',
        'sources': {name: identity(run / name) for name in
                    ['panel.json', 'training_schedule.npz', 'loss_scales.json', 'source_measurement_cost.json']}}
    return corrected_content, correction


def read_run(run, family):
    assert read_json(run / 'status.json')['status'] == 'PASS', run
    panel = read_json(run / 'panel.json')
    config = read_json(run / 'config.resolved.json')
    assert config['training_request_family'] == family, (run, family)
    assert config['normalization_request_family'] == 'single_parts', run
    order = panel['query_order']
    assert len(order) == len(set(order)) and set(order) == set(panel['queries']), run
    with np.load(run / 'responses.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name in ['source', 'none', 'initial', 'readout_initial', 'program', 'readout_program']:
        assert arrays[name].shape == (len(order), len(panel['rows'])), (run, name)
        assert np.isfinite(arrays[name]).all(), (run, name)
    assert len({(row['task'], row['row_id']) for row in panel['rows']}) == len(panel['rows']), run
    with np.load(run / 'training_schedule.npz', allow_pickle=False) as schedule:
        requests = schedule['requests'].copy()
        matched_schedule = {name: schedule[name].copy() for name in
                            ['rows', 'row_batches', 'natural', 'source_members', 'source_parts']}
    assert requests.ndim == 2 and requests.shape[1] == 3, run
    assert len(requests) == config['steps'], (run, requests.shape, config['steps'])
    singleton = ((requests == 0) | (requests == 1)).all(1) & (requests.sum(1) == 1)
    assert singleton.all() if family == 'single_parts' else (~singleton).any(), run
    costs = {}
    for name in ['source_measurement_cost.json', 'environment.json', 'metrics.summary.json']:
        path = run / name
        costs[name] = {'identity': identity(path), 'content': read_json(path)} if path.exists() else {'missing': True}
    measurement = costs['source_measurement_cost.json']
    assert 'content' in measurement, run
    measurement['corrected_content'], measurement['identity_correction'] = correct_measurement_identity(
        run, panel, config, matched_schedule, requests, measurement['content'])
    files = ['status.json', 'panel.json', 'config.resolved.json', 'responses.npz', 'training_schedule.npz']
    return dict(run=str(run.resolve()), panel=panel, config=config, arrays=arrays,
                identities={name: identity(run / name) for name in files}, costs=costs,
                matched_schedule=matched_schedule,
                training_requests={'total': len(requests), 'singleton': int(singleton.sum()),
                                   'distinct': np.unique(requests, axis=0).tolist()})


def prepare(single_runs, mixed_runs, base):
    assert len(single_runs) == len(mixed_runs) > 0
    all_values, records, checks, targets = [], [], [], []
    reference_panel = None
    matched_keys = ['target_seed', 'source_checkpoint', 'target_checkpoint', 'model_revision',
                    'tasks', 'steps', 'training_seed', 'fit_pairs_per_task', 'batch_pairs',
                    'dictionary_lr', 'response_weight', 'reconstruction_weight',
                    'members_per_source', 'natural_batch_states', 'normalization_request_family']
    for single_path, mixed_path in zip(single_runs, mixed_runs):
        single = read_run(single_path, 'single_parts')
        mixed = read_run(mixed_path, 'mixed')
        for key in matched_keys:
            assert single['config'][key] == mixed['config'][key], (key, single_path, mixed_path)
        for key in single['matched_schedule']:
            assert np.array_equal(single['matched_schedule'][key], mixed['matched_schedule'][key]), (key, single_path, mixed_path)
        for key in ['fit', 'rows', 'queries', 'query_order']:
            assert single['panel'][key] == mixed['panel'][key], (key, single_path, mixed_path)
        panel = single['panel']
        assert panel['queries'] == base['queries'], single_path
        if reference_panel is not None:
            for key in ['fit', 'rows', 'queries', 'query_order']:
                assert panel[key] == reference_panel[key], (key, single_path)
        else:
            reference_panel = panel
        query_order = panel['query_order']
        vectors = np.array([panel['queries'][key] for key in query_order], dtype=np.float64)
        unit_ids = [query_order.index(name) for name in FAMILIES['parts3']]
        assert np.array_equal(vectors[unit_ids], np.eye(3))
        a, b = single['arrays'], mixed['arrays']
        pair_checks = {name: maximum_difference(a[name], b[name], (single_path, mixed_path, name))
                       for name in ['source', 'none', 'initial', 'readout_initial']}
        pair_checks['none_request_invariance'] = maximum_difference(
            a['none'], np.broadcast_to(a['none'][0], a['none'].shape), (single_path, 'none'))
        if all_values:
            for name in ['source', 'none']:
                pair_checks['across_targets_' + name] = maximum_difference(
                    a[name], all_values[0][METHODS.index(name)], (single_path, name))
        additive = a['none'] + vectors @ (a['source'][unit_ids] - a['none'][unit_ids])
        all_values.append(np.stack([a['program'], b['program'], a['initial'], a['readout_initial'],
                                    a['readout_program'], b['readout_program'], additive,
                                    a['source'], a['none']]))
        targets.append(single['config']['target_seed'])
        checks.append({'target_seed': targets[-1], 'max_absolute_differences': pair_checks,
                       'matched_training_schedule': list(single['matched_schedule'])})
        excluded = ['arrays', 'panel', 'matched_schedule']
        records.append({'single': {k: v for k, v in single.items() if k not in excluded},
                        'mixed': {k: v for k, v in mixed.items() if k not in excluded}})
    assert len(targets) == len(set(targets)), targets
    return np.stack(all_values, axis=1), reference_panel, targets, records, checks


def measure(values, interactions, groups, query_ids):
    # 所有方法、请求和固定 target 共用同一组句对抽样。
    shape = (len(METHODS), values.shape[1], len(groups), len(query_ids), len(METRICS))
    result = np.empty(shape, dtype=np.float64)
    for gi, rows in enumerate(groups.values()):
        for fi, requests in enumerate(query_ids.values()):
            block = values[:, :, requests][:, :, :, rows]
            source = block[METHODS.index('source')]
            clean = block[METHODS.index('none')]
            energy = np.square(source - clean).sum((1, 2))
            assert (energy > 0).all(), ('zero_source_energy', gi, fi)
            result[:, :, gi, fi, 0] = np.sqrt(np.square(block - source).sum((2, 3)) / energy)
            result[:, :, gi, fi, 1] = (block > 0).mean((2, 3))
            result[:, :, gi, fi, 2] = ((block > 0) == (source > 0)).mean((2, 3))
            result[:, :, gi, fi, 3] = (np.sign(block - clean) == np.sign(source - clean)).mean((2, 3))
            residual = interactions[:, :, requests][:, :, :, rows]
            source_residual = residual[METHODS.index('source')]
            result[:, :, gi, fi, 4] = np.sqrt(np.square(residual).mean((2, 3)))
            result[:, :, gi, fi, 5] = np.sqrt(np.square(residual - source_residual).mean((2, 3)))
    return result


def summarize(point, bootstrap, targets, tasks):
    summary = {}
    for mi, method in enumerate(METHODS):
        summary[method] = {}
        for fi, family in enumerate(FAMILIES):
            summary[method][family] = {
                metric: {'mean': float(point[mi, :, :, fi, ki].mean()),
                         'ci95': np.quantile(bootstrap[:, mi, fi, ki], [.025, .975]).tolist(),
                         'by_target': {str(seed): float(point[mi, ti, :, fi, ki].mean())
                                       for ti, seed in enumerate(targets)},
                         'by_grammar': {task: float(point[mi, :, gi, fi, ki].mean())
                                        for gi, task in enumerate(tasks)}}
                for ki, metric in enumerate(METRICS)}
    comparisons = {}
    focal = METHODS.index('single_program')
    for other in METHODS[:7]:
        if other == 'single_program':
            continue
        oi = METHODS.index(other)
        comparisons['single_program_minus_' + other] = {}
        for fi, family in enumerate(FAMILIES):
            differences = bootstrap[:, focal, fi, 0] - bootstrap[:, oi, fi, 0]
            comparisons['single_program_minus_' + other][family] = {
                'mean': float((point[focal, :, :, fi, 0] - point[oi, :, :, fi, 0]).mean()),
                'ci95': np.quantile(differences, [.025, .975]).tolist(),
                'one_sided_upper95': float(np.quantile(differences, .95))}
    return summary, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--mixed-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    assert args.bootstrap >= 20, args.bootstrap
    config_path = ROOT / 'configs/cc23_single_parts_base.json'
    base = read_json(config_path)
    values, panel, targets, inputs, checks = prepare(args.single_runs, args.mixed_runs, base)
    tasks = base['tasks']
    assert set(row['task'] for row in panel['rows']) == set(tasks)
    groups = {task: np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task]) for task in tasks}
    assert all(len(rows) > 0 for rows in groups.values())
    query_ids = {name: np.array([panel['query_order'].index(q) for q in queries]) for name, queries in FAMILIES.items()}
    vectors = np.array([panel['queries'][name] for name in panel['query_order']])
    clean = values[METHODS.index('none')]
    # 每个方法用自身的 singleton 响应构造加性预测，以测量其非加性响应。
    additive = clean[None] + np.einsum('qk,mtkr->mtqr', vectors, values[:, :, query_ids['parts3']] - clean[:, query_ids['parts3']])
    interactions = values - additive
    point = measure(values, interactions, groups, query_ids)
    request_ids = {name: np.array([index]) for index, name in enumerate(panel['query_order'])}
    per_request = measure(values, interactions, groups, request_ids)
    random = np.random.default_rng(92323)
    bootstrap = np.empty((args.bootstrap, len(METHODS), len(FAMILIES), len(METRICS)))
    for iteration in range(args.bootstrap):
        draw = {task: random.choice(rows, len(rows), replace=True) for task, rows in groups.items()}
        bootstrap[iteration] = measure(values, interactions, draw, query_ids).mean((1, 2))
        if (iteration + 1) % 100 == 0 or iteration + 1 == args.bootstrap:
            print(json.dumps({'bootstrap_completed': iteration + 1, 'bootstrap_total': args.bootstrap}), flush=True)
    summary, comparisons = summarize(point, bootstrap, targets, tasks)
    primary = comparisons['single_program_minus_mixed_program']['primary_binary4']
    margin = base['noninferiority_margin']
    decision = {'margin': margin, 'one_sided_upper95_below_margin': primary['one_sided_upper95'] < margin,
                'two_sided_ci95_upper_below_margin': primary['ci95'][1] < margin,
                'primary_difference': primary,
                'meaning': 'Single-part response nRMSE is within the specified mixed-training margin on the fixed source, target cohort and grammar panel when the paired upper bound is below the margin.'}
    costs = {'runs': [], 'summed_observed_run_totals': {}, 'missing': []}
    totals = ['wall_seconds', 'process_cpu_seconds', 'sequence_forwards', 'token_forwards']
    for item in inputs:
        for family in ['single', 'mixed']:
            run = item[family]
            costs['runs'].append({'run': run['run'], 'family': family, 'costs': run['costs']})
            for name, asset in run['costs'].items():
                if asset.get('missing'):
                    costs['missing'].append({'run': run['run'], 'file': name})
    for key in totals:
        observations = [run['costs']['metrics.summary.json']['content'][key] for run in costs['runs']
                        if 'content' in run['costs']['metrics.summary.json'] and key in run['costs']['metrics.summary.json']['content']]
        costs['summed_observed_run_totals'][key] = {'value': sum(observations), 'observed_runs': len(observations), 'total_runs': len(costs['runs'])}
    costs['by_training_family'] = {}
    for family in ['single', 'mixed']:
        selected = [run for run in costs['runs'] if run['family'] == family]
        measurements = [run['costs']['source_measurement_cost.json']['corrected_content'] for run in selected
                        if 'corrected_content' in run['costs']['source_measurement_cost.json']]
        teacher = {'observed_runs': len(measurements), 'total_runs': len(selected),
                   'summed_run_distinct_fitting_pair_requests': sum(m['distinct_fitting_pair_requests'] for m in measurements),
                   'summed_run_distinct_pair_requests_total': sum(m['distinct_pair_requests_total'] for m in measurements),
                   'actual_source_calls_by_phase': {}, 'cache': {}}
        for measurement in measurements:
            for phase, sources in measurement['actual_calls'].items():
                counts = teacher['actual_source_calls_by_phase'].setdefault(phase, dict(calls=0, pairs=0, sequences=0, tokens=0))
                for source in sources.values():
                    for key in counts:
                        counts[key] += source[key]
        for key in ['hits', 'misses', 'validation_calls', 'stored_bytes']:
            teacher['cache'][key] = sum(m['teacher_cache'][key] for m in measurements)
        teacher['scope'] = 'Distinct measurements are counted within each run and then summed; repeated measurements across runs are retained as acquisition cost. Actual calls include normalization, cache validation and evaluation separately.'
        teacher['observed_run_totals'] = {
            key: sum(run['costs']['metrics.summary.json']['content'][key] for run in selected
                     if 'content' in run['costs']['metrics.summary.json'] and key in run['costs']['metrics.summary.json']['content'])
            for key in totals}
        costs['by_training_family'][family] = teacher
    output = {'written_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
              'script': identity(Path(__file__)), 'base_config': identity(config_path),
              'methods': METHODS, 'target_seeds': targets, 'tasks': tasks,
              'families': FAMILIES, 'metrics': METRICS, 'inputs': inputs, 'checks': checks,
              'bootstrap': args.bootstrap, 'bootstrap_seed': 92323,
              'inference': 'Sentence pairs jointly resampled within each fixed grammar. All requests, methods and fixed targets share every draw. Intervals condition on this source, model, grammar set and target cohort.',
              'source_additive_role': 'Prediction from source singleton measurements; no target-member execution is defined for this control.',
              'interaction_definition': 'R(q) - [R(clean) + sum_k q_k (R(unit_k)-R(clean))], using each method own singleton responses and the common clean response. Reconstruction RMSE compares each method interaction residual to the source residual.',
              'sign_definition': 'Exact sign of response minus clean; zero effects remain a separate sign.',
              'summary': summary, 'comparisons': comparisons, 'noninferiority': decision,
              'per_request': {method: {request: {metric: float(per_request[mi, :, :, qi, ki].mean())
                              for ki, metric in enumerate(METRICS)}
                              for qi, request in enumerate(panel['query_order'])}
                              for mi, method in enumerate(METHODS)},
              'samples_by_grammar': {task: len(rows) for task, rows in groups.items()}, 'cost': costs}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'ARRAYS.npz', responses=values, additive=additive,
                        interaction_residuals=interactions, source_answer=values[-2] > 0,
                        source_effect_sign=np.sign(values[-2] - values[-1]), point_metrics=point,
                        per_request_metrics=per_request, bootstrap_metrics=bootstrap, request_vectors=vectors)
    (args.output / 'PANEL.json').write_text(json.dumps(panel, indent=2) + '\n', encoding='utf-8')
    output['array_axes'] = {'responses': ['method', 'target', 'request', 'row'],
                           'point_metrics': ['method', 'target', 'grammar', 'family', 'metric'],
                           'per_request_metrics': ['method', 'target', 'grammar', 'request', 'metric'],
                           'bootstrap_metrics': ['bootstrap', 'method', 'family', 'metric']}
    output['outputs'] = {name: identity(args.output / name) for name in ['ARRAYS.npz', 'PANEL.json']}
    (args.output / 'RESULTS.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'output': str(args.output), 'noninferiority': decision}), flush=True)


if __name__ == '__main__':
    main()
