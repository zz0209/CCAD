import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_run(run):
    assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
    config = json.loads((run / 'config.resolved.json').read_text())
    panel = json.loads((run / 'panel.json').read_text())
    with np.load(run / 'responses.npz') as asset:
        arrays = {key: asset[key].copy() for key in asset.files}
    positions = {(row['task'], row['row_id']): i for i, row in enumerate(panel['rows'])}
    assert len(positions) == len(panel['rows'])
    with (run / 'metrics.raw.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    records = [record for record in records if record['kind'] == 'grammar_program']
    orders = {}
    for seed in config['target_seeds']:
        order = list(dict.fromkeys(record['operation'] for record in records
                                  if record['target_seed'] == seed and record['method'] == 'none'))
        assert set(order) == set(panel['queries'])
        assert order == panel['query_order']
        orders[seed] = order
    order = orders[config['target_seeds'][0]]
    assert all(value == order for value in orders.values())
    seen = set()
    for record in records:
        key = f't{record["target_seed"]}_{record["method"]}'
        qi = order.index(record['operation'])
        ri = positions[record['task'], record['row_id']]
        identity = (key, qi, ri)
        assert identity not in seen, identity
        seen.add(identity)
        assert arrays[key][qi, ri] == record['margin'], identity
    assert len(seen) == len(arrays) * len(order) * len(positions)
    assert all(value.shape == (len(order), len(positions)) and np.isfinite(value).all()
               for value in arrays.values())
    with np.load(run / 'training_schedule.npz') as asset:
        schedule = {key: asset[key].copy() for key in ('requests', 'row_batches', 'targets')}
    assert not set(config['heldout_targets']) & set(schedule['targets'].tolist())
    if config.get('evaluation_only'):
        assert all(len(value) == 0 for value in schedule.values())
        rule = json.loads((run / 'rule_identity.json').read_text())
        assert rule['updates_performed'] == 0 and rule['dictionary_parameters_updated'] == 0
    else:
        assert set(schedule['targets'].tolist()) == set(config['training_targets'])
    identity = dict(run=run.as_posix(), records_verified=len(records),
                    response_sha256=hashlib.sha256((run / 'responses.npz').read_bytes()).hexdigest(),
                    raw_metrics_sha256=hashlib.sha256((run / 'metrics.raw.jsonl').read_bytes()).hexdigest(),
                    training_target_counts={str(seed): int((schedule['targets'] == seed).sum())
                                            for seed in config['training_targets']})
    return config, panel, order, arrays, schedule, identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=4000)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    assert args.bootstrap > 0
    loaded = [read_run(run) for run in args.runs]
    config, panel, queries, arrays, schedule, _ = loaded[0]
    arrays = dict(arrays)
    replay = []
    tolerance = 2e-5
    for other_config, other_panel, other_queries, other_arrays, other_schedule, identity in loaded[1:]:
        assert panel['fit'] == other_panel['fit'] and panel['rows'] == other_panel['rows']
        assert panel['queries'] == other_panel['queries'] and queries == other_queries
        for field in ('tasks', 'primary_requests', 'target_seeds', 'training_targets', 'heldout_targets'):
            assert config[field] == other_config[field], field
        for key in schedule:
            assert np.array_equal(schedule[key], other_schedule[key]), key
        for key, value in other_arrays.items():
            if key in arrays:
                difference = float(np.max(np.abs(arrays[key] - value)))
                np.testing.assert_allclose(arrays[key], value, atol=tolerance, rtol=0)
                replay.append(dict(run=identity['run'], array=key, max_absolute_difference=difference,
                                   exact=bool(np.array_equal(arrays[key], value))))
            else:
                arrays[key] = value
    seeds = config['target_seeds']
    methods_by_seed = [{key[len(f't{seed}_'):] for key in arrays if key.startswith(f't{seed}_')}
                       for seed in seeds]
    assert all(value == methods_by_seed[0] for value in methods_by_seed)
    methods = sorted(methods_by_seed[0] - {'none', 'source'})
    source = np.array([arrays[f't{seed}_source'] for seed in seeds])
    clean = np.array([arrays[f't{seed}_none'] for seed in seeds])
    target = np.array([[arrays[f't{seed}_{method}'] for seed in seeds] for method in methods])
    for value in source:
        np.testing.assert_allclose(value, source[0], atol=tolerance, rtol=0)
    for value in clean:
        np.testing.assert_allclose(value, clean[0], atol=tolerance, rtol=0)
    tasks = config['tasks']
    groups = {task: np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task])
              for task in tasks}
    families = dict(full=['full'], parts=['verb', 'number', 'gender'], primary=config['primary_requests'])
    assert len(families['primary']) == 7
    query_ids = {name: np.array([queries.index(query) for query in names]) for name, names in families.items()}
    cohorts = {name: [seed for seed in values if seed in seeds]
               for name, values in {'training': config['training_targets'], 'heldout': config['heldout_targets']}.items()}
    cohorts = {name: values for name, values in cohorts.items() if values}
    cohort_indices = {name: np.array([seeds.index(seed) for seed in values]) for name, values in cohorts.items()}

    def measure(draws, selected_seeds):
        result = np.empty((len(methods), len(tasks), len(families)))
        for ti, task in enumerate(tasks):
            documents = draws[task]
            for fi, family in enumerate(families):
                qi = query_ids[family]
                reference = source[np.ix_(selected_seeds, qi, documents)]
                baseline = clean[np.ix_(selected_seeds, qi, documents)]
                denominator = np.square(reference - baseline).sum((1, 2))
                assert (denominator > 0).all(), (task, family)
                predicted = np.take(np.take(np.take(target, selected_seeds, axis=1), qi, axis=2), documents, axis=3)
                result[:, ti, fi] = np.sqrt(np.square(predicted - reference).sum((2, 3)) / denominator).mean(1)
        return result

    point = {name: measure(groups, indices) for name, indices in cohort_indices.items()}
    by_seed = {str(seed): measure(groups, np.array([si])) for si, seed in enumerate(seeds)}
    rng = np.random.default_rng(9261)
    bootstrap = {name: np.empty((args.bootstrap, len(methods), len(tasks), len(families))) for name in cohorts}
    for iteration in range(args.bootstrap):
        draws = {task: rng.choice(ids, len(ids), replace=True) for task, ids in groups.items()}
        for name, indices in cohort_indices.items():
            selected_seeds = rng.choice(indices, len(indices), replace=True)
            bootstrap[name][iteration] = measure(draws, selected_seeds)
        if (iteration + 1) % 1000 == 0:
            print(json.dumps(dict(bootstrap_completed=iteration + 1, bootstrap_total=args.bootstrap)), flush=True)
    summary = {}
    for name, values in point.items():
        summary[name] = {}
        for mi, method in enumerate(methods):
            summary[name][method] = {
                family: dict(mean=float(values[mi, :, fi].mean()),
                    ci=np.quantile(bootstrap[name][:, mi, :, fi].mean(1), [.025, .975]).tolist(),
                    by_task={task: float(values[mi, ti, fi]) for ti, task in enumerate(tasks)},
                    by_target={str(seed): float(by_seed[str(seed)][mi, :, fi].mean()) for seed in cohorts[name]},
                    by_target_task={str(seed): {task: float(by_seed[str(seed)][mi, ti, fi])
                                              for ti, task in enumerate(tasks)} for seed in cohorts[name]})
                for fi, family in enumerate(families)}
    comparisons = {}
    for name, values in point.items():
        comparisons[name] = {}
        for focal in ('shared', 'source_columns'):
            if focal not in methods:
                continue
            fi_method = methods.index(focal)
            for other in methods:
                if other == focal or (focal == 'source_columns' and other == 'shared'):
                    continue
                oi = methods.index(other)
                comparisons[name][focal + '_minus_' + other] = {
                    family: dict(mean=float((values[fi_method, :, fi] - values[oi, :, fi]).mean()),
                        ci=np.quantile((bootstrap[name][:, fi_method, :, fi] - bootstrap[name][:, oi, :, fi]).mean(1), [.025, .975]).tolist(),
                        by_task={task: float(values[fi_method, ti, fi] - values[oi, ti, fi]) for ti, task in enumerate(tasks)},
                        by_target={str(seed): float((by_seed[str(seed)][fi_method, :, fi] - by_seed[str(seed)][oi, :, fi]).mean())
                                   for seed in cohorts[name]})
                    for fi, family in enumerate(families)}
    source_effects = {task: {family: dict(
        rms=float(np.sqrt(np.square((source[0] - clean[0])[np.ix_(qi, ids)]).mean())),
        source_accuracy=float((source[0][np.ix_(qi, ids)] > 0).mean()),
        clean_accuracy=float((clean[0][np.ix_(qi, ids)] > 0).mean()))
        for family, qi in query_ids.items()} for task, ids in groups.items()}
    output = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=[value[-1] for value in loaded], source_seed=config.get('source_seed', 1),
        bootstrap=args.bootstrap, bootstrap_seed=9261,
        metric='For each target and task, pool squared prediction error and source-minus-clean squared effect over all family requests and paired sentences; take their square-root ratio; average targets and tasks equally.',
        inference='Paired sentence resampling within each fixed grammar and replacement resampling of the listed targets within each cohort. All methods and requests share each draw. A one-target cohort conditions on that target. Intervals describe the specified source, model and data split; source-target directions sharing SAE seeds remain dependent.',
        evaluation_split=config.get('evaluation_split', 'development'),
        common_controls='Duplicate arrays across runs are verified and used once; runs do not add independent observations.',
        initialization='Analytic uses the identical four-step recurrence with identity weight and original tangent initialization; shared learns its weight.',
        comparison_tolerance=tolerance, cross_run_replay=replay,
        tasks=tasks, query_order=queries, families=families, cohorts=cohorts,
        samples_by_task={task: len(ids) for task, ids in groups.items()},
        summary=summary, comparisons=comparisons, source_effects=source_effects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {method: {key: value['primary'][key] for key in ('mean', 'ci')}
                            for method, value in table.items()}
                      for name, table in summary.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
