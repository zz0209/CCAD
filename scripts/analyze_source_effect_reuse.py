import argparse
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_run(run):
    assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
    config = json.loads((run/'config.resolved.json').read_text())
    panel = json.loads((run/'panel.json').read_text())
    assert config['steps'] == 0 and not config['variants']
    with np.load(run/'responses.npz') as archive:
        values = {method: archive[method].copy() for method in archive.files}
    assert all(np.isfinite(value).all() for value in values.values())
    records = [json.loads(line) for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
    order = list(dict.fromkeys(row['operation'] for row in records if row['method'] == 'none'))
    assert set(order) == set(panel['queries'])
    positions = {(row['task'], row['row_id']): i for i, row in enumerate(panel['rows'])}
    assert len(positions) == len(panel['rows'])
    seen = set()
    for row in records:
        key = row['method'], row['operation'], row['task'], row['row_id']
        assert key not in seen
        seen.add(key)
        assert values[row['method']][order.index(row['operation']), positions[row['task'], row['row_id']]] == row['margin']
    assert len(seen) == len(values)*len(order)*len(positions)
    recorded = json.loads((run/'inputs.json').read_text())['inputs']
    identities = {}
    for method in values.keys()-{'none', 'source'}:
        path = (config['evaluation_checkpoints'][method]['path'].format(
            training_run=config['checkpoint_run_by_target'][str(config['target_seed'])],
            target_seed=config['target_seed']) if method in config['evaluation_checkpoints'] else config['target_checkpoint'])
        matches = {entry['sha256'] for entry in recorded if Path(entry['path']).resolve() == Path(path).resolve()}
        assert len(matches) == 1, (run, method, path)
        identities[method] = dict(path=Path(path).as_posix(), recorded_sha256=next(iter(matches)))
    return dict(config=config, panel=panel, values=values, order=order, identities=identities,
                evidence=dict(run=run.as_posix(), source_seed=config['source_seed'], target_seed=config['target_seed'],
                              responses_sha256=hashlib.sha256((run/'responses.npz').read_bytes()).hexdigest(),
                              jsonl_rows_verified=len(seen), checkpoint_identities=identities))


def contributions(runs, targets, tasks, groups, methods):
    result = {}
    reference = next(iter(runs.values()))
    for target in targets:
        sources = sorted(source for source, t in runs if t == target)
        pairs = list(itertools.combinations(sources, 2))
        assert len(pairs) > 0
        source_difference = np.stack([runs[a, target]['values']['source']-runs[b, target]['values']['source'] for a, b in pairs])
        prediction = np.stack([np.stack([runs[a, target]['values'][method]-runs[b, target]['values'][method]
                                        for a, b in pairs]) for method in methods])
        errors = np.square(prediction-source_difference)
        agreement = (np.sign(prediction) == np.sign(source_difference)) & (source_difference != 0)
        ties = (prediction == 0) & (source_difference != 0)
        energy, nonzero = np.square(source_difference), source_difference != 0
        source_responses = np.stack([runs[source, target]['values']['source'] for source in sources])
        own_error = np.stack([np.stack([runs[source, target]['values'][method]
                                       for source in sources]) for method in methods])-source_responses
        source_effect = source_responses-reference['values']['none']
        mean_error = own_error.mean(1, keepdims=True)
        total_energy = np.square(own_error).sum(1)
        common_energy = len(sources)*np.square(mean_error[:, 0])
        centered_energy = np.square(own_error-mean_error).sum(1)
        assert np.allclose(total_energy, common_energy+centered_energy, atol=1e-10, rtol=1e-12)
        assert np.allclose(errors.sum(1), len(sources)*centered_energy, atol=1e-10, rtol=1e-12)
        result[target] = {}
        for task in tasks:
            rows = np.array([i for i, row in enumerate(reference['panel']['rows']) if row['task'] == task])
            entry = dict(pairs=pairs, sources=sources, rows=rows, per_pair_events=np.array([len(q) for q in groups.values()]))
            for name, array in [('error', errors), ('agreement', agreement), ('ties', ties), ('energy', energy), ('nonzero', nonzero),
                                ('own_error', np.square(own_error)), ('source_effect', np.square(source_effect)),
                                ('total_energy', total_energy), ('common_energy', common_energy), ('centered_energy', centered_energy)]:
                entry[name] = np.stack([np.take(array, indices, axis=-2).sum(-2)[..., rows]
                                        for indices in groups.values()], axis=-2)
            result[target][task] = entry
    return result


def pool_pairs(data, excluded=None):
    pooled = {}
    for target, tasks in data.items():
        if target == excluded:
            continue
        pooled[target] = {}
        for task, entry in tasks.items():
            selected = [i for i, pair in enumerate(entry['pairs']) if excluded not in pair]
            source_ids = [i for i, source in enumerate(entry['sources']) if source != excluded]
            assert selected, (target, task, excluded)
            pooled[target][task] = dict(
                **{name: np.take(entry[name], selected, axis=-3).sum(-3)
                   for name in ['error', 'agreement', 'ties', 'energy', 'nonzero']},
                own_error=entry['own_error'][:, source_ids], source_effect=entry['source_effect'][source_ids],
                pairs=[entry['pairs'][i] for i in selected], rows=entry['rows'],
                events=len(selected)*entry['per_pair_events'])
    return pooled


def measure(pooled, documents, methods, tasks):
    values = []
    for target in pooled:
        task_values = []
        for task in tasks:
            entry, ids = pooled[target][task], documents[task]
            denominator = entry['energy'][:, ids].sum(-1)
            numerator = entry['error'][:, :, ids].sum(-1)
            nrmse = np.sqrt(np.divide(numerator, denominator[None], out=np.full_like(numerator, np.nan), where=denominator[None] > 0))
            counts = entry['nonzero'][:, ids].sum(-1)
            correct = entry['agreement'][:, :, ids].sum(-1)
            ordering = np.divide(correct, counts[None], out=np.full(correct.shape, np.nan), where=counts[None] > 0)
            source_energy = entry['source_effect'][:, :, ids].sum(-1)
            own_error = entry['own_error'][:, :, :, ids].sum(-1)
            absolute = np.sqrt(np.divide(own_error, source_energy[None], out=np.full_like(own_error, np.nan),
                                         where=source_energy[None] > 0)).mean(1)
            task_values.append(np.stack([nrmse, ordering, absolute], axis=-1))
        values.append(np.stack(task_values, axis=1))
    return np.stack(values, axis=1)


def interval(point, samples):
    finite = np.isfinite(samples)
    return dict(mean=float(point) if np.isfinite(point) else None,
                ci95=np.quantile(samples[finite], [.025, .975]).tolist() if finite.any() else None,
                valid_bootstrap=int(finite.sum()), undefined_bootstrap=int((~finite).sum()))


def report_cell(point, bootstrap, targets, tasks):
    output = interval(point.mean(), bootstrap.mean((1, 2)))
    output['by_target'] = {str(target): interval(point[ti].mean(), bootstrap[:, ti].mean(1)) for ti, target in enumerate(targets)}
    output['by_task'] = {task: interval(point[:, si].mean(), bootstrap[:, :, si].mean(1)) for si, task in enumerate(tasks)}
    output['by_target_task'] = {str(target): {task: interval(point[ti, si], bootstrap[:, ti, si])
                                             for si, task in enumerate(tasks)} for ti, target in enumerate(targets)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round09'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--targets', type=int, nargs='+')
    args = parser.parse_args()
    assert args.bootstrap > 0 and not args.output.exists()
    started = time.perf_counter()
    runs = {}
    for path in sorted(args.run_root.glob('RG09_GPT2_SOURCE_REUSE_CONFIRM_*')):
        run = read_run(path)
        edge = run['config']['source_seed'], run['config']['target_seed']
        assert edge not in runs and edge[0] != edge[1]
        runs[edge] = run
        print(json.dumps(dict(stage='READ', run=path.name, completed=len(runs))), flush=True)
    assert len(runs) == 12
    reference = next(iter(runs.values()))
    methods = sorted(reference['values'].keys()-{'source'})
    tasks, queries = reference['config']['tasks'], reference['order']
    for (source, target), run in runs.items():
        assert run['panel']['rows'] == reference['panel']['rows']
        assert run['panel']['queries'] == reference['panel']['queries'] and run['order'] == queries
        assert run['config']['tasks'] == tasks
        assert run['config']['primary_requests'] == reference['config']['primary_requests']
        assert run['values'].keys() == reference['values'].keys()
        assert np.array_equal(run['values']['none'], reference['values']['none'])
        same_source = next(value for (s, _), value in runs.items() if s == source)
        same_target = next(value for (_, t), value in runs.items() if t == target)
        assert np.array_equal(run['values']['source'], same_source['values']['source'])
        assert run['identities'] == same_target['identities']
    targets = sorted({target for _, target in runs}) if args.targets is None else sorted(args.targets)
    assert set(targets) <= {target for _, target in runs}
    families = dict(full=['full'], parts=['verb', 'number', 'gender'], primary=reference['config']['primary_requests'])
    names = dict(families, **{'query/'+query: [query] for query in queries})
    groups = {name: np.array([queries.index(query) for query in family]) for name, family in names.items()}
    data = contributions(runs, targets, tasks, groups, methods)
    pooled = pool_pairs(data)
    documents = {task: np.arange(len(pooled[targets[0]][task]['rows'])) for task in tasks}
    point = measure(pooled, documents, methods, tasks)
    assert np.allclose(point[methods.index('none'), :, :, :, 0], 1., atol=1e-12, rtol=0)
    rng = np.random.default_rng(9261)
    boot = np.empty((args.bootstrap, *point.shape))
    for iteration in range(args.bootstrap):
        draws = {task: rng.choice(ids, len(ids), replace=True) for task, ids in documents.items()}
        boot[iteration] = measure(pooled, draws, methods, tasks)
        if (iteration+1) % 200 == 0 or iteration+1 == args.bootstrap:
            print(json.dumps(dict(stage='BOOTSTRAP', completed=iteration+1, total=args.bootstrap)), flush=True)
    summary, per_query, comparisons, query_comparisons = {}, {}, {}, {}
    focal = methods.index('reuse_program')
    for mi, method in enumerate(methods):
        summary[method], per_query[method] = {}, {}
        if mi != focal:
            comparisons['reuse_program_minus_'+method], query_comparisons['reuse_program_minus_'+method] = {}, {}
        for fi, name in enumerate(names):
            item = {metric: report_cell(point[mi, :, :, fi, ki], boot[:, mi, :, :, fi, ki], targets, tasks)
                    for ki, metric in enumerate(['nrmse', 'ordering_accuracy', 'absolute_nrmse'])}
            (per_query[method] if name.startswith('query/') else summary[method])[name.removeprefix('query/')] = item
            if mi != focal:
                item = {metric: report_cell(point[focal, :, :, fi, ki]-point[mi, :, :, fi, ki],
                                            boot[:, focal, :, :, fi, ki]-boot[:, mi, :, :, fi, ki], targets, tasks)
                        for ki, metric in enumerate(['nrmse', 'ordering_accuracy', 'absolute_nrmse'])}
                (query_comparisons if name.startswith('query/') else comparisons)['reuse_program_minus_'+method][name.removeprefix('query/')] = item
    effects = {}
    for target in targets:
        effects[str(target)] = {}
        for task, entry in pooled[target].items():
            effects[str(target)][task] = {name: dict(
                rms=float(np.sqrt(entry['energy'][fi].sum()/(entry['events'][fi]*len(entry['rows'])))),
                squared_energy=float(entry['energy'][fi].sum()), events=int(entry['events'][fi]*len(entry['rows'])),
                nonzero_events=int(entry['nonzero'][fi].sum()),
                prediction_ties={method: int(entry['ties'][mi, fi].sum()) for mi, method in enumerate(methods)})
                for fi, name in enumerate(names)}
    sensitivity = {}
    for node in sorted({seed for edge in runs for seed in edge}):
        reduced = pool_pairs(data, node)
        if not reduced:
            sensitivity[str(node)] = dict(status='no_retained_target')
            continue
        values = measure(reduced, documents, methods, tasks)
        sensitivity[str(node)] = dict(targets=list(reduced), source_pairs={str(t): reduced[t][tasks[0]]['pairs'] for t in reduced},
            summary={method: {family: dict(nrmse=float(values[mi, :, :, fi, 0].mean()),
                                          ordering_accuracy=float(values[mi, :, :, fi, 1].mean()),
                                          absolute_nrmse=float(values[mi, :, :, fi, 2].mean()))
                              for fi, family in enumerate(families)} for mi, method in enumerate(methods)})
    decomposition = {}
    for mi, method in enumerate(methods):
        decomposition[method] = {}
        for fi, name in enumerate(names):
            cells = {str(target): {task: {kind: float(data[target][task][kind][mi, fi].sum())
                                         for kind in ['total_energy', 'common_energy', 'centered_energy']}
                                   for task in tasks} for target in targets}
            totals = {kind: sum(cells[str(target)][task][kind] for target in targets for task in tasks)
                      for kind in ['total_energy', 'common_energy', 'centered_energy']}
            totals['source_mean_fraction'] = totals['common_energy']/totals['total_energy'] if totals['total_energy'] else None
            totals['source_centered_fraction'] = totals['centered_energy']/totals['total_energy'] if totals['total_energy'] else None
            decomposition[method][name] = dict(**totals, by_target_task=cells)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), evidence_level='exploratory_reanalysis_of_exposed_R9_confirmation',
        analysis_seconds=time.perf_counter()-started, bootstrap=args.bootstrap, bootstrap_seed=9261,
        inputs=[run['evidence'] for run in runs.values()], targets=targets, tasks=tasks, queries=queries, families=families,
        samples_by_task={task: len(ids) for task, ids in documents.items()},
        source_pairs_by_target={str(t): pooled[t][tasks[0]]['pairs'] for t in targets},
        definitions=dict(nrmse='For each target, task and family, sum squared prediction-difference error over all source pairs, queries and documents; divide by summed squared source difference and take square root. Average fixed target-task cells equally.',
            ordering_accuracy='For every exactly nonzero source difference, count a correct prediction only when both signs agree. Predicted ties are incorrect. Pool events within target-task-family, then average cells equally.',
            absolute_nrmse='Original R9 metric. Normalize each source-target-task error by its own source-versus-clean effect, take square root, then equally average sources, targets and tasks.',
            decomposition='For each target and input, e_s=prediction_s-source_s. Sum_s e_s^2=S*mean_s(e_s)^2+sum_s(e_s-mean_s(e_s))^2. Squared pair-error sum equals S times source-centered error energy. Report unnormalized energies over the fixed requests and documents.',
            inference='Paired sentence resampling within each fixed grammar; every method, request and target shares each draw. Four shared seed nodes are fixed. Source pairs and transfer edges are dependent. Node deletion is descriptive sensitivity.',
            undefined='Zero source-difference energy or no nonzero differences produces an undefined cell; aggregate means require all fixed cells. Confidence intervals report valid and undefined bootstrap counts.'),
        checks=dict(all_jsonl_arrays_equal=True, all_clean_bit_equal=True, same_source_response_bit_equal=True,
                    same_target_checkpoint_path_and_recorded_sha=True, zero_prediction_nrmse=1., exact_error_decomposition=True),
        summary=summary, per_query=per_query, comparisons=comparisons, query_comparisons=query_comparisons,
        source_difference=effects, squared_error_decomposition=decomposition, delete_one_shared_seed=sensitivity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({method: {metric: summary[method]['primary'][metric]['mean']
                              for metric in ['nrmse', 'ordering_accuracy']} for method in methods}), flush=True)


if __name__ == '__main__':
    main()
