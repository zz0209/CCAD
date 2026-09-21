import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    arrays, configs, panels, inputs, query_orders = [], [], [], [], []
    for run in args.runs:
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS', run
        panel = json.loads((run/'panel.json').read_text())
        config = json.loads((run/'config.resolved.json').read_text())
        with np.load(run/'responses.npz') as raw:
            arrays.append({key: raw[key].copy() for key in raw.files})
        with (run/'metrics.raw.jsonl').open() as stream:
            records = [json.loads(line) for line in stream]
        order = list(dict.fromkeys(r['operation'] for r in records if r['method'] == 'none'))
        assert set(order) == set(panel['queries'])
        positions = {(r['task'], r['row_id']): i for i, r in enumerate(panel['rows'])}
        keys = set()
        for record in records:
            key = (record['method'], record['operation'], record['task'], record['row_id'])
            assert key not in keys, key
            keys.add(key)
            qi, ri = order.index(record['operation']), positions[record['task'], record['row_id']]
            assert arrays[-1][record['method']][qi, ri] == record['margin']
        assert len(keys) == len(arrays[-1])*len(order)*len(positions)
        query_orders.append(order)
        configs.append(config)
        panels.append(panel)
        inputs.append(dict(run=run.as_posix(), target_seed=config['target_seed'],
            response_sha256=hashlib.sha256((run/'responses.npz').read_bytes()).hexdigest()))
    assert len({c['target_seed'] for c in configs}) == len(configs)
    for panel, values in zip(panels, arrays):
        assert panel['rows'] == panels[0]['rows'] and panel['queries'] == panels[0]['queries']
        for key in ['source', 'none']:
            assert np.allclose(values[key], arrays[0][key], atol=2e-5, rtol=0), key
    methods = sorted(set.intersection(*(set(a) for a in arrays))-{'source', 'none'})
    assert all(order == query_orders[0] for order in query_orders)
    rows, queries = panels[0]['rows'], query_orders[0]
    tasks = configs[0]['tasks']
    groups = {task: np.array([i for i, r in enumerate(rows) if r['task'] == task]) for task in tasks}
    families = dict(full=['full'], parts=['verb', 'number', 'gender'],
        primary=configs[0]['primary_requests'])
    query_ids = {name: np.array([queries.index(q) for q in qs]) for name, qs in families.items()}
    source, clean = arrays[0]['source'], arrays[0]['none']
    target = np.array([[a[method] for a in arrays] for method in methods])
    assert np.isfinite(target).all() and np.isfinite(source).all()

    def measure(draws, seeds):
        output = np.empty((len(methods), len(tasks), len(families)))
        for ti, task in enumerate(tasks):
            documents = draws[task]
            for fi, name in enumerate(families):
                qi = query_ids[name]
                ref = source[np.ix_(qi, documents)]
                baseline = clean[np.ix_(qi, documents)]
                denominator = np.square(ref-baseline).sum()
                assert denominator > 0, (task, name)
                selected = np.take(np.take(np.take(target, seeds, axis=1), qi, axis=2), documents, axis=3)
                errors = np.sqrt(np.square(selected-ref).sum((2, 3))/denominator)
                output[:, ti, fi] = errors.mean(1)
        return output

    point = measure(groups, np.arange(len(arrays)))
    by_seed = [measure(groups, np.array([i])) for i in range(len(arrays))]
    rng = np.random.default_rng(9261)
    boot = np.empty((args.bootstrap, *point.shape))
    for iteration in range(args.bootstrap):
        draws = {task: rng.choice(ids, len(ids), replace=True) for task, ids in groups.items()}
        seeds = rng.integers(len(arrays), size=len(arrays))
        boot[iteration] = measure(draws, seeds)
    summary, comparisons = {}, {}
    for mi, method in enumerate(methods):
        summary[method] = {}
        for fi, family in enumerate(families):
            summary[method][family] = dict(mean=float(point[mi, :, fi].mean()),
                ci=np.quantile(boot[:, mi, :, fi].mean(1), [.025, .975]).tolist(),
                by_seed={str(c['target_seed']): float(values[mi, :, fi].mean()) for c, values in zip(configs, by_seed)},
                by_task={task: float(point[mi, ti, fi]) for ti, task in enumerate(tasks)})
    for other in methods:
        if other == 'program':
            continue
        pi, oi = methods.index('program'), methods.index(other)
        comparisons['program_minus_'+other] = {
            family: dict(mean=float((point[pi, :, fi]-point[oi, :, fi]).mean()),
                ci=np.quantile((boot[:, pi, :, fi]-boot[:, oi, :, fi]).mean(1), [.025, .975]).tolist())
            for fi, family in enumerate(families)}
    effects = {task: {
        family: dict(rms=float(np.sqrt(np.square((source-clean)[np.ix_(qi, ids)]).mean())),
            clean_accuracy=float((clean[np.ix_(qi, ids)] > 0).mean()),
            source_accuracy=float((source[np.ix_(qi, ids)] > 0).mean()))
        for family, qi in query_ids.items()} for task, ids in groups.items()}
    output = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        bootstrap=args.bootstrap, tasks=tasks, queries=families,
        inference='Paired sentence resampling within fixed grammatical paradigms and target-seed resampling; fixed source and fixed model.',
        samples_by_task={task: len(ids) for task, ids in groups.items()},
        summary=summary, comparisons=comparisons, source_effects=effects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({method: {family: round(summary[method][family]['mean'], 5) for family in families} for method in methods}))


if __name__ == '__main__':
    main()
