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
    parser.add_argument('--source-reuse', action='store_true')
    parser.add_argument('--heldout-function', choices=['verb', 'number', 'gender'])
    parser.add_argument('--focal')
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    arrays, configs, panels, inputs, query_orders = [], [], [], [], []
    for run in args.runs:
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS', run
        panel = json.loads((run/'panel.json').read_text())
        config = json.loads((run/'config.resolved.json').read_text())
        if config.get('heldout_part'):
            omitted = ['verb', 'number', 'gender'].index(config['heldout_part'])
            with np.load(run/'training_schedule.npz') as schedule:
                assert (schedule['requests'][:, omitted] == 0).all()
                assert (schedule['source_parts'] != omitted).all()
                assert len(schedule['source_members']) == 128
                if config.get('member_requests'):
                    parts = np.array(json.loads((run/'method_summary.json').read_text())['source_parts'])
                    assert (schedule['member_requests'][:, parts == omitted] == 0).all()
            assert all(row['task'] != config['tasks'][omitted] for row in panel['fit'])
        if args.heldout_function:
            assert config['heldout_part'] == args.heldout_function
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
        inputs.append(dict(run=run.as_posix(), source_seed=config.get('source_seed', 1), target_seed=config['target_seed'],
            response_sha256=hashlib.sha256((run/'responses.npz').read_bytes()).hexdigest()))
    edges = [(c.get('source_seed', 1), c['target_seed']) for c in configs]
    assert len(set(edges)) == len(edges)
    if args.source_reuse:
        assert all(s != t for s, t in edges)
        assert all(c['steps'] == 0 and not c['variants'] for c in configs)
    else:
        assert len({s for s, _ in edges}) == 1
    for index, (panel, values) in enumerate(zip(panels, arrays)):
        assert panel['rows'] == panels[0]['rows'] and panel['queries'] == panels[0]['queries']
        assert np.allclose(values['none'], arrays[0]['none'], atol=2e-5, rtol=0)
        reference = next(j for j, edge in enumerate(edges) if edge[0] == edges[index][0])
        assert np.allclose(values['source'], arrays[reference]['source'], atol=2e-5, rtol=0)
    methods = sorted(set.intersection(*(set(a) for a in arrays))-{'source', 'none'})
    assert all(order == query_orders[0] for order in query_orders)
    rows, queries = panels[0]['rows'], query_orders[0]
    tasks = configs[0]['tasks']
    if args.heldout_function:
        tasks = [tasks[['verb', 'number', 'gender'].index(args.heldout_function)]]
    groups = {task: np.array([i for i, r in enumerate(rows) if r['task'] == task]) for task in tasks}
    families = dict(full=['full'], parts=['verb', 'number', 'gender'],
        primary=configs[0]['primary_requests'])
    if args.heldout_function:
        families['heldout'] = [args.heldout_function]
    query_ids = {name: np.array([queries.index(q) for q in qs]) for name, qs in families.items()}
    source = np.array([a['source'] for a in arrays])
    clean = np.array([a['none'] for a in arrays])
    target = np.array([[a[method] for a in arrays] for method in methods])
    assert np.isfinite(target).all() and np.isfinite(source).all()

    def measure(draws, seeds):
        output = np.empty((len(methods), len(tasks), len(families)))
        for ti, task in enumerate(tasks):
            documents = draws[task]
            for fi, name in enumerate(families):
                qi = query_ids[name]
                ref = source[np.ix_(seeds, qi, documents)]
                baseline = clean[np.ix_(seeds, qi, documents)]
                denominator = np.square(ref-baseline).sum((1, 2))
                assert (denominator > 0).all(), (task, name)
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
        seeds = np.arange(len(arrays)) if args.source_reuse else rng.integers(len(arrays), size=len(arrays))
        boot[iteration] = measure(draws, seeds)
    summary, comparisons = {}, {}
    for mi, method in enumerate(methods):
        summary[method] = {}
        for fi, family in enumerate(families):
            summary[method][family] = dict(mean=float(point[mi, :, fi].mean()),
                ci=np.quantile(boot[:, mi, :, fi].mean(1), [.025, .975]).tolist(),
                by_seed={(f's{edge[0]}_t{edge[1]}' if args.source_reuse else str(edge[1])):
                         float(values[mi, :, fi].mean()) for edge, values in zip(edges, by_seed)},
                by_task={task: float(point[mi, ti, fi]) for ti, task in enumerate(tasks)})
    focal = args.focal or ('reuse_program' if args.source_reuse else 'program')
    for other in methods:
        if other == focal:
            continue
        pi, oi = methods.index(focal), methods.index(other)
        comparisons[focal+'_minus_'+other] = {
            family: dict(mean=float((point[pi, :, fi]-point[oi, :, fi]).mean()),
                ci=np.quantile((boot[:, pi, :, fi]-boot[:, oi, :, fi]).mean(1), [.025, .975]).tolist())
            for fi, family in enumerate(families)}
    effects = {}
    for source_seed in sorted({s for s, _ in edges}):
        index = next(i for i, edge in enumerate(edges) if edge[0] == source_seed)
        effects[str(source_seed)] = {task: {
            family: dict(rms=float(np.sqrt(np.square((source[index]-clean[index])[np.ix_(qi, ids)]).mean())),
                clean_accuracy=float((clean[index][np.ix_(qi, ids)] > 0).mean()),
                source_accuracy=float((source[index][np.ix_(qi, ids)] > 0).mean()))
            for family, qi in query_ids.items()} for task, ids in groups.items()}
    node_sensitivity = {}
    if args.source_reuse and len(edges) > 1:
        for node in sorted({n for edge in edges for n in edge}):
            remaining = np.array([i for i, edge in enumerate(edges) if node not in edge])
            assert len(remaining) > 0
            retained = measure(groups, remaining)
            node_sensitivity[str(node)] = dict(edges=[edges[i] for i in remaining],
                summary={method: {family: float(retained[mi, :, fi].mean())
                    for fi, family in enumerate(families)} for mi, method in enumerate(methods)})
    output = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        bootstrap=args.bootstrap, tasks=tasks, queries=families,
        inference=('Paired sentence resampling within fixed grammatical paradigms and a fixed source-target seed cohort. Shared seed nodes are not independent edges; node deletion removes all incident edges.'
            if args.source_reuse else 'Paired sentence resampling within fixed grammatical paradigms and target-seed resampling; fixed source and fixed model.'),
        samples_by_task={task: len(ids) for task, ids in groups.items()},
        summary=summary, comparisons=comparisons,
        source_effects=effects if args.source_reuse else next(iter(effects.values())),
        delete_one_shared_seed=node_sensitivity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({method: {family: round(summary[method][family]['mean'], 5) for family in families} for method in methods}))


if __name__ == '__main__':
    main()
