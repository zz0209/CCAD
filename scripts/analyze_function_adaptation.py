import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_run(run):
    assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
    config = json.loads((run/'config.resolved.json').read_text())
    panel = json.loads((run/'panel.json').read_text())
    with np.load(run/'responses.npz') as archive:
        responses = {key: archive[key].copy() for key in archive.files}
    order = panel['query_order']
    positions = {(r['task'], r['row_id']): i for i, r in enumerate(panel['rows'])}
    seen = set()
    with (run/'metrics.raw.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['method'], row['mode'], row['task'], row['row_id'])
            assert key not in seen
            seen.add(key)
            assert responses[row['method']][order.index(row['mode']), positions[(row['task'], row['row_id'])]] == row['margin']
    assert len(seen) == len(responses)*len(order)*len(positions)
    parts = ['verb', 'number', 'gender']
    new_index = parts.index(config['adapt_part'])
    access = json.loads((run/'source_response_access.json').read_text())
    for call in access['calls']:
        if config['adapt_part'] in call['parts']:
            assert all(r['task'] == config['tasks'][new_index] for r in call['rows'])
    with np.load(run/'training_schedule.npz') as schedule:
        assert len(schedule['source_members']) == 192
        for request, batch in zip(schedule['requests'], schedule['row_batches']):
            if request[new_index] != 0:
                assert all(panel['fit'][i]['task'] == config['tasks'][new_index] for i in batch)
    pairs = {variant: sum(len(call['rows']) for call in access['calls'] if call['phase'] == variant)
             for variant in config['variants']}
    assert len(set(pairs.values())) == 1
    assert access['new_context_count'] <= config['new_fit_pairs']
    fit_ids = {r[key] for r in panel['fit'] for key in ['sentence_good', 'sentence_bad']}
    assert not fit_ids & {r[key] for r in panel['rows'] for key in ['sentence_good', 'sentence_bad']}
    return config, panel, responses, access, len(seen)


def analyze(run, bootstrap):
    config, panel, responses, access, record_count = read_run(run)
    parts = ['verb', 'number', 'gender']
    new_index = parts.index(config['adapt_part'])
    groups = [np.array([i for i, r in enumerate(panel['rows']) if r['task'] == task]) for task in config['tasks']]
    methods = sorted(set(responses)-{'none', 'source'})
    qi = [panel['query_order'].index(part) for part in parts]
    mixed = [panel['query_order'].index(q) for q in config['primary_requests']]

    def measure(draws):
        own, combinations = [], []
        for method in methods:
            own.append([float(np.linalg.norm(responses[method][q, indices]-responses['source'][q, indices])/
                              np.linalg.norm(responses['source'][q, indices]-responses['none'][q, indices])) for q, indices in zip(qi, draws)])
            combinations.append(float(np.mean([np.linalg.norm((responses[method]-responses['source'])[np.ix_(mixed, ids)])/
                np.linalg.norm((responses['source']-responses['none'])[np.ix_(mixed, ids)]) for ids in draws])))
        return np.array(own), np.array(combinations)

    point, mixed_point = measure(groups)
    rng = np.random.default_rng(9311)
    boot = np.array([measure([rng.choice(ids, len(ids), replace=True) for ids in groups])[0] for _ in range(bootstrap)])
    old = [i for i in range(3) if i != new_index]
    summary = {method: dict(by_function=dict(zip(parts, point[mi].tolist())), new=float(point[mi, new_index]),
        old=float(point[mi, old].mean()), all=float(point[mi].mean()), mixed=float(mixed_point[mi])) for mi, method in enumerate(methods)}
    comparisons = {}
    for step in [*config['learning_curve_steps'], config['steps']]:
        suffix = '' if step == config['steps'] else f'_step{step}'
        warm, fresh, prior = [methods.index(m) for m in ['program_warm'+suffix, 'program_fresh'+suffix, 'prior']]
        comparisons[str(step)] = {}
        for name, left, right, selected in [('new_warm_minus_fresh', warm, fresh, [new_index]),
                                          ('old_warm_minus_fresh', warm, fresh, old),
                                          ('old_warm_minus_prior', warm, prior, old),
                                          ('all_warm_minus_fresh', warm, fresh, [0,1,2])]:
            samples = (boot[:, left, selected]-boot[:, right, selected]).mean(1)
            comparisons[str(step)][name] = dict(mean=float((point[left, selected]-point[right, selected]).mean()),
                                                ci=np.quantile(samples, [.025,.975]).tolist())
    costs = {}
    for phase in ['normalization', *config['variants']]:
        calls = [call for call in access['calls'] if call['phase'] == phase]
        costs[phase] = dict(pair_response_evaluations=sum(len(call['rows']) for call in calls),
            new_pair_response_evaluations=sum(len(call['rows']) for call in calls if config['adapt_part'] in call['parts']))
    return dict(run=run.as_posix(), source_seed=config.get('source_seed', 1), target_seed=config['target_seed'],
        new_function=config['adapt_part'], new_contexts=access['new_context_count'], prior_training_steps=config['prior_training_steps'],
        response_sha256=hashlib.sha256((run/'responses.npz').read_bytes()).hexdigest(), records=record_count,
        summary=summary, comparisons=comparisons, fitting_costs=costs,
        inference='Paired sentence bootstrap within each fixed grammar; fixed source, target and function set; development evidence.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    assert not args.output.exists()
    studies = [analyze(run, args.bootstrap) for run in args.runs]
    output = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), bootstrap=args.bootstrap, studies=studies)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    for study in studies:
        print(json.dumps(dict(function=study['new_function'], contexts=study['new_contexts'],
            initial=study['summary']['initial'], prior=study['summary']['prior'],
            fresh=study['summary']['program_fresh'], warm=study['summary']['program_warm'], comparisons=study['comparisons'])))


if __name__ == '__main__':
    main()
