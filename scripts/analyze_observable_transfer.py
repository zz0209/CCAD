import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analyze_grammar_member_program as grammar_analysis


def family_analysis(run, repetitions, focal='adjoint'):
    with tempfile.TemporaryDirectory(prefix='ccad-observable-analysis-') as temporary:
        output = Path(temporary)/'families.json'
        previous = sys.argv
        sys.argv = [str(Path(grammar_analysis.__file__)), '--runs', str(run),
                    '--output', str(output), '--bootstrap', str(repetitions), '--focal', focal]
        try:
            grammar_analysis.main()
        finally:
            sys.argv = previous
        return json.loads(output.read_text())


def estimate(point, draws):
    valid = np.isfinite(draws)
    return dict(mean=float(point) if np.isfinite(point) else None,
                ci=np.quantile(draws[valid], [.025, .975]).tolist() if valid.any() else None,
                valid_draws=int(valid.sum()), undefined_draws=int((~valid).sum()))


def query_analysis(run, methods, repetitions, focal='adjoint'):
    panel = json.loads((run/'panel.json').read_text())
    config = json.loads((run/'config.resolved.json').read_text())
    with (run/'metrics.raw.jsonl').open() as stream:
        order = list(dict.fromkeys(row['operation'] for line in stream
                     if (row := json.loads(line))['method'] == 'none'))
    tasks = config['tasks']
    groups = {task: np.array([i for i, row in enumerate(panel['rows']) if row['task'] == task])
              for task in tasks}
    with np.load(run/'responses.npz') as arrays:
        source, clean = arrays['source'].copy(), arrays['none'].copy()
        target = np.stack([arrays[method] for method in methods])
    assert np.isfinite(source).all() and np.isfinite(clean).all() and np.isfinite(target).all()
    assert all(len(ids) > 0 for ids in groups.values())

    def measure(documents):
        values = np.full((len(methods), len(tasks), len(order)), np.nan)
        energies = np.empty((len(tasks), len(order)))
        for ti, task in enumerate(tasks):
            ids = documents[task]
            reference = source[:, ids]
            denominator = np.square(reference-clean[:, ids]).sum(-1)
            numerator = np.square(target[:, :, ids]-reference).sum(-1)
            ratio = np.divide(numerator, denominator[None], out=np.full_like(numerator, np.nan),
                              where=denominator[None] > 0)
            values[:, ti] = np.sqrt(ratio)
            energies[ti] = denominator
        return values, energies

    point, energies = measure(groups)
    boot = np.empty((repetitions, *point.shape))
    rng = np.random.default_rng(9261)
    for iteration in range(repetitions):
        documents = {task: rng.choice(ids, len(ids), replace=True) for task, ids in groups.items()}
        rng.integers(1, size=1)
        boot[iteration] = measure(documents)[0]
    summary, comparisons = {}, {}
    for mi, method in enumerate(methods):
        summary[method] = {}
        for qi, query in enumerate(order):
            entry = estimate(point[mi, :, qi].mean(), boot[:, mi, :, qi].mean(1))
            entry['by_task'] = {task: estimate(point[mi, ti, qi], boot[:, mi, ti, qi])
                                for ti, task in enumerate(tasks)}
            summary[method][query] = entry
    focal_index = methods.index(focal)
    for oi, other in enumerate(methods):
        if other == focal:
            continue
        comparisons[focal+'_minus_'+other] = {}
        for qi, query in enumerate(order):
            difference = point[focal_index, :, qi]-point[oi, :, qi]
            draws = boot[:, focal_index, :, qi]-boot[:, oi, :, qi]
            entry = estimate(difference.mean(), draws.mean(1))
            entry['by_task'] = {task: estimate(difference[ti], draws[:, ti])
                                for ti, task in enumerate(tasks)}
            comparisons[focal+'_minus_'+other][query] = entry
    effects = {task: {query: dict(sum_squared_effect=float(energies[ti, qi]),
                                rms=float(np.sqrt(energies[ti, qi]/len(groups[task]))))
                      for qi, query in enumerate(order)} for ti, task in enumerate(tasks)}
    return dict(query_order=order, summary=summary, comparisons=comparisons,
                source_effects=effects,
                undefined='Zero source-effect energy produces an undefined normalized error. '
                          'The document draw remains shared across all methods and requests; '
                          'an all-task mean is defined only when every fixed task is defined.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    assert args.bootstrap > 0
    with np.load(args.run/'responses.npz') as arrays:
        assert 'adjoint' in arrays.files, arrays.files
    output = family_analysis(args.run, args.bootstrap)
    methods = list(output['summary'])
    requested = ['initial', 'program_reference', 'rec', 'adjoint', 'readout', 'fixed']
    output['reported_methods'] = [method for method in requested if method in methods]
    output['additional_methods'] = [method for method in methods if method not in requested]
    output['primary_comparisons'] = {name: output['comparisons'][name]['primary']
                                     for other in ['rec', 'initial', 'program_reference']
                                     if (name := 'adjoint_minus_'+other) in output['comparisons']}
    output['per_query'] = query_analysis(args.run, methods, args.bootstrap)
    output['inference'] = ('Paired document resampling within each fixed grammatical task, '
                           'with all requests and methods sharing each draw. One fixed source '
                           'and target dictionary; intervals describe the document population '
                           'for these fixed tasks and dictionaries.')
    output['analysis'] = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                             script=str(Path(__file__).resolve()),
                             reused_script=str(Path(grammar_analysis.__file__).resolve()),
                             reused_script_sha256=hashlib.sha256(Path(grammar_analysis.__file__).read_bytes()).hexdigest(),
                             bootstrap_seed=9261,
                             primary_definition='Exact config.resolved.json primary_requests; '
                                                'source-normalized RMSE per fixed task, then equal task mean.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=args.output.as_posix(), primary=output['primary_comparisons'])))


if __name__ == '__main__':
    main()
