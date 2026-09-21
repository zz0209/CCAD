import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyses', nargs='+', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--table', type=Path)
    args = parser.parse_args()
    assert len(args.analyses) == len(args.seeds) >= 2
    banks = [np.load(p/'function_arrays.npz') for p in args.analyses]
    for bank in banks[1:]:
        for key in ['documents', 'queries', 'clean', 'source', 'resamples']:
            assert np.array_equal(bank[key], banks[0][key]), key
    methods = ['native', 'geometry_gain', 'raw', 'local_whole', 'local_parts']
    queries = banks[0]['queries'].tolist()
    source, clean = banks[0]['source'], banks[0]['clean']
    draws = banks[0]['resamples']
    rng = np.random.default_rng(607)
    seed_draws = rng.integers(len(banks), size=(len(draws), len(banks)))
    result = dict(target_seeds=args.seeds, documents=len(clean),
                  evidence='Replication on the same exposed development documents; target seeds and documents resampled jointly; source and query family fixed.',
                  estimand='Mean target-wise response nRMSE', methods={}, contrasts={}, per_target={})
    for family, predicate in [('singletons', lambda q:'+' not in q and q != 'full'),
                              ('pairs', lambda q:'+' in q), ('full', lambda q:q == 'full')]:
        ids = [i for i,q in enumerate(queries) if predicate(q)]
        denominator = ((source[:, ids]-clean[:, None])**2).sum(1)
        samples, estimates = {}, {}
        for method in methods:
            errors = np.stack([((b[method][:, ids]-source[:, ids])**2).sum(1) for b in banks])
            by_seed = np.sqrt(errors.sum(1)/denominator.sum())
            estimates[method] = float(by_seed.mean())
            boot = np.sqrt(errors[:, draws].sum(2)/denominator[draws].sum(1)).T
            samples[method] = np.take_along_axis(boot, seed_draws, axis=1).mean(1)
            result['methods'].setdefault(method, {})[family] = dict(mean=estimates[method],
                ci95=np.quantile(samples[method], [.025, .975]).tolist())
            result['per_target'].setdefault(method, {})[family] = by_seed.tolist()
        for control in ['native', 'local_whole', 'raw']:
            key = control+' minus local_parts'
            result['contrasts'].setdefault(key, {})[family] = dict(
                mean=estimates[control]-estimates['local_parts'],
                ci95=np.quantile(samples[control]-samples['local_parts'], [.025, .975]).tolist())
    result['inputs'] = [str(p) for p in args.analyses]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    if args.table:
        labels = ['Initial relation', 'Calibrated geometry', 'Source-direction readout',
                  'Local complete-action training', 'Local part-action training']
        lines = [r'\begin{tabular}{lrrr}', r'\toprule',
                 r'Method & Single parts & Pairs & Full \\', r'\midrule']
        for method, label in zip(methods, labels):
            numbers = [result['methods'][method][f]['mean'] for f in ['singletons','pairs','full']]
            lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in numbers)+r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}'])
        with args.table.open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
