from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

from naacl_analyze_e1_shift import FAMILIES, load

DESCRIPTION = ('Part, complete and union errors of the Pythia-70M SHIFT correspondences of CONTAINMENT experiments 2 and '
               '2b, with natural-text errors and a bootstrap over target dictionaries, training seeds within each target '
               'and biographies within label-by-gender strata.')
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
SEEDS = (42, 43, 44)
TRAINED = ['parts_relation', 'natural_relation', 'natural_matched', 'parts_relation_cap4', 'natural_relation_cap4']
FIXED = ['hungarian_decoder', 'geometry_gain', 'native', 'raw_reconstruction']
CONTRASTS = [('natural_relation', 'parts_relation'), ('natural_relation', 'hungarian_decoder'),
             ('natural_relation', 'natural_matched'), ('natural_relation_cap4', 'natural_matched'),
             ('natural_relation_cap4', 'parts_relation_cap4'), ('parts_relation', 'hungarian_decoder')]


def family_error(values, method, queries, idx):
    base = values[('none', 'full')][idx]
    num = sum(((values[(method, q)][idx] - values[('source', q)][idx]) ** 2).sum() for q in queries)
    den = sum(((values[('source', q)][idx] - base) ** 2).sum() for q in queries)
    return np.sqrt(num / den)


def merged(target, versions):
    # Both evaluations score the same panel with the same unedited and source passes; their other methods are joined.
    docs, values, labels, genders = None, {}, None, None
    for run in versions:
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        d, v, lab, gen = load(run)
        if docs is None:
            docs, labels, genders = d, lab, gen
        assert d == docs and np.array_equal(lab, labels) and np.array_equal(gen, genders)
        for key in [k for k in v if k[0] in ('none', 'source') and k in values]:
            assert np.allclose(v[key], values[key], atol=1e-5), (target, key)
        for (method, query), arr in v.items():
            values[(method.replace('_cap4_ts', '_cap4@').replace('_ts', '@'), query)] = arr
    return docs, values, labels, genders


def method_error(values, method, seed, queries, idx):
    key = f'{method}@{seed}' if method in TRAINED else method
    return family_error(values, key, queries, idx)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'artifacts/containment/C2_PYTHIA.json')
    args = parser.parse_args()
    data, natural = {}, {}
    for t in (2, 3, 4, 5):
        data[t] = merged(t, [RUNS / f'CSM_C2_PYTHIA70M_eval_t{t}_v1_20261011', RUNS / f'CSM_C2B_PYTHIA70M_eval_t{t}_v1_20261011'])
        natural[t] = {}
        for s in SEEDS:
            a = json.loads((RUNS / f'CSM_C2_PYTHIA70M_program_t{t}_ts{s}_v1_20261011' / 'NATURAL.json').read_text())
            b = json.loads((RUNS / f'CSM_C2B_PYTHIA70M_program_t{t}_ts{s}_v1_20261011' / 'NATURAL.json').read_text())
            natural[t][s] = dict(a, parts_relation_cap4=b['parts_relation'], natural_relation_cap4=b['natural_relation'])
    docs0, _, labels, genders = data[2]
    for docs, _, lab, gen in data.values():
        assert docs == docs0 and np.array_equal(lab, labels) and np.array_equal(gen, genders)
    methods = TRAINED + FIXED
    everything = np.arange(len(docs0))
    point = {m: {f: [[float(method_error(data[t][1], m, s, q, everything)) for s in SEEDS] for t in data]
                 for f, q in FAMILIES.items()} for m in methods}
    summary = {m: {f: float(np.mean(point[m][f])) for f in FAMILIES} for m in methods}
    nat_names = ['initial', 'hungarian_decoder'] + TRAINED
    nat_summary = {m: float(np.mean([natural[t][s][m] for t in natural for s in SEEDS])) for m in nat_names}
    rng = np.random.default_rng(20261011)
    strata = [np.flatnonzero((labels == y) & (genders == g)) for y in (0, 1) for g in (0, 1)]
    targets = list(data)
    boot = {m: {f: np.empty(args.draws) for f in FAMILIES} for m in methods}
    for b in range(args.draws):
        picks = [(targets[i], SEEDS[j]) for i in rng.integers(len(targets), size=len(targets))
                 for j in rng.integers(len(SEEDS), size=len(SEEDS))]
        idx = np.concatenate([rng.choice(s, size=len(s)) for s in strata])
        for m in methods:
            for f, q in FAMILIES.items():
                boot[m][f][b] = np.mean([method_error(data[t][1], m, s, q, idx) for t, s in picks])
    intervals = {m: {f: np.quantile(boot[m][f], [.025, .975]).tolist() for f in FAMILIES} for m in methods}
    contrasts = {f'{a}-{c}': {f: dict(point=summary[a][f] - summary[c][f],
                                      interval=np.quantile(boot[a][f] - boot[c][f], [.025, .975]).tolist())
                              for f in FAMILIES} for a, c in CONTRASTS}
    d1 = dict(ratio_to_part_supervision=summary['natural_relation']['parts'] / summary['parts_relation']['parts'],
              below_hungarian=summary['natural_relation']['parts'] < summary['hungarian_decoder']['parts'],
              below_one_to_one=summary['natural_relation']['parts'] < summary['natural_matched']['parts'])
    d1['pass'] = bool(d1['ratio_to_part_supervision'] <= 1.15 and d1['below_hungarian'] and d1['below_one_to_one'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                           draws=args.draws, summary=summary, intervals=intervals, contrasts=contrasts,
                                           per_target_seed=point, natural=nat_summary, natural_per_target_seed=natural,
                                           d1=d1), indent=1))
    for m in methods:
        print(f'{m:<24}', '  '.join(f'{f} {summary[m][f]:.3f}' for f in FAMILIES), f"  natural {nat_summary.get(m, float('nan')):.3f}")
    for k, v in contrasts.items():
        print(f'{k:<46}', '  '.join(f"{f} {c['point']:+.3f} [{c['interval'][0]:+.3f}, {c['interval'][1]:+.3f}]" for f, c in v.items()))
    print('D1 Pythia', d1)


if __name__ == '__main__':
    main()
