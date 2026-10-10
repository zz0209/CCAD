from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

from naacl_analyze_e1_shift import FAMILIES, load

DESCRIPTION = ('Pythia-70M SHIFT correspondences over three training seeds per target dictionary, with a bootstrap that '
               'resamples target dictionaries, training seeds within each target and biographies within strata.')
BULK = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010')
METHODS = ['whole', 'random_parts', 'parts_relation', 'parts']
CONTRASTS = [('parts', 'whole'), ('random_parts', 'whole'), ('parts_relation', 'whole'), ('parts', 'random_parts')]


def runs_for(target):
    return [BULK / f'NAACL_E1_shift_r59panel_seed{target}_v2_20261010'] + \
           [BULK / f'NAACL_E14_shift_eval_t{target}_ts{ts}_v1_20261010' for ts in (43, 44)]


def family_error(values, method, queries, idx):
    base = values[('none', 'full')][idx]
    num = sum(((values[(method, q)][idx] - values[('source', q)][idx]) ** 2).sum() for q in queries)
    den = sum(((values[('source', q)][idx] - base) ** 2).sum() for q in queries)
    return np.sqrt(num / den)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'artifacts/naacl_sprint_20261010/e1_matching/E14_SEEDS.json')
    args = parser.parse_args()
    data = {}
    for target in (2, 3, 4, 5):
        data[target] = []
        for run in runs_for(target):
            assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
            data[target].append(load(run))
    docs0, _, labels, genders = data[2][0]
    for runs in data.values():
        for docs, _, lab, gen in runs:
            assert docs == docs0 and np.array_equal(lab, labels) and np.array_equal(gen, genders)
    everything = np.arange(len(docs0))
    point = {m: {f: [[float(family_error(v, m, q, everything)) for _, v, _, _ in data[t]] for t in data]
                 for f, q in FAMILIES.items()} for m in METHODS}
    summary = {m: {f: dict(mean=float(np.mean(point[m][f])),
                           seed_sd_within_target=float(np.mean([np.std(row, ddof=1) for row in point[m][f]])),
                           per_target_seed=point[m][f]) for f in FAMILIES} for m in METHODS}
    rng = np.random.default_rng(20261010)
    strata = [np.flatnonzero((labels == y) & (genders == g)) for y in (0, 1) for g in (0, 1)]
    targets = list(data)
    boot = {m: {f: np.empty(args.draws) for f in FAMILIES} for m in METHODS}
    for b in range(args.draws):
        picks = [(targets[t], s) for t in rng.integers(len(targets), size=len(targets))
                 for s in rng.integers(3, size=3)]
        idx = np.concatenate([rng.choice(s, size=len(s)) for s in strata])
        for m in METHODS:
            for f, q in FAMILIES.items():
                boot[m][f][b] = np.mean([family_error(data[t][s][1], m, q, idx) for t, s in picks])
    contrasts = {f'{a}-{c}': {f: dict(point=summary[a][f]['mean'] - summary[c][f]['mean'],
                                      interval=np.quantile(boot[a][f] - boot[c][f], [.025, .975]).tolist())
                              for f in FAMILIES} for a, c in CONTRASTS}
    intervals = {m: {f: np.quantile(boot[m][f], [.025, .975]).tolist() for f in FAMILIES} for m in METHODS}
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                           draws=args.draws, summary=summary, intervals=intervals, contrasts=contrasts,
                                           runs={t: [str(r) for r in runs_for(t)] for t in targets}), indent=1))
    for m in METHODS:
        print(f'{m:<16}', '  '.join(f"{f} {summary[m][f]['mean']:.3f} (seed sd {summary[m][f]['seed_sd_within_target']:.3f})"
                                     for f in FAMILIES))
    for k, v in contrasts.items():
        print(f'{k:<26}', '  '.join(f"{f} {c['point']:+.3f} [{c['interval'][0]:+.3f}, {c['interval'][1]:+.3f}]" for f, c in v.items()))


if __name__ == '__main__':
    main()
