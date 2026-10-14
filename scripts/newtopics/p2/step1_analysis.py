import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from summarize import write_json

DESCRIPTION = ('P2 formal-study step 1 decision. For each checkpoint, on the correctly answered development prompts, '
               'swap success of the J-lens is compared with the converged tuned lens at natural norm (tuned) and at the '
               'J-swap norm (tuned_nm). Intervals: 95% percentile cluster bootstrap over bridge countries (10,000 '
               'resamples, seed 0), paired on the same prompts. Tests: exact two-sided McNemar, Holm-adjusted over the '
               'primary contrasts of all checkpoints. Step 1 passes when at least one checkpoint with n >= 20 has a '
               'lower bound > 0 and a Holm-adjusted p < 0.05 for both primary contrasts; otherwise the lens-comparison '
               'line stops. Logit-lens contrasts are reported without adjustment.')
PRIMARY = [('jlens', 'tuned'), ('jlens', 'tuned_nm')]
SECONDARY = [('jlens', 'logit'), ('jlens', 'logit_nm')]
N_BOOT = 10000
SEED = 0
MIN_N = 20


def cluster_boot(groups, values, rng):
    # values: dict method -> 0/1 array; returns bootstrap draws of each method's success rate, resampling clusters.
    clusters = sorted(set(groups))
    idx = {c: i for i, c in enumerate(clusters)}
    g = np.array([idx[c] for c in groups])
    n_c = np.bincount(g, minlength=len(clusters))
    sums = {m: np.bincount(g, weights=v, minlength=len(clusters)) for m, v in values.items()}
    draws = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=N_BOOT)
    tot = draws @ n_c
    return {m: (draws @ s) / tot for m, s in sums.items()}


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True, help='label=SWAP.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    per, tests = {}, []
    for spec in args.runs:
        label, path = spec.split('=')
        d = json.loads(Path(path).read_text())
        methods = sorted({m for a, b in PRIMARY + SECONDARY for m in (a, b)})
        missing = [m for m in methods if m not in d['methods']]
        if missing:
            raise ValueError(f'{label}: swap file lacks methods {missing}')
        ok = [r for r in d['rows'] if r['correct']]
        vals = {m: np.array([r['swap'][m]['success'] for r in ok], dtype=bool) for m in methods}
        groups = [r['bridge'] for r in ok]
        rng = np.random.default_rng(SEED)
        boot = cluster_boot(groups, {m: v.astype(float) for m, v in vals.items()}, rng)
        rates = {m: dict(k=int(v.sum()), n=len(v), rate=float(v.mean()) if len(v) else None,
                         ci95=[float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))])
                 for m, v in vals.items()}
        contrasts = {}
        for a, b in PRIMARY + SECONDARY:
            diff = boot[a] - boot[b]
            n01, n10, p = mcnemar(vals[a], vals[b])
            contrasts[f'{a}-{b}'] = dict(primary=(a, b) in PRIMARY, delta=float(vals[a].mean() - vals[b].mean()),
                                         ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                         a_only=n01, b_only=n10, mcnemar_p=float(p))
            if (a, b) in PRIMARY:
                tests.append((label, f'{a}-{b}', p))
        norms = {m: float(np.median([r['swap'][m]['pert_norm'] for r in ok])) for m in methods}
        per[label] = dict(n=len(ok), n_clusters=len(set(groups)), rates=rates, contrasts=contrasts,
                          median_pert_norm=norms, eligible=len(ok) >= MIN_N)
    reject, p_adj, _, _ = multipletests([p for _, _, p in tests], alpha=0.05, method='holm')
    for (label, name, _), rj, pa in zip(tests, reject, p_adj):
        per[label]['contrasts'][name]['holm_p'] = float(pa)
    passing = [lb for lb, s in per.items() if s['eligible'] and all(
        s['contrasts'][f'{a}-{b}']['ci95'][0] > 0 and s['contrasts'][f'{a}-{b}']['holm_p'] < 0.05 for a, b in PRIMARY)]
    decision = dict(rule='pass if some checkpoint with n >= 20 has lower bound > 0 and Holm p < 0.05 for both '
                         'jlens-tuned and jlens-tuned_nm', checkpoints_passing=passing,
                    verdict='PASS_STEP1' if passing else 'STOP_LENS_COMPARISON')
    write_json(args.out, dict(per_checkpoint=per, decision=decision, n_boot=N_BOOT, seed=SEED, min_n=MIN_N))
    for lb, s in per.items():
        print(lb, 'n', s['n'], {k: (round(c['delta'], 3), [round(x, 3) for x in c['ci95']], round(c['mcnemar_p'], 4))
                                for k, c in s['contrasts'].items()})
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
