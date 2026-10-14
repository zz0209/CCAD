import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from p3_common import LETTERS, read_jsonl, tv

DESCRIPTION = ('P3 evaluation under the frozen protocol: resampled answer distributions P_i per test boundary (valid '
               'answers only), step importance I_i = TV(P_i, P_i+1), split-half noise ceilings, and for the probe and '
               'each baseline the mean boundary TV to P_i, the mean per-trace Spearman correlation between predicted '
               'and actual importance, and recall of the top 3 steps; then the three kill criteria.')
N_SPLITS = 50
N_BOOT = 2000


def dist(answers):
    v = [a for a in answers if a is not None]
    if not v:
        return None
    return np.array([v.count(c) for c in LETTERS], float) / len(v)


def rank_metrics(pred, true):
    pred = np.where(np.isnan(pred), -np.inf, pred)
    if np.all(pred == pred[0]):
        rho = 0.0
    else:
        finite = np.isfinite(pred)
        p = pred.copy()
        p[~finite] = np.min(p[finite]) - 1 if finite.any() else 0
        rho = float(spearmanr(p, true).statistic)
    top_t = set(np.argsort(-true, kind='stable')[:3].tolist())
    top_p = set(np.argsort(-pred, kind='stable')[:3].tolist())
    return rho, len(top_t & top_p) / 3


def boot(values, rng, fn=np.mean):
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    stats = np.array([fn(values[i]) for i in idx])
    return [float(np.quantile(stats, .025)), float(np.quantile(stats, .975))]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['probe', 'forced', 'reader', 'receiver'])
    args = parser.parse_args()
    rd = args.run_dir
    items = {t['id']: t for t in json.loads((rd / 'items.json').read_text())['test']}
    labels = {r['id']: r for r in read_jsonl(rd / 'resample.jsonl')}
    assert 'probe' in args.methods
    dist_methods = {name: {t['id']: np.array(t['probs']) for t in json.loads((rd / f'{name}.json').read_text())['traces']}
                    for name in ['probe', 'forced', 'reader'] if name in args.methods}
    receiver = {t['id']: np.array([np.nan if x is None else x for x in t['step_scores']], float)
                for t in json.loads((rd / 'receiver.json').read_text())['traces']} if 'receiver' in args.methods else None
    rng = np.random.default_rng(0)
    ids = [i for i in items if i in labels]
    assert len(ids) == len(items), 'missing resampling labels'
    per_trace, tv_rows = [], {m: [] for m in dist_methods}
    half_tv, n_cont, n_valid = [], 0, 0
    half_rho = np.full((N_SPLITS, len(ids)), np.nan)
    for k, qid in enumerate(ids):
        ans = labels[qid]['answers']
        n_cont += sum(len(a) for a in ans)
        n_valid += sum(x is not None for a in ans for x in a)
        P = [dist(a) for a in ans]
        ok_b = np.array([p is not None for p in P])
        steps = [j for j in range(len(P) - 1) if ok_b[j] and ok_b[j + 1]]
        I = np.array([tv(P[j], P[j + 1]) for j in steps])
        for m, d in dist_methods.items():
            tv_rows[m] += [tv(d[qid][j], P[j]) for j in range(len(P)) if ok_b[j]]
        for s in range(N_SPLITS):
            halves = []
            for a in ans:
                perm = rng.permutation(len(a))
                halves.append((dist([a[i] for i in perm[:len(a) // 2]]), dist([a[i] for i in perm[len(a) // 2:]])))
            half_tv += [tv(a, b) for a, b in halves if a is not None and b is not None]
            if len(steps) >= 4 and np.ptp(I) > 0:
                ia = np.array([tv(halves[j][0], halves[j + 1][0]) if halves[j][0] is not None
                               and halves[j + 1][0] is not None else np.nan for j in steps])
                ib = np.array([tv(halves[j][1], halves[j + 1][1]) if halves[j][1] is not None
                               and halves[j + 1][1] is not None else np.nan for j in steps])
                ok = ~np.isnan(ia) & ~np.isnan(ib)
                if ok.sum() >= 4 and np.ptp(ia[ok]) > 0 and np.ptp(ib[ok]) > 0:
                    half_rho[s, k] = spearmanr(ia[ok], ib[ok]).statistic
        row = dict(id=qid, subject=items[qid]['subject'], n_boundaries=len(P), n_steps=len(steps),
                   valid_fraction=[float(np.mean([x is not None for x in a])) for a in ans],
                   true_importance=I.tolist(), included=bool(len(steps) >= 4 and np.ptp(I) > 0))
        if row['included']:
            preds = {m: np.array([tv(d[qid][j], d[qid][j + 1]) for j in steps]) for m, d in dist_methods.items()}
            if receiver is not None:
                preds['receiver'] = receiver[qid][steps]
            for m, p in preds.items():
                row[f'{m}_rho'], row[f'{m}_recall3'] = rank_metrics(p, I)
                row[f'{m}_pred_importance'] = [None if np.isnan(x) else float(x) for x in p]
        per_trace.append(row)
    inc = [r for r in per_trace if r['included']]
    methods = [m for m in ['probe', 'forced', 'reader', 'receiver'] if m in args.methods]
    baselines = [m for m in methods if m != 'probe']
    res = {}
    for m in methods:
        res[m] = {}
        if m in tv_rows:
            res[m]['tv'] = float(np.mean(tv_rows[m]))
            res[m]['tv_ci'] = boot(tv_rows[m], rng)
        if inc:
            res[m].update(rho=float(np.mean([r[f'{m}_rho'] for r in inc])),
                          rho_ci=boot([r[f'{m}_rho'] for r in inc], rng),
                          recall3=float(np.mean([r[f'{m}_recall3'] for r in inc])),
                          recall3_ci=boot([r[f'{m}_recall3'] for r in inc], rng))
    summary = dict(
        methods_evaluated=methods, n_test_traces=len(ids), n_traces_in_rank_metrics=len(inc),
        n_boundaries=sum(r['n_boundaries'] for r in per_trace), n_steps_in_rank_metrics=sum(r['n_steps'] for r in inc),
        n_continuations=n_cont, valid_answer_fraction=n_valid / n_cont,
        noise_ceiling=dict(split_half_tv=float(np.mean(half_tv)),
                           split_half_importance_rho=float(np.nanmean(np.nanmean(half_rho, axis=1)))
                           if np.isfinite(half_rho).any() else None),
        methods=res)
    if 'forced' in methods:
        summary['probe_tv_over_forced_tv'] = res['probe']['tv'] / res['forced']['tv']
    if not inc:
        summary['decision'] = 'NO_RANK_METRICS (fewer than 4 valid steps in every trace)'
    else:
        # A criterion that cannot be computed from the methods given is None; any True criterion kills, and PASS
        # needs all three computed with every baseline present.
        kills = dict(rho_below_0_35=res['probe']['rho'] < 0.35,
                     rho_below_best_baseline_plus_0_10=None,
                     tv_not_20pct_below_forced=res['probe']['tv'] > 0.8 * res['forced']['tv']
                     if 'forced' in methods else None)
        if baselines:
            base = max(baselines, key=lambda m: res[m]['rho'])
            diff = [r['probe_rho'] - r[f'{base}_rho'] for r in inc]
            if res['probe']['rho'] < res[base]['rho'] + 0.10 or len(baselines) == 3:
                kills['rho_below_best_baseline_plus_0_10'] = res['probe']['rho'] < res[base]['rho'] + 0.10
            summary.update(best_baseline_by_rho=base, probe_minus_best_baseline_rho=float(np.mean(diff)),
                           probe_minus_best_baseline_rho_ci=boot(diff, rng))
        summary['kill_criteria'] = kills
        if any(v is True for v in kills.values()):
            summary['decision'] = 'KILL'
        elif all(v is False for v in kills.values()):
            summary['decision'] = 'PASS'
        else:
            summary['decision'] = 'UNDECIDED (criteria not computable from the methods given: ' + ', '.join(
                k for k, v in kills.items() if v is None) + ')'
    args.out.write_text(json.dumps(dict(summary=summary, per_trace=per_trace), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
