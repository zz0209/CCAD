import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from p1const import ANIMALS, DATASETS, SECOND_SEED, run_dir, save_json

DESCRIPTION = ('P1 analysis under the frozen protocol. Actual shift = seed-0 fine-tuned score minus base score on the '
               'held-out battery; sigma_t per behaviour = pooled seed SD from the four two-seed sets; z = shift / '
               'sigma_t. Each forecaster predicts in its own units; hyperparameters (layer, alpha or variant) are '
               'chosen by leave-one-dataset-out on pooled Spearman over the other 11 sets. Reports pooled Spearman, '
               'sign accuracy on |z| > 2, teacher-animal ranks, the validity fraction and the kill decision. Writes '
               'analysis/summary.json and analysis/table.json.')
BASELINES = ['mdf', 'pp', 'ga', 'emb', 'rev']
TEACHER = {'sub_owl': 'owl', 'sub_dolphin': 'dolphin'}


def rho(p, z):
    return float(spearmanr(p.ravel(), z.ravel())[0])


def sign_acc(p, z, thr=2.0):
    m = np.abs(z) > thr
    return float(np.mean(np.sign(p[m]) == np.sign(z[m]))), int(m.sum())


def lodo(variants, z):
    keys = list(variants)
    out = np.zeros_like(z)
    chosen = {}
    for i, d in enumerate(DATASETS):
        others = [j for j in range(len(DATASETS)) if j != i]
        scores = [rho(variants[k][others], z[others]) for k in keys]
        best = keys[int(np.nanargmax(scores))]
        out[i] = variants[best][i]
        chosen[d] = dict(variant=best, train_rho=float(np.nanmax(scores)))
    return out, chosen


def bootstrap(pred_by_f, z, reps, seed):
    rng = np.random.default_rng(seed)
    out = {f: [] for f in pred_by_f}
    for _ in range(reps):
        idx = rng.integers(0, len(DATASETS), len(DATASETS))
        for f, p in pred_by_f.items():
            out[f].append(rho(p[idx], z[idx]))
    return {f: np.array(v) for f, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--root', type=Path, default=None, help='read a copy of the run directory from here')
    parser.add_argument('--partial', action='store_true',
                        help='evaluate with the baselines whose forecasts exist; the margin criterion stays pending '
                             'until all five exist. Writes analysis/summary_partial.json.')
    args = parser.parse_args()
    rd = args.root if args.root is not None else run_dir(args.run)
    cfg = json.loads((rd / 'config.json').read_text())
    battery = json.loads((rd / 'battery.json').read_text())
    names = [b['name'] for b in battery['behaviours']]
    kinds = np.array([b['kind'] for b in battery['behaviours']])
    base = json.loads((rd / 'base_battery_eval.json').read_text())['scores']
    b0 = np.array([base[n] for n in names])

    def ft(d, s):
        sc = json.loads((rd / 'train' / f'{d}_s{s}' / 'battery.json').read_text())['scores']
        return np.array([sc[n] for n in names])

    S0 = np.stack([ft(d, 0) for d in DATASETS])
    S1 = {d: ft(d, 1) for d in SECOND_SEED}
    shift = S0 - b0
    sigma = np.sqrt(np.mean([(S0[DATASETS.index(d)] - S1[d]) ** 2 / 2 for d in SECOND_SEED], 0))
    assert (sigma > 0).all()
    z = shift / sigma
    frac_sig = float(np.mean(np.abs(z) > 2))

    preds = {}
    alpha_c = cfg['alpha_c']
    for kind in ['sc', 'mdf']:
        preds[kind] = {}
        for l in cfg['layers']:
            for c in range(len(alpha_c)):
                P = np.zeros_like(z)
                for i, d in enumerate(DATASETS):
                    st = json.loads((rd / 'forecast' / 'steer' / kind / d / f'l{l}_c{c}.json').read_text())['scores']
                    P[i] = np.array([st[n] for n in names]) - b0
                preds[kind][f'l{l}_c{alpha_c[c]}'] = P
    missing = [f for f in BASELINES if not (rd / 'forecast' / f'{f}.json').exists() and f != 'mdf']
    assert args.partial or not missing, missing
    for f in ['pp', 'ga', 'emb']:
        if f in missing:
            continue
        pr = json.loads((rd / 'forecast' / f'{f}.json').read_text())['pred']
        preds[f] = {v: np.array([[pr[v][d][n] for n in names] for d in DATASETS]) for v in pr}
    if 'rev' not in missing:
        rev = json.loads((rd / 'forecast' / 'rev.json').read_text())['pred']
        preds['rev'] = {'expected_rating': np.array([[rev[d][n] for n in names] for d in DATASETS])}
    for f in preds:
        for v, P in preds[f].items():
            assert np.isfinite(P).all(), (f, v)
    bases = [f for f in BASELINES if f in preds]

    res, lodo_pred = {}, {}
    for f in ['sc'] + bases:
        P, chosen = lodo(preds[f], z)
        lodo_pred[f] = P
        acc, n_sig = sign_acc(P, z)
        fam = {k: rho(P[:, kinds == k], z[:, kinds == k]) for k in ['persona', 'animal']}
        oracle = {v: rho(Q, z) for v, Q in preds[f].items()}
        sub = {}
        for d, a in TEACHER.items():
            i = DATASETS.index(d)
            cols = [names.index(f'animal_{x}') for x in ANIMALS]
            pa, za = P[i, cols], z[i, cols]
            t = ANIMALS.index(a)
            sub[d] = dict(teacher=a, transfer=bool(za[t] > 2), z_teacher=float(za[t]),
                          actual_rank=int(1 + np.sum(za > za[t])), predicted_rank=int(1 + np.sum(pa > pa[t])))
        res[f] = dict(pooled_rho=rho(P, z), sign_accuracy=acc, n_pairs_abs_z_gt_2=n_sig, rho_by_family=fam,
                      lodo_choices=chosen, rho_fixed_variant=oracle,
                      rho_without_animals=rho(P[:, kinds != 'animal'], z[:, kinds != 'animal']), subliminal=sub)

    # reference outside the protocol's baseline list: each set's forecast is the mean z of the other 11 sets
    # (the shift that fine-tuning on anything produces); it shows how much of rho needs no knowledge of the set
    ref = np.stack([z[[j for j in range(len(DATASETS)) if j != i]].mean(0) for i in range(len(DATASETS))])
    ref_acc, _ = sign_acc(ref, z)
    reference = dict(other_sets_mean_shift=dict(pooled_rho=rho(ref, z), sign_accuracy=ref_acc))
    boot = bootstrap(lodo_pred, z, cfg['analysis']['bootstrap_reps'], 0)
    best_base = max(bases, key=lambda f: res[f]['pooled_rho'])
    margin = res['sc']['pooled_rho'] - res[best_base]['pooled_rho']
    marg_boot = boot['sc'] - boot[best_base]
    sc_sub = res['sc']['subliminal']
    k = cfg['kill']
    # with baselines missing, a margin already below the threshold is final (more baselines can only lower it)
    margin_hit = margin < k['min_margin']
    kill = dict(
        pooled_rho_below=dict(value=res['sc']['pooled_rho'], threshold=k['min_rho'],
                              triggered=res['sc']['pooled_rho'] < k['min_rho']),
        margin_below=dict(value=margin, best_baseline=best_base, threshold=k['min_margin'], missing_baselines=missing,
                          triggered=margin_hit if (margin_hit or not missing) else None),
        sign_accuracy_below=dict(value=res['sc']['sign_accuracy'], threshold=k['min_sign_acc'],
                                 triggered=res['sc']['sign_accuracy'] < k['min_sign_acc']),
        subliminal_both_miss=dict(sets=sc_sub, top_k=k['teacher_top_k'],
                                  evaluable=all(s['transfer'] for s in sc_sub.values()),
                                  triggered=all(s['transfer'] and s['predicted_rank'] > k['teacher_top_k']
                                                for s in sc_sub.values())),
    )
    valid = frac_sig >= k['min_frac_significant']
    trig = [v['triggered'] for v in kill.values()]
    decision = ('INVALID_SCALE_UP' if not valid else 'KILL' if any(t is True for t in trig) else
                'PENDING' if any(t is None for t in trig) else 'PASS')
    summary = dict(
        run_id=cfg['run_id'], decision=decision,
        validity=dict(frac_pairs_abs_z_gt_2=frac_sig, threshold=k['min_frac_significant'], valid=valid,
                      n_pairs=int(z.size)),
        kill=kill,
        forecasters={f: {kk: v for kk, v in r.items() if kk != 'rho_fixed_variant'} for f, r in res.items()},
        fixed_variant_rho={f: r['rho_fixed_variant'] for f, r in res.items()},
        reference_not_in_criteria=reference,
        bootstrap_over_datasets=dict(reps=cfg['analysis']['bootstrap_reps'],
                                     rho_ci95={f: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                                               for f, v in boot.items()},
                                     margin_ci95=[float(np.quantile(marg_boot, .025)),
                                                  float(np.quantile(marg_boot, .975))],
                                     p_margin_ge_threshold=float(np.mean(marg_boot >= k['min_margin']))),
        sigma_t=dict(zip(names, sigma.tolist())),
        actual=dict(per_dataset_frac_abs_z_gt_2={d: float(np.mean(np.abs(z[i]) > 2)) for i, d in enumerate(DATASETS)},
                    per_behaviour_frac_abs_z_gt_2={n: float(np.mean(np.abs(z[:, j]) > 2)) for j, n in enumerate(names)}),
    )
    tag = '_partial' if args.partial else ''
    save_json(rd / 'analysis' / f'summary{tag}.json', summary)
    save_json(rd / 'analysis' / f'table{tag}.json', dict(datasets=DATASETS, behaviours=names, base=b0.tolist(),
                                                         shift=shift.tolist(), z=z.tolist(),
                                                         lodo_pred={f: P.tolist() for f, P in lodo_pred.items()}))
    print(json.dumps(dict(decision=decision, validity=summary['validity'],
                          rho={f: round(r['pooled_rho'], 3) for f, r in res.items()},
                          sign_acc={f: round(r['sign_accuracy'], 3) for f, r in res.items()},
                          margin=round(margin, 3), best_baseline=best_base, reference=reference,
                          subliminal=sc_sub), indent=1))


if __name__ == '__main__':
    main()
