import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from step1_analysis import MIN_N, N_BOOT, SEED, cluster_boot, mcnemar
from summarize import write_json

DESCRIPTION = ('P2 formal-study step 2 decisions (protocol: artifacts/newtopics/p2/STEP2_PROTOCOL.md). Reads the '
               'per-checkpoint outputs under --root (<label>/swap_<set>.json, c4_<set>.json, h5_swap_<set>.json, '
               'eligibility.json) and decides H1-H5 and the stop rules of FORMAL_STUDY.md section 4. Unit: the '
               'correctly answered prompt. Intervals: 95% percentile cluster bootstrap over bridge countries (10,000 '
               'resamples, seed 0), paired on the same prompts. Tests: exact two-sided McNemar; Holm over the primary '
               'contrasts at each final checkpoint and prompt set. Thresholds are compared on integer counts. '
               '--mode stopcheck evaluates only the two v2 stop rules on OLMo-1B main and Pythia-2.8B.')
SETS = ['v1', 'v2']
OLMO_MAIN = 'olmo1b_main'
OLMO_STEPS = [10000, 20000, 50000, 100000, 200000, 430000, 900000, 1907359]
OLMO_INTER = [f'olmo1b_step{s}' for s in OLMO_STEPS]
PYTHIA = ['pythia410m', 'pythia14b', 'pythia28b']
FINALS = [OLMO_MAIN] + PYTHIA
H5_LABELS = [OLMO_MAIN, 'pythia410m']
LENS_X = ['logit', 'logit_nm', 'tmatch', 'tmatch_nm', 'tuned', 'tuned_nm']
C4 = ['dim', 'dim_outside_j', 'dim_outside_j_clamp', 'random']


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def stats(groups, vals, pairs):
    # Success rates and paired contrasts with cluster-bootstrap intervals and exact McNemar tests.
    n = len(groups)
    boot = cluster_boot(groups, {m: v.astype(float) for m, v in vals.items()}, np.random.default_rng(SEED)) \
        if n else None
    rates = {m: dict(k=int(v.sum()), n=n, rate=int(v.sum()) / n if n else None, ci95=ci(boot[m]) if n else None)
             for m, v in vals.items()}
    con = {}
    for a, b in pairs:
        n01, n10, p = mcnemar(vals[a], vals[b])
        con[f'{a}-{b}'] = dict(k_diff=int(vals[a].sum()) - int(vals[b].sum()), n=n,
                               delta=(int(vals[a].sum()) - int(vals[b].sum())) / n if n else None,
                               ci95=ci(boot[a] - boot[b]) if n else None, a_only=n01, b_only=n10, mcnemar_p=float(p))
    return rates, con


def at_least(c, margin_num, margin_den):
    # delta >= margin_num / margin_den on integer counts, and bootstrap lower bound > 0.
    return c['n'] >= MIN_N and margin_den * c['k_diff'] >= margin_num * c['n'] and c['ci95'][0] > 0


def tl_info(path):
    d = load(str(path) + '.json') if path else None
    if d is None:
        return None
    kl = d['heldout_kl']
    return dict(file=str(path), recipe=d['recipe'], all_converged=d['all_converged'],
                steps=[g['steps'] for g in d['group_meta']], train_tokens=[g['train_tokens'] for g in d['group_meta']],
                conv_kl_first_last=[[g['conv_kl'][min(g['conv_kl'], key=int)], g['conv_kl'][max(g['conv_kl'], key=int)]]
                                    for g in d['group_meta']],
                heldout_kl_mean=dict(tuned=float(np.mean([v['tuned'] for v in kl.values()])),
                                     logit=float(np.mean([v['logit'] for v in kl.values()]))))


def lens_block(d, j_override=None):
    ok = [r for r in d['rows'] if r['correct']]
    vals = {m: np.array([r['swap'][m]['success'] for r in ok], dtype=bool) for m in ['jlens'] + LENS_X}
    if j_override is not None:
        vals['jlens'] = np.array([j_override[r['id']] for r in ok], dtype=bool)
    groups = [r['bridge'] for r in ok]
    rates, con = stats(groups, vals, [('jlens', x) for x in LENS_X])
    readout = {lens: dict(median_best_rank=float(np.median([r['bridge_rank_min'][lens] for r in ok])) if ok else None,
                          top5_frac=float(np.mean([r['bridge_rank_min'][lens] < 5 for r in ok])) if ok else None)
               for lens in ('jlens', 'logit', 'tuned')}
    return dict(n=len(ok), n_items=len(d['rows']), n_clusters=len(set(groups)), rates=rates, contrasts=con,
                h1_rule=all(at_least(c, 1, 10) for c in con.values()),
                h3_rule=all(at_least(con[f'jlens-{x}'], 1, 10) for x in ('tuned', 'tuned_nm')),
                median_pert_norm={m: float(np.median([r['swap'][m]['pert_norm'] for r in ok])) if ok else None
                                  for m in ['jlens'] + LENS_X},
                bridge_readout_correct_prompts=readout, j_success_by_id={r['id']: bool(v) for r, v in
                                                                          zip(ok, vals['jlens'])},
                lenses=dict(jlens=d['jlens'], tl_full=tl_info(d['tuned']), tl_match=tl_info(d.get('tuned_match'))))


def h5_block(orig, refit):
    ok = [r for r in orig['rows'] if r['correct']]
    ref = {r['id']: r for r in refit['rows']}
    if any(ref[r['id']]['correct'] != r['correct'] for r in orig['rows']):
        raise ValueError('H5: refit swap file disagrees on correctness')
    vals = dict(refit=np.array([ref[r['id']]['swap']['jlens']['success'] for r in ok], dtype=bool),
                orig=np.array([r['swap']['jlens']['success'] for r in ok], dtype=bool))
    rates, con = stats([r['bridge'] for r in ok], vals, [('refit', 'orig')])
    c = con['refit-orig']
    lower = 'refit' if vals['refit'].sum() < vals['orig'].sum() else 'orig'
    return dict(n=len(ok), rates=rates, contrast=c, stable=len(ok) > 0 and -0.10 < c['ci95'][0] and c['ci95'][1] < 0.10,
                lower_fit=lower, j_success_lower={r['id']: bool(v) for r, v in zip(ok, vals[lower])})


def c4_block(d):
    if not all(r['held_out'] for r in d['rows']):
        raise ValueError('C4 file contains training-pair prompts')
    ok = [r for r in d['rows'] if r['correct']]
    if any(r['swap'][m] is None for r in ok for m in C4):
        raise ValueError('C4: a correct held-out prompt lacks a concept vector')
    vals = {m: np.array([r['swap'][m]['success'] for r in ok], dtype=bool) for m in C4}
    rates, con = stats([r['bridge'] for r in ok], vals, [('dim', 'dim_outside_j'),
                                                         ('dim_outside_j', 'dim_outside_j_clamp'), ('dim', 'random')])
    k_n, k_c = rates['dim_outside_j']['k'], rates['dim_outside_j_clamp']['k']
    clamp_ok = k_n == 0 or 2 * (k_n - k_c) >= k_n
    ef = d['concept']['jspace_energy_frac']
    return dict(n=len(ok), rates=rates, contrasts=con, clamp_halves_nonj=clamp_ok,
                privilege=at_least(con['dim-dim_outside_j'], 3, 20) and clamp_ok,
                jspace_energy_frac_median=float(np.median([v for e in ef.values() for v in e.values()])),
                median_pert_norm={m: float(np.median([r['swap'][m]['pert_norm'] for r in ok])) if ok else None
                                  for m in C4})


def stop_check(lens):
    keys = (OLMO_MAIN, 'pythia28b')
    v2 = {k: lens[k]['v2'] for k in keys}
    return dict(
        whole_study_stop=all(10 * b['rates']['jlens']['k'] < 3 * b['n'] for b in v2.values()),
        lens_comparison_stop=all(b['contrasts']['jlens-tuned']['ci95'][0] <= 0 for b in v2.values()),
        v2={k: dict(n=b['n'], S_J=b['rates']['jlens']['rate'], S_J_ci95=b['rates']['jlens']['ci95'],
                    J_minus_TLfull=b['contrasts']['jlens-tuned']['delta'],
                    J_minus_TLfull_ci95=b['contrasts']['jlens-tuned']['ci95']) for k, b in v2.items()})


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--root', type=Path, required=True, help='artifacts/newtopics/p2/P2_STEP2_v1')
    parser.add_argument('--mode', choices=['stopcheck', 'full'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    labels = FINALS + OLMO_INTER
    raw = {lb: {s: load(args.root / lb / f'swap_{s}.json') for s in SETS} for lb in labels}
    if args.mode == 'stopcheck':
        lens = {lb: {s: lens_block(raw[lb][s]) for s in SETS} for lb in (OLMO_MAIN, 'pythia28b')}
        out = dict(stop_rules=stop_check(lens), n_boot=N_BOOT, seed=SEED, min_n=MIN_N)
        write_json(args.out, out)
        print(json.dumps(out['stop_rules']))
        return

    # H5 first: an unstable refit replaces the J-swap outcomes of that checkpoint and set by the lower-scoring fit.
    h5 = {}
    for lb in H5_LABELS:
        for s in SETS:
            refit = load(args.root / lb / f'h5_swap_{s}.json')
            if refit is not None and raw[lb][s] is not None:
                h5.setdefault(lb, {})[s] = h5_block(raw[lb][s], refit)
    lens = {}
    for lb in labels:
        for s in SETS:
            if raw[lb][s] is None:
                continue
            hb = h5.get(lb, {}).get(s)
            lens.setdefault(lb, {})[s] = lens_block(raw[lb][s], None if hb is None or hb['stable']
                                                    else hb['j_success_lower'])
    elig = {lb: load(args.root / lb / 'eligibility.json') for lb in OLMO_INTER}
    counts = {lb: {s: e['sets'][s]['n_correct'] for s in SETS} for lb, e in elig.items() if e is not None}
    inter_eligible = [lb for lb in OLMO_INTER if lb in counts and all(counts[lb][s] >= MIN_N for s in SETS)]
    missing_finals = [lb for lb in FINALS if lb not in lens or set(lens[lb]) != set(SETS)]

    # H1: rule on both sets at every final checkpoint and at >= 80% of intermediates eligible on both sets.
    rule_ck = {lb: all(lens[lb][s]['h1_rule'] for s in SETS) for lb in lens if set(lens[lb]) == set(SETS)}
    counted = [lb for lb in inter_eligible if lb in rule_ck]
    passing = [lb for lb in counted if rule_ck[lb]]
    h1 = dict(rule_by_checkpoint=rule_ck, intermediates_counted=counted, intermediates_passing=passing,
              finals_missing=missing_finals)
    if missing_finals or len(counts) < len(OLMO_INTER) or len(counted) < len(inter_eligible):
        h1['verdict'] = 'incomplete'
    else:
        h1['verdict'] = 'supported' if all(rule_ck[lb] for lb in FINALS) and (
            10 * len(passing) >= 8 * len(counted)) else 'not_supported'

    # H3: rule against TL-full at natural and matched norm at all three Pythia sizes, both sets.
    h3_ck = {lb: all(lens[lb][s]['h3_rule'] for s in SETS) for lb in PYTHIA if lb in rule_ck}
    h3 = dict(rule_by_size=h3_ck,
              S_J_by_size={lb: {s: lens[lb][s]['rates']['jlens'] for s in SETS} for lb in PYTHIA if lb in rule_ck},
              verdict='incomplete' if len(h3_ck) < len(PYTHIA) else
              ('supported' if all(h3_ck.values()) else 'not_supported'))

    # H2: earliest intermediate eligible on both sets; panel of prompts correct at c* and at main.
    h2 = dict(c_star=inter_eligible[0] if inter_eligible else None, by_set={})
    if not inter_eligible:
        h2['verdict'] = 'stopped_no_eligible_intermediate' if len(counts) == len(OLMO_INTER) else 'incomplete'
    elif inter_eligible[0] not in lens or OLMO_MAIN not in lens:
        h2['verdict'] = 'incomplete'
    else:
        c = inter_eligible[0]
        for s in SETS:
            jc, jm = lens[c][s]['j_success_by_id'], lens[OLMO_MAIN][s]['j_success_by_id']
            panel = [r for r in raw[OLMO_MAIN][s]['rows'] if r['id'] in jc and r['id'] in jm]
            vals = dict(main=np.array([jm[r['id']] for r in panel], dtype=bool),
                        cstar=np.array([jc[r['id']] for r in panel], dtype=bool))
            rates, con = stats([r['bridge'] for r in panel], vals, [('main', 'cstar')])
            d, n = con['main-cstar'], len(panel)
            k_c, k_m = rates['cstar']['k'], rates['main']['k']
            if n < MIN_N:
                verdict = 'undetermined_panel_below_min_n'
            elif 5 * k_c >= 4 * k_m and d['ci95'][1] < 0.15:
                verdict = 'immediate'
            elif 20 * (k_m - k_c) >= 3 * n and d['ci95'][0] > 0:
                verdict = 'late_consolidation'
            else:
                verdict = 'undetermined'
            h2['by_set'][s] = dict(panel_n=n, rates=rates, contrast=d, verdict=verdict)
        vs = {h2['by_set'][s]['verdict'] for s in SETS}
        h2['verdict'] = vs.pop() if len(vs) == 1 and vs <= {'immediate', 'late_consolidation'} else 'undetermined'
        h2['pythia14b_checkpoints_needed'] = c == 'olmo1b_step10000'

    # Trajectory over OLMo-1B training (descriptive): S_J on each checkpoint's own correct prompts and on a fixed
    # panel correct at every checkpoint eligible on that set; Spearman over log10 step of the intermediates.
    traj = {}
    for s in SETS:
        cks = [lb for lb in OLMO_INTER if lb in lens and s in lens[lb] and lens[lb][s]['n'] >= MIN_N]
        own = {lb: lens[lb][s]['rates']['jlens'] for lb in cks + ([OLMO_MAIN] if OLMO_MAIN in lens else [])}
        ids = set.intersection(*[set(lens[lb][s]['j_success_by_id']) for lb in own]) if own else set()
        rows = [r for r in raw[next(iter(own))][s]['rows'] if r['id'] in ids] if own else []
        fixed = {lb: stats([r['bridge'] for r in rows],
                           {'jlens': np.array([lens[lb][s]['j_success_by_id'][r['id']] for r in rows], dtype=bool)},
                           [])[0]['jlens'] for lb in own}
        steps = [OLMO_STEPS[OLMO_INTER.index(lb)] for lb in cks]

        def rho(series):
            if len(cks) < 3:
                return None
            r = spearmanr([math.log10(x) for x in steps], [series[lb]['rate'] for lb in cks])
            return dict(rho=float(r.statistic), p=float(r.pvalue), n_checkpoints=len(cks))
        traj[s] = dict(checkpoints=cks, own_prompts=own, fixed_panel_n=len(rows), fixed_panel=fixed,
                       spearman_own=rho(own), spearman_fixed=rho(fixed))

    # H4: privilege of the J-space for concept vectors on held-out pairs.
    c4 = {lb: {s: c4_block(d) for s in SETS if (d := load(args.root / lb / f'c4_{s}.json')) is not None}
          for lb in labels}
    c4 = {lb: v for lb, v in c4.items() if v}
    evaluable = {lb: all(c4[lb][s]['n'] >= MIN_N for s in SETS) for lb in c4 if set(c4[lb]) == set(SETS)}
    priv = {lb: all(c4[lb][s]['privilege'] for s in SETS) for lb, ev in evaluable.items() if ev}
    finals_eval = [lb for lb in FINALS if lb in priv]
    h4 = dict(evaluable=evaluable, privilege=priv, holds_at=[lb for lb, v in priv.items() if v])
    if any(lb not in evaluable for lb in FINALS):
        h4['verdict'] = 'incomplete'
    elif not finals_eval:
        h4['verdict'] = 'not_testable_no_final_with_min_n'
    elif not any(priv[lb] for lb in finals_eval):
        h4['verdict'] = 'absent_at_these_scales'
    else:
        h4['verdict'] = 'privilege_at_' + ','.join(lb for lb in finals_eval if priv[lb])

    # Holm over the primary contrasts at each final checkpoint and set: J - TL-full natural and matched norm, the
    # H2 panel contrast (OLMo-1B main), and DIM - non-J part (H4).
    holm = {}
    for lb in FINALS:
        for s in SETS:
            fam = []
            if lb in lens and s in lens[lb]:
                fam += [(f'H1_{x}', lens[lb][s]['contrasts'][x]['mcnemar_p']) for x in ('jlens-tuned', 'jlens-tuned_nm')]
            if lb == OLMO_MAIN and s in h2['by_set']:
                fam.append(('H2_main-cstar', h2['by_set'][s]['contrast']['mcnemar_p']))
            if lb in c4 and s in c4[lb]:
                fam.append(('H4_dim-dim_outside_j', c4[lb][s]['contrasts']['dim-dim_outside_j']['mcnemar_p']))
            if fam:
                adj = multipletests([p for _, p in fam], method='holm')[1]
                holm[f'{lb}/{s}'] = {name: dict(p=p, holm_p=float(a)) for (name, p), a in zip(fam, adj)}

    stops = stop_check(lens) if all(lb in lens and 'v2' in lens[lb] for lb in (OLMO_MAIN, 'pythia28b')) else None
    stops_training = dict(no_intermediate_eligible=len(counts) == len(OLMO_INTER) and not inter_eligible)
    for lb in lens:
        for s in lens[lb]:
            del lens[lb][s]['j_success_by_id']
    for lb in h5:
        for s in h5[lb]:
            del h5[lb][s]['j_success_lower']
    out = dict(H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, stop_rules=stops, stop_rules_training=stops_training,
               olmo_trajectory=traj, eligibility_counts=counts, lens=lens, c4=c4, holm=holm, n_boot=N_BOOT,
               seed=SEED, min_n=MIN_N)
    write_json(args.out, out)
    print(json.dumps(dict(H1=h1['verdict'], H2=h2['verdict'], H3=h3['verdict'], H4=h4['verdict'],
                          H5={lb: {s: v['stable'] for s, v in x.items()} for lb, x in h5.items()}, stops=stops,
                          stops_training=stops_training)))


if __name__ == '__main__':
    main()
