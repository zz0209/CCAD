from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010')
MIRROR = BULK / 'mayo_mirror'
RG08 = Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round08')
DESCRIPTION = ('Complete-request and part-request errors of every executor across the infinitive, public-dictionary, '
               'GPT-2 and Qwen settings, with paired bootstrap intervals over targets and held-out inputs.')
SETTINGS = [
    dict(key='infinitive', kind='infinitive', runs=[BULK / f'NAACL_E2_infinitive_t{t}_v1_20261010' for t in (1, 3, 4, 5)],
         methods=dict(complete='tangent_full', part='tangent_mixed', readout='raw_reconstruction',
                      hungarian_decoder='hungarian_decoder', hungarian_activation='hungarian_activation', natural='native')),
    dict(key='infinitive_public', kind='infinitive', runs=[MIRROR / f'NAACL_E4A_public_t{t}_v1_20261010' for t in range(101, 117)],
         methods=dict(complete='tangent_full', part='tangent_mixed', readout='raw_reconstruction',
                      hungarian_decoder='hungarian_decoder', hungarian_activation='hungarian_activation',
                      gain='tangent_gain')),
    dict(key='gpt2', kind='grammar', runs=[RG08 / f'RG08_GPT2_PROGRAM_CONFIRM_T{t}_20260921' for t in (2, 3, 4, 5)],
         extra_runs=[BULK / f'NAACL_E1_GPT2_T{t}_v1_20261010' for t in (2, 3, 4, 5)],
         methods=dict(complete='whole', part='program', readout='readout_initial', untrained='initial',
                      hungarian_decoder='hungarian_decoder', hungarian_activation='hungarian_activation', gain='gain')),
    dict(key='qwen', kind='grammar', runs=[MIRROR / f'NAACL_E4B_QWEN17_S{s}_T{t}_v4_20261010' for s, t in ((1, 2), (2, 1))],
         methods=dict(complete='whole', part='program', readout='readout_initial', untrained='initial',
                      hungarian_decoder='hungarian_decoder', hungarian_activation='hungarian_activation', gain='gain')),
    # Fixed memberships run with the target SAE alone; membership_initial executes the Hungarian decoder matches.
    dict(key='gpt2_fixed', kind='grammar', runs=[BULK / f'NAACL_E9_GPT2_FIXED_T{t}_v1_20261010' for t in (2, 3, 4, 5)],
         methods=dict(complete='membership_whole', part='membership', readout='readout_initial',
                      hungarian_decoder='membership_initial')),
    dict(key='qwen_fixed', kind='grammar',
         runs=[MIRROR / f'NAACL_E9_QWEN17_FIXED_S{s}_T{t}_v1_20261010' for s, t in ((1, 2), (2, 1))],
         methods=dict(complete='membership_whole', part='membership', readout='readout_initial',
                      hungarian_decoder='membership_initial')),
]
CONTRASTS = [('part', 'complete'), ('part', 'readout'), ('part', 'hungarian_decoder')]


def load(run, kind, extra=None):
    assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
    values = dict(np.load(run / 'responses.npz'))
    if extra is not None:
        main_panel, extra_panel = (json.loads((r / 'panel.json').read_text()) for r in (run, extra))
        assert main_panel['rows'] == extra_panel['rows'], extra
        order = [extra_panel['query_order'].index(q) for q in main_panel['query_order']]
        more = np.load(extra / 'responses.npz')
        for k in more.files:
            if k in ('none', 'source'):
                assert np.allclose(more[k][order], values[k], atol=1e-4), (extra, k)
            elif k not in values:
                values[k] = more[k][order]
    if kind == 'infinitive':
        queries = json.loads((run / 'INDEX.json').read_text())['queries']
        fam = dict(complete=['full'], parts=['predicate', 'object'],
                   new=[q for q in queries if q.startswith(('interior_', 'boundary_', 'member_subset_'))])
        groups = np.zeros(values['none'].shape[1], dtype=int)
    else:
        panel = json.loads((run / 'panel.json').read_text())
        queries = list(panel['query_order'])
        fam = dict(complete=['full'], parts=['verb', 'number', 'gender'],
                   new=['verb_number', 'verb_gender', 'number_gender', 'participation_a', 'participation_b',
                        'participation_c', 'half'])
        tasks = sorted({r['task'] for r in panel['rows']})
        groups = np.array([tasks.index(r['task']) for r in panel['rows']])
    qi = {q: i for i, q in enumerate(queries)}
    return values, {f: [qi[q] for q in qs] for f, qs in fam.items()}, groups


def per_item(values, fam, method):
    src, none = values['source'], values['none']
    err = {f: ((values[method][ix] - src[ix]) ** 2).sum(0) for f, ix in fam.items()}
    eff = {f: ((src[ix] - none[ix]) ** 2).sum(0) for f, ix in fam.items()}
    return err, eff


def nrmse(err, eff, groups, weights):
    # Pooled within each group of held-out inputs, then averaged over groups.
    out = []
    for g in np.unique(groups):
        sel = groups == g
        out.append(np.sqrt((weights[..., sel] * err[sel]).sum(-1) / (weights[..., sel] * eff[sel]).sum(-1)))
    return np.mean(out, axis=0)


def analyse(setting, draws, rng):
    loaded = []
    extras = setting.get('extra_runs', [None] * len(setting['runs']))
    for run, extra in zip(setting['runs'], extras):
        loaded.append(load(run, setting['kind'], extra))
    groups = loaded[0][2]
    for _, _, g in loaded:
        assert np.array_equal(g, groups)
    n_items, n_runs = len(groups), len(loaded)
    item_w = np.zeros((draws, n_items))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        picks = rng.choice(idx, size=(draws, len(idx)), replace=True)
        for d in range(draws):
            np.add.at(item_w[d], picks[d], 1.)
    run_w = np.stack([np.bincount(rng.integers(0, n_runs, n_runs), minlength=n_runs) for _ in range(draws)]) / n_runs
    point, boot = {}, {}
    for name, method in setting['methods'].items():
        point[name], boot[name] = {}, {}
        per_run_point, per_run_boot = [], []
        for values, fam, _ in loaded:
            err, eff = per_item(values, fam, method)
            per_run_point.append({f: float(nrmse(err[f], eff[f], groups, np.ones(n_items))) for f in fam})
            per_run_boot.append({f: nrmse(err[f], eff[f], groups, item_w) for f in fam})
        for f in per_run_point[0]:
            point[name][f] = float(np.mean([r[f] for r in per_run_point]))
            stack = np.stack([r[f] for r in per_run_boot], axis=1)
            boot[name][f] = (stack * run_w).sum(1)
        point[name]['per_run'] = per_run_point
    contrasts = {}
    for a, b in CONTRASTS:
        if a in boot and b in boot:
            contrasts[f'{a}-{b}'] = {f: dict(point=point[a][f] - point[b][f],
                                             interval=np.quantile(boot[a][f] - boot[b][f], [.025, .975]).tolist())
                                     for f in boot[a]}
    intervals = {m: {f: np.quantile(v, [.025, .975]).tolist() for f, v in fams.items()} for m, fams in boot.items()}
    return dict(runs=[str(r) for r in setting['runs']], n_items=n_items, n_runs=n_runs, methods=setting['methods'],
                point=point, intervals=intervals, contrasts=contrasts, coaction=coaction(loaded))


def coaction(loaded):
    # Share of part effect energy outside the dominant part of each input; zero when one part acts per input.
    values, fam, _ = loaded[0]
    eff = (values['source'][fam['parts']] - values['none'][fam['parts']]) ** 2
    total = eff.sum(0)
    keep = total > 0
    return float(np.mean(1 - eff.max(0)[keep] / total[keep]))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/naacl_sprint_20261010/e2_supervision/SUPERVISION.json')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), draws=args.draws, seed=args.seed,
               inference='Paired bootstrap over target dictionaries and held-out inputs within each task.', settings={})
    for setting in SETTINGS:
        out['settings'][setting['key']] = res = analyse(setting, args.draws, rng)
        print(setting['key'])
        for m in setting['methods']:
            p = res['point'][m]
            print(f"  {m:<22} complete {p['complete']:.3f}  parts {p['parts']:.3f}  new {p['new']:.3f}")
        for k, c in res['contrasts'].items():
            print(f"  {k:<26}" + '  '.join(f"{f} {v['point']:+.3f} [{v['interval'][0]:+.3f}, {v['interval'][1]:+.3f}]"
                                            for f, v in c.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
