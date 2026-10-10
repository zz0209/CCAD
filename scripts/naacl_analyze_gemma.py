from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = ('Part, complete and new-request errors of every executor in the Gemma SHIFT transfer runs, with '
               'bootstrap intervals over balanced biographies, and the SHIFT use accuracies.')
BASE = ['readout', 'hungarian_decoder', 'hungarian_activation', 'top2_decoder', 'native']


def groups_of(methods):
    learned = {}
    for m in methods:
        if '_sup_' in m:
            learned.setdefault(m.rsplit('_s', 1)[0], []).append(m)
    return {**{m: [m] for m in BASE if m in methods}, **learned}


def nrmse(resp, src, ix, w):
    err = ((resp[ix] - src[ix]) ** 2).sum(0)
    eff = (src[ix] ** 2).sum(0)
    return np.sqrt((w * err).sum(-1) / (w * eff).sum(-1))


def analyse(run, draws, rng):
    req = json.loads((run / 'requests.json').read_text())
    data = np.load(run / 'responses.npz')
    names = list(data['request_names'])
    fam = {f: [names.index(r) for r in rs] for f, rs in req['families'].items()}
    strata = data['eval_true'] * 2 + data['eval_spur']
    n = len(strata)
    w = np.zeros((draws, n))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        picks = rng.choice(idx, size=(draws, len(idx)))
        for d in range(draws):
            np.add.at(w[d], picks[d], 1.)
    src = data['source']
    methods = [k for k in data.files if k not in ('request_names', 'eval_true', 'eval_spur', 'eval_base', 'source')]
    groups = groups_of(methods)
    point, boot = {}, {}
    # Training seeds are resampled together with biographies, so intervals include seed variation.
    seed_draws = {len(m): rng.integers(0, len(m), (len(w), len(m))) for m in groups.values()}
    for g, members in groups.items():
        point[g], boot[g] = {}, {}
        picks = seed_draws[len(members)]
        for f, ix in fam.items():
            point[g][f] = float(np.mean([nrmse(data[m], src, ix, np.ones(n)) for m in members]))
            per_member = np.stack([nrmse(data[m], src, ix, w) for m in members])
            boot[g][f] = per_member[picks, np.arange(len(w))[:, None]].mean(1)
            point[g][f + '_effect_ratio'] = float(np.mean([np.sqrt((data[m][ix] ** 2).sum() / (src[ix] ** 2).sum())
                                                           for m in members]))
        point[g]['seeds'] = len(members)
    contrasts = {}
    parts = [g for g in groups if g.startswith(('part_sup_', 'subset_sup_', 'random_sup_', 'singleton_sup_'))]
    for p in parts:
        init = p.split('_sup_')[1]
        for other in [f'complete_sup_{init}', f'part_sup_{init}', f'random_sup_{init}', f'singleton_sup_{init}',
                      'readout', 'hungarian_decoder', 'native']:
            if other == p or f'{other}-{p}' in contrasts:
                continue
            if other in boot:
                contrasts[f'{p}-{other}'] = {f: dict(point=point[p][f] - point[other][f],
                                                     interval=np.quantile(boot[p][f] - boot[other][f], [.025, .975]).tolist())
                                             for f in fam}
    intervals = {g: {f: np.quantile(v, [.025, .975]).tolist() for f, v in fs.items()} for g, fs in boot.items()}
    out = dict(run=str(run), n_bios=n, families=req['families'], point=point, intervals=intervals, contrasts=contrasts)
    base = data['eval_base']
    per_part, agreement = {}, {}
    for g, members in groups.items():
        per_part[g] = {r: float(np.mean([nrmse(data[m], src, [names.index(r)], np.ones(n)) for m in members]))
                       for r in req['families']['parts']}
        agreement[g] = {}
        for f in ['complete', 'parts']:
            vals = []
            for m in members:
                for r in req['families'][f]:
                    i = names.index(r)
                    ref, mine = (base + src[i]) > 0, (base + data[m][i]) > 0
                    changed = ref != (base > 0)
                    agree = mine == ref
                    vals.append(.5 * (agree[changed].mean() + agree[~changed].mean()))
            agreement[g][f] = float(np.mean(vals))
    out.update(per_part=per_part, agreement=agreement)
    maps_path = run / 'maps.json'
    if maps_path.exists():
        maps = json.loads(maps_path.read_text())
        structure = {}
        for name, sites in maps.items():
            per_source, shared, used = [], 0, 0
            for s in sites.values():
                w = np.array(s['weights'])
                active = w > .05
                per_source += active.sum(1).tolist()
                used += int(active.any(0).sum())
                shared += int((active.sum(0) >= 2).sum())
            structure[name] = dict(targets_per_source=float(np.mean(per_source)), max_per_source=int(max(per_source)),
                                   target_features=used, shared_by_two_or_more=shared)
        out['structure'] = structure
    stats_path = run / 'natural_statistics.npz'
    if stats_path.exists():
        z = np.load(stats_path)
        if any(k.endswith('__source_retention') for k in z.files):
            sites = sorted({k.split('__')[0] for k in z.files})
            out['retention'] = {s: z[s + '__source_retention'].tolist() for s in sites}
            out['target_fve'] = {s: float(z[s + '__target_fve']) for s in sites}
            out['hungarian_decoder_cos'] = {s: z[s + '__hungarian_decoder_cos'].tolist() for s in sites}
    use_path = run / 'USE.json'
    if use_path.exists():
        use = json.loads(use_path.read_text())
        summary = {}
        for g, members in {'clean': ['clean'], 'oracle': ['oracle'], 'source': ['source'], **groups}.items():
            rows = [use[m] for m in members if m in use]
            if rows:
                summary[g] = {c: {k: float(np.mean([r[c][k] for r in rows])) for k in rows[0][c]} for c in rows[0]}
        out['use'] = summary
    results = run / 'RESULTS.json'
    if results.exists():
        out['target_features'] = json.loads(results.read_text())['target_features']
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), draws=args.draws,
                  inference='Bootstrap over balanced biographies within profession-by-gender strata, shared across methods.',
                  runs={})
    for run in args.runs:
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        res = record['runs'][run.name] = analyse(run, args.draws, rng)
        print(run.name, res['n_bios'])
        for g, p in res['point'].items():
            print(f"  {g:<26} complete {p['complete']:.3f}  parts {p['parts']:.3f}  new {p['new']:.3f}  "
                  f"ratio parts {p['parts_effect_ratio']:.2f}")
        for k, c in res['contrasts'].items():
            print(f"  {k:<44}" + '  '.join(f"{f} {v['point']:+.3f} [{v['interval'][0]:+.3f}, {v['interval'][1]:+.3f}]"
                                            for f, v in c.items()))
        for g, u in res.get('use', {}).items():
            print(f"  use {g:<22}" + '  '.join(f"{c}: prof {v['profession']:.3f} gender {v['gender']:.3f} wg {v['worst_group']:.3f}"
                                              for c, v in u.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
