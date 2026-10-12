import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Does agreement between two dictionaries predict which writes of a CLT unit the model confirms? '
               'Agreement of a CLT unit is how well the PLT features active at the same token reconstruct its write '
               'into its own layer (one minus the residual of the converse rows) and into its later layers '
               '(energy-weighted). Confirmation is the fraction of its later writes that the model removes when the '
               'unit\'s read is removed (u) and the excess over random PLT features when the PLT units of the prompt '
               'at or before its position are removed (a minus ra, stored as s and r). Spearman correlations over '
               'units, overall and within read-layer bands, with '
               'graph-bootstrap intervals, and confirmation of units the PLT shares (own residual below .4) against '
               'the others.')


def ranks(x):
    return np.argsort(np.argsort(x))


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def boot_spearman(graphs, x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(set(graphs))
    idx = {g: np.flatnonzero(np.array(graphs) == g) for g in keys}
    stats = []
    for _ in range(n):
        pick = np.concatenate([idx[keys[i]] for i in rng.choice(len(keys), len(keys))])
        if len(set(x[pick])) > 2:
            stats.append(spearman(x[pick], y[pick]))
    return spearman(x, y), float(np.quantile(stats, .025)), float(np.quantile(stats, .975))


def partial_spearman(x, y, covs):
    # Spearman correlation of x and y after removing a linear fit on the ranks of the covariates from both ranks.
    Z = np.column_stack([np.ones(len(x))] + [ranks(c) for c in covs])
    res = lambda v: ranks(v) - Z @ np.linalg.lstsq(Z, ranks(v), rcond=None)[0]
    return float(np.corrcoef(res(x), res(y))[0, 1])


def boot_stat(graphs, stat, n=2000, seed=0):
    # Statistic of the units with an interval that resamples graphs.
    rng = np.random.default_rng(seed)
    keys = sorted(set(graphs))
    idx = {g: np.flatnonzero(np.array(graphs) == g) for g in keys}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[keys[i]] for i in rng.choice(len(keys), len(keys))])
        v = stat(pick)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return stat(np.arange(len(graphs))), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def later(r, key):
    c = np.array(r['claim_energy'][1:])
    return float(np.sum(r[key][1:]) / c.sum())


def join(depth_path, reverse_path):
    rev = {(r['graph'], r['unit']): r for r in json.loads(reverse_path.read_text())['rows'] if r['features'] == 1}
    out = []
    for r in json.loads(depth_path.read_text())['rows']:
        v = rev.get((r['graph'], r['unit']))
        if v is None or len(r['claim_energy']) < 2 or sum(r['claim_energy'][1:]) <= 0:
            continue
        own = [res for k, res, _ in v['residual_by_offset'] if k == 0]
        lat = [(res, nrm) for k, res, nrm in v['residual_by_offset'] if k > 0]
        if not own or not lat:
            continue
        w = np.array([nrm ** 2 for _, nrm in lat])
        # PLT units of the prompt at or before the unit's position (a) and matched random features (ra).
        has_s = 'removed_a' in r and 'removed_ra' in r
        out.append(dict(graph=r['graph'], layer=r['layer'], log_norm=float(np.log(np.sum(r['claim_energy']))),
                        agree_own=1 - own[0],
                        agree_later=1 - float(np.sum(w * np.array([res for res, _ in lat])) / w.sum()),
                        own_u=r['removed_u'][0] / r['claim_energy'][0], confirm_u=later(r, 'removed_u'),
                        later_s=later(r, 'removed_a') if has_s else None, later_r=later(r, 'removed_ra') if has_s else None,
                        confirm_s=later(r, 'removed_a') - later(r, 'removed_ra') if has_s else None))
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--pairs', nargs='+', required=True, help='name=DEPTH_REFEREE_ROWS.json,REVERSE_ROWS.json')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    summary = {}
    for pair in args.pairs:
        name, paths = pair.split('=')
        depth, reverse = (Path(p) for p in paths.split(','))
        units = join(depth, reverse)
        res = {}
        for band, lo, hi in [('all', 0, 99), ('layers 0-1', 0, 1), ('layers 2+', 2, 99)]:
            sel = [u for u in units if lo <= u['layer'] <= hi]
            d = dict(units=len(sel))
            for a in ('agree_own', 'agree_later'):
                for c in ('confirm_u', 'confirm_s'):
                    s = [u for u in sel if u[c] is not None]
                    if len(s) > 10:
                        d[f'{a}~{c}'] = boot_spearman([u['graph'] for u in s], np.array([u[a] for u in s]),
                                                      np.array([u[c] for u in s]))
            for c in ('confirm_u', 'confirm_s'):
                s = [u for u in sel if u[c] is not None]
                shared = [u[c] for u in s if u['agree_own'] > .6]
                other = [u[c] for u in s if u['agree_own'] <= .6]
                if shared and other:
                    d[f'{c} shared/other'] = (float(np.median(shared)), float(np.median(other)), len(shared), len(other))
            res[band] = d
        # Medians for the table: units read at layers 0-1, and later units with and without a PLT counterpart.
        groups = {}
        for gname, keep in [('early', lambda u: u['layer'] <= 1),
                            ('later_shared', lambda u: u['layer'] >= 2 and u['agree_own'] > .6),
                            ('later_other', lambda u: u['layer'] >= 2 and u['agree_own'] <= .6)]:
            sel = [u for u in units if keep(u)]
            withs = [u for u in sel if u['later_s'] is not None]
            groups[gname] = dict(units=len(sel), with_plt=len(withs),
                                 **{k: float(np.median([u[k] for u in sel])) for k in ('own_u', 'confirm_u')},
                                 **{k: float(np.median([u[k] for u in withs])) for k in ('later_s', 'later_r')})
        res['groups'] = groups
        # Robustness for units read after layer 1: the shared-minus-other difference of medians, and the Spearman
        # correlation of agreement with confirmation after removing read layer and write norm.
        sel = [u for u in units if u['layer'] >= 2]
        g = [u['graph'] for u in sel]
        a = np.array([u['agree_own'] for u in sel])
        lay = np.array([u['layer'] for u in sel], dtype=float)
        nrm = np.array([u['log_norm'] for u in sel])
        rob = {}
        for c in ('confirm_u', 'confirm_s'):
            keep = np.array([u[c] is not None for u in sel])
            y = np.array([u[c] if u[c] is not None else np.nan for u in sel])
            def diff(pick, y=y, keep=keep):
                pick = pick[keep[pick]]
                sh, ot = y[pick][a[pick] > .6], y[pick][a[pick] <= .6]
                return float(np.median(sh) - np.median(ot)) if len(sh) and len(ot) else None
            def part(pick, y=y, keep=keep):
                pick = pick[keep[pick]]
                return partial_spearman(a[pick], y[pick], [lay[pick], nrm[pick]]) if len(pick) > 10 else None
            rob[c] = dict(shared_minus_other=boot_stat(g, diff), partial_spearman_layer_norm=boot_stat(g, part))
        rob['spearman_agree_layer'] = spearman(a, lay)
        rob['spearman_agree_norm'] = spearman(a, nrm)
        res['robust_later'] = rob
        res['points'] = units
        summary[name] = res
        print(name)
        for band, d in res.items():
            if band == 'robust_later':
                print('  robust_later', json.dumps(d))
                continue
            if band in ('points', 'groups'):
                if band == 'groups':
                    for gname, v in d.items():
                        print(f'  {gname:13s} ' + ' '.join(f'{k} {x:.3f}' if isinstance(x, float) else f'{k} {x}' for k, x in v.items()))
                continue
            print(f'  {band:11s} ' + ' '.join(f'{k} ' + ('[' + ', '.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in v) + ']'
                                                       if isinstance(v, tuple) else str(v)) for k, v in d.items()))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
