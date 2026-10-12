import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Summary of depth referee rows by the read layer of each CLT unit (0-1, 2-12, 13 on): the fraction of '
               'the unit\'s own-layer write that removing its read takes out of the MLP output of that layer, and the '
               'fraction of its later writes taken out of the later MLP outputs by removing its read (u), the inputs '
               'of the PLT units of the same prompt and token after its layer (s), or as many random active PLT '
               'features at the same layers (r); medians over units with graph-bootstrap intervals for s minus r and '
               's minus u, and the logit effect of deleting the claimed writes with the later MLP outputs clamped or '
               'free against removing the read. PLT units give the own-layer fraction on the PLT side.')
BANDS = [('layers 0-1', 0, 1), ('layers 2-12', 2, 12), ('layers 13+', 13, 99)]


def boot(rows, value, n=2000, seed=0):
    # Median of a per-unit value with an interval that resamples graphs.
    rng = np.random.default_rng(seed)
    graphs = sorted({r['graph'] for r in rows})
    by = {g: [value(r) for r in rows if r['graph'] == g] for g in graphs}
    med = float(np.median([v for g in graphs for v in by[g]]))
    stats = []
    for _ in range(n):
        pick = rng.choice(len(graphs), len(graphs))
        vals = [v for i in pick for v in by[graphs[i]]]
        stats.append(np.median(vals))
    return med, float(np.quantile(stats, .025)), float(np.quantile(stats, .975))


def later(r, key):
    c = np.array(r['claim_energy'][1:])
    return float(np.sum(r[key][1:]) / c.sum())


def summarize(rows, plt_rows):
    out = {}
    for name, lo, hi in BANDS:
        sel = [r for r in rows if lo <= r['layer'] <= hi and len(r['claim_energy']) > 1 and sum(r['claim_energy'][1:]) > 0]
        if not sel:
            continue
        d = dict(units=len(sel), graphs=len({r['graph'] for r in sel}))
        d['own_layer_u'] = boot(sel, lambda r: r['removed_u'][0] / r['claim_energy'][0])
        d['later_u'] = boot(sel, lambda r: later(r, 'removed_u'))
        d['later_share_of_claim'] = float(np.median([sum(r['claim_energy'][1:]) / sum(r['claim_energy']) for r in sel]))
        both = [r for r in sel if 'removed_s' in r and 'removed_r' in r]
        d['with_plt_units'] = len(both)
        if both:
            d['later_s'] = boot(both, lambda r: later(r, 'removed_s'))
            d['later_r'] = boot(both, lambda r: later(r, 'removed_r'))
            d['later_s_minus_r'] = boot(both, lambda r: later(r, 'removed_s') - later(r, 'removed_r'))
            d['later_s_minus_u'] = boot(both, lambda r: later(r, 'removed_s') - later(r, 'removed_u'))
        wide = [r for r in sel if 'removed_a' in r and 'removed_ra' in r]
        d['with_plt_units_any_position'] = len(wide)
        if wide:
            d['later_a'] = boot(wide, lambda r: later(r, 'removed_a'))
            d['later_ra'] = boot(wide, lambda r: later(r, 'removed_ra'))
            d['later_a_minus_ra'] = boot(wide, lambda r: later(r, 'removed_a') - later(r, 'removed_ra'))
        own = np.array([r['removed_u'][0] / r['claim_energy'][0] for r in sel])
        lat = np.array([later(r, 'removed_u') for r in sel])
        d['own_over_half_later_below_fifth'] = int(np.sum((own >= .5) & (lat <= .2)))
        d['own_over_half'] = int(np.sum(own >= .5))
        strong = [r for r in sel if r['energy_w_free'] ** .5 > .3]
        d['strong'] = len(strong)
        if strong:
            d['write_free_over_clamped'] = float(np.median([(r['energy_w_free'] / max(r['energy_w_clamped'], 1e-12)) ** .5 for r in strong]))
            d['write_clamped_over_read'] = float(np.median([(r['energy_w_clamped'] / max(r['energy_u'], 1e-12)) ** .5 for r in strong]))
            d['write_free_over_read'] = float(np.median([(r['energy_w_free'] / max(r['energy_u'], 1e-12)) ** .5 for r in strong]))
        d['check_off_max'] = float(max(r['check_off'] for r in sel))
        out[name] = d
    for name, lo, hi in [('PLT layers 0-4', 0, 4), ('PLT layers 5-12', 5, 12), ('PLT layers 13+', 13, 99)]:
        sel = [r for r in plt_rows if lo <= r['layer'] <= hi and r['claim_energy'] > 0]
        if sel:
            out[name] = dict(units=len(sel), own_layer=boot(sel, lambda r: r['removed'] / r['claim_energy']))
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    summary = {}
    for path in args.rows:
        d = json.loads(path.read_text())
        s = summarize(d['rows'], d['plt_rows'])
        summary[path.parent.name] = s
        print(path.parent.name)
        for band, v in s.items():
            print(f'  {band:16s} ' + ' '.join(f'{k} ' + (f'{x:.3f}' if isinstance(x, float) else
                                                          ('[' + ', '.join(f'{y:.3f}' for y in x) + ']' if isinstance(x, tuple) else str(x)))
                                              for k, x in v.items()))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
