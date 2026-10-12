import argparse
import json
from pathlib import Path

import numpy as np

from agreement_vs_referee import boot_spearman, boot_stat, partial_spearman, spearman

DESCRIPTION = ('The PLT side of the unit test. For every unit of the PLT\'s own graphs, the fraction of its own-layer '
               'write that the model removes when the unit\'s input is removed (plt_rows of the depth referee) against '
               'its agreement with the CLT (one minus the site residual of its write in the CLT features active at its '
               'site, oracle rows on the PLT\'s own-graph registry). Medians for units the CLT shares (agreement above '
               '.6) and the others by read layer, with graph-bootstrap intervals for the difference, and Spearman '
               'correlations with and without read layer and write norm.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--pairs', nargs='+', required=True, help='name=DEPTH_REFEREE_ROWS.json,ORACLE_ROWS.json')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    summary = {}
    for pair in args.pairs:
        name, paths = pair.split('=')
        depth, oracle = (Path(p) for p in paths.split(','))
        res_by = {(r['graph'], r['supernode']): r['site_residual'] for r in json.loads(oracle.read_text())['rows']
                  if r.get('site_residual') is not None}
        units = []
        for r in json.loads(depth.read_text())['plt_rows']:
            key = (r['graph'], r['unit'])
            if key not in res_by or r['claim_energy'] <= 0:
                continue
            units.append(dict(graph=r['graph'], layer=r['layer'], agree=1 - res_by[key],
                              confirm=r['removed'] / r['claim_energy'], log_norm=float(np.log(r['claim_energy']))))
        out = dict(units=len(units))
        for band, lo, hi in [('all', 0, 99), ('layers 0-4', 0, 4), ('layers 5+', 5, 99)]:
            sel = [u for u in units if lo <= u['layer'] <= hi]
            if len(sel) < 10:
                continue
            g = [u['graph'] for u in sel]
            a = np.array([u['agree'] for u in sel])
            c = np.array([u['confirm'] for u in sel])
            lay = np.array([u['layer'] for u in sel], dtype=float)
            nrm = np.array([u['log_norm'] for u in sel])
            shared, other = c[a > .6], c[a <= .6]
            d = dict(units=len(sel), shared=int(len(shared)), other=int(len(other)),
                     confirm_shared=float(np.median(shared)) if len(shared) else None,
                     confirm_other=float(np.median(other)) if len(other) else None,
                     spearman=boot_spearman(g, a, c))
            if len(shared) and len(other):
                def diff(pick):
                    s, o = c[pick][a[pick] > .6], c[pick][a[pick] <= .6]
                    return float(np.median(s) - np.median(o)) if len(s) and len(o) else None
                d['shared_minus_other'] = boot_stat(g, diff)
            d['partial_spearman_layer_norm'] = boot_stat(g, lambda pick: partial_spearman(a[pick], c[pick], [lay[pick], nrm[pick]])
                                                         if len(pick) > 10 else None)
            out[band] = d
        summary[name] = out
        print(name, json.dumps(out))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
