import argparse
import json
from pathlib import Path

import numpy as np

from agreement_vs_referee import boot_stat

DESCRIPTION = ('Summary of the edge test. Edges whose claimed change of the downstream pre-activation is at least '
               '--min-share of that pre-activation enter. The confirmed share of an edge is the model\'s change over '
               'the claimed change. Medians with graph-bootstrap intervals for edges leaving units read at layers 0 '
               'and 1 and for edges whose ends are both read later, and, given the CLT\'s converse rows, for later '
               'edges leaving units that the PLT shares (own-layer residual below .4) or not, with their difference.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True, help='name=EDGE_REFEREE_ROWS.json[,REVERSE_ROWS.json]')
    parser.add_argument('--min-share', type=float, default=.1)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    summary = {}
    for spec in args.runs:
        name, paths = spec.split('=')
        paths = paths.split(',')
        rows = [r for r in json.loads(Path(paths[0]).read_text())['rows']
                if abs(r['claimed']) >= args.min_share * abs(r['v_pre']) and abs(r['claimed']) > 0]
        shared = None
        if len(paths) > 1:
            shared = {}
            for r in json.loads(Path(paths[1]).read_text())['rows']:
                own = [res for k, res, _ in r['residual_by_offset'] if k == 0] if r['features'] == 1 else []
                if own:
                    shared[(r['graph'], r['unit'])] = own[0] < .4
        ratio = lambda r: r['model'] / r['claimed']
        groups = {'from layers 0-1': [r for r in rows if r['u_layer'] <= 1],
                  'from layers 2+': [r for r in rows if r['u_layer'] >= 2]}
        if shared is not None:
            groups['from 2+, shared'] = [r for r in groups['from layers 2+'] if shared.get((r['graph'], r['u']))]
            groups['from 2+, other'] = [r for r in groups['from layers 2+'] if shared.get((r['graph'], r['u'])) is False]
        d = dict(edges=len(rows))
        for gname, sel in groups.items():
            if len(sel) < 5:
                continue
            g = [r['graph'] for r in sel]
            vals = np.array([ratio(r) for r in sel])
            d[gname] = dict(edges=len(sel), graphs=len(set(g)),
                            median=boot_stat(g, lambda pick: float(np.median(vals[pick]))),
                            share_above_half=float(np.mean(vals > .5)))
        if shared is not None and 'from 2+, shared' in d and 'from 2+, other' in d:
            sel = groups['from layers 2+']
            sel = [r for r in sel if (r['graph'], r['u']) in shared]
            g = [r['graph'] for r in sel]
            vals = np.array([ratio(r) for r in sel])
            sh = np.array([shared[(r['graph'], r['u'])] for r in sel])
            def diff(pick):
                a, b = vals[pick][sh[pick]], vals[pick][~sh[pick]]
                return float(np.median(a) - np.median(b)) if len(a) and len(b) else None
            d['shared_minus_other'] = boot_stat(g, diff)
        summary[name] = d
        print(name, json.dumps(d))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
