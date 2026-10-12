import argparse
import json
from pathlib import Path

import numpy as np

from agreement_vs_referee import boot_stat

DESCRIPTION = ('Positive control for the edge test at the first layers. Edges leaving units read at layers 0 and 1 are '
               'split by the read layer of the downstream unit (up to layer 3, 4 to 12, 13 on). The removal of the '
               'upstream unit\'s input is the same edit for every downstream unit, so a high confirmed share for nearby '
               'downstream units shows that the removal does take out what the unit carries. Medians of the confirmed '
               'share with graph-bootstrap intervals, and the share of the claimed change in each band.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True, help='name=EDGE_REFEREE_ROWS.json')
    parser.add_argument('--min-share', type=float, default=.1)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    summary = {}
    for spec in args.runs:
        name, path = spec.split('=')
        rows = [r for r in json.loads(Path(path).read_text())['rows']
                if r['u_layer'] <= 1 and abs(r['claimed']) >= args.min_share * abs(r['v_pre']) and abs(r['claimed']) > 0]
        total = sum(abs(r['claimed']) for r in rows)
        d = dict(edges=len(rows))
        for band, lo, hi in [('to layers 2-3', 0, 3), ('to layers 4-12', 4, 12), ('to layers 13+', 13, 99)]:
            sel = [r for r in rows if lo <= r['v_layer'] <= hi]
            if len(sel) < 5:
                d[band] = dict(edges=len(sel))
                continue
            vals = np.array([r['model'] / r['claimed'] for r in sel])
            g = [r['graph'] for r in sel]
            d[band] = dict(edges=len(sel), graphs=len(set(g)), median=boot_stat(g, lambda pick: float(np.median(vals[pick]))),
                           share_of_claimed=sum(abs(r['claimed']) for r in sel) / total)
        summary[name] = d
        print(name, json.dumps(d))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
