import argparse
import json
from pathlib import Path

import numpy as np

from summarize_depth_referee import bounded, later

DESCRIPTION = ('Per-unit numbers of the unit test for one graph, for the worked example of the paper: read layer and '
               'position, activation, the share of the claimed write in later layers, the own-layer fraction removed '
               'by removing the input, the later-layer fraction under each removal, the bounded share, and the PLT-unit '
               'and random-feature removals.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    out = []
    for r in json.loads(args.rows.read_text())['rows']:
        if r['graph'] != args.graph or len(r['claim_energy']) < 2:
            continue
        c = np.array(r['claim_energy'])
        d = dict(unit=r['unit'], layer=r['layer'], position=r['position'], activation=r['activation'],
                 claim_norm=float(np.sqrt(c.sum())), later_share=float(c[1:].sum() / c.sum()),
                 own_u=r['removed_u'][0] / c[0], bounded_u=bounded(r, 'removed_u'),
                 effect_clamped=r['energy_w_clamped'] ** .5)
        for k in ('u', 'u_all', 'u_zero', 'u_mean', 'a', 'ra'):
            if f'removed_{k}' in r:
                d[f'later_{k}'] = later(r, f'removed_{k}')
        out.append(d)
    out.sort(key=lambda d: (d['layer'], d['position']))
    for d in out:
        print(' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in d.items()))
    if args.out is not None:
        args.out.write_text(json.dumps(dict(graph=args.graph, units=out), indent=1))


if __name__ == '__main__':
    main()
