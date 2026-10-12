import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('How much of a CLT graph the PLT shares, by depth: for the single-feature units of the CLT\'s own graphs '
               '(converse rows), the share read at layers 0 and 1, and by read layer the share of units whose write into '
               'their own layer the PLT reconstructs with residual below a threshold, with the median residual and the '
               'median share of the write that goes into later layers.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('reverse', type=Path, nargs='+')
    parser.add_argument('--threshold', type=float, default=.4)
    args = parser.parse_args()
    for path in args.reverse:
        rows = [r for r in json.loads(path.read_text())['rows'] if r['features'] == 1]
        units = []
        for r in rows:
            own = [(res, n) for k, res, n in r['residual_by_offset'] if k == 0]
            if not own:
                continue
            later = sum(n ** 2 for k, _, n in r['residual_by_offset'] if k > 0)
            layer = int(r['unit'][1:].split('F')[0])
            units.append((layer, own[0][0], later / (later + own[0][1] ** 2)))
        layer, res, later = (np.array([u[i] for u in units]) for i in range(3))
        print(path.parent.name, 'units', len(units), f'read at layers 0-1 {np.mean(layer <= 1):.2f}')
        for name, mask in [('layers 0-1', layer <= 1), ('layers 2-12', (layer >= 2) & (layer <= 12)), ('layers 13+', layer >= 13)]:
            if mask.any():
                print(f'  {name:11s} n {int(mask.sum()):3d} shared (residual < {args.threshold}) {np.mean(res[mask] < args.threshold):.2f} '
                      f'median residual {np.median(res[mask]):.2f} median later share {np.median(later[mask]):.2f}')


if __name__ == '__main__':
    main()
