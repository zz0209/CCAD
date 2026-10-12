import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Edit-site containment by depth: for each published supernode, its mean layer (from the registry) '
               'against the relative residual of reconstructing its writes in the target dictionary and against the '
               'random-direction control, in layer bands, with the Spearman correlation between layer and residual.')


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('rows', type=Path, nargs='+')
    args = parser.parse_args()
    graphs = json.loads(args.registry.read_text())['graphs']
    for path in args.rows:
        rows = [r for r in json.loads(path.read_text())['rows'] if r.get('site_residual') is not None]
        layer = np.array([np.mean([l for l, _, _ in graphs[r['graph']]['variants'][0]['supernodes'][r['supernode']] if l < 26])
                          for r in rows])
        res = np.array([r['site_residual'] for r in rows])
        rnd = np.array([r['site_residual_random'] for r in rows])
        print(path.parent.name, 'supernodes', len(rows), 'spearman(layer, residual)', round(spearman(layer, res), 3))
        for lo, hi in [(0, 6), (6, 13), (13, 20), (20, 26)]:
            m = (layer >= lo) & (layer < hi)
            if m.any():
                print(f'   layers {lo:2d}-{hi - 1:2d}: n {m.sum():3d}  residual {np.median(res[m]):.3f}  random {np.median(rnd[m]):.3f}')


if __name__ == '__main__':
    main()
