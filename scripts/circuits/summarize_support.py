import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Write-support test of P1e rows: relative residual of reconstructing a single-layer source write from '
               'whole target features matched over all layers (their writes into other layers count as error), next '
               'to the slice residual at the written layer, and the share of squared write norm that active target '
               'candidates put into later layers.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path, nargs='+')
    args = parser.parse_args()
    for path in args.rows:
        rows = [r for r in json.loads(path.read_text())['rows'] if r.get('site_residual_whole') is not None]
        f = lambda k: np.array([r[k] for r in rows])
        print(path.parent.name, 'supernodes', len(rows))
        for k in ['site_residual', 'site_residual_whole', 'site_residual_random', 'candidate_later_share']:
            v = f(k)
            print(f'   {k:24s} median {np.median(v):.3f} [{np.quantile(v, .25):.3f}, {np.quantile(v, .75):.3f}]')
        print(f'   whole residual above slice residual on {np.mean(f("site_residual_whole") > f("site_residual")):.2f} of supernodes')


if __name__ == '__main__':
    main()
