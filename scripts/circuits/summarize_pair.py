import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = 'Summary of CLT-to-CLT transfer rows (P1c): residuals, controls, effect errors and top-token agreement.'


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path)
    parser.add_argument('--min-energy', type=float, default=1e-3)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.rows.read_text())['rows'] if r['source_energy'] > args.min_energy]
    q = lambda k: (np.median([r[k] for r in rows]), np.quantile([r[k] for r in rows], .25), np.quantile([r[k] for r in rows], .75))
    print(args.rows.parent.name, 'units with source effect', len(rows))
    for k in ['best_corr', 'residual_whole', 'residual_whole_random', 'residual_slices', 'nrmse_whole', 'nrmse_slices']:
        m, a, b = q(k)
        print(f'   {k:22s} median {m:.3f} [{a:.3f}, {b:.3f}]')
    changed = [r for r in rows if r['top_source'] != r['top_base']]
    for k in ['whole', 'slices']:
        print(f'   share below .5 ({k}): {np.mean([r[f"nrmse_{k}"] < .5 for r in rows]):.2f}; same new top token on '
              f'{len(changed)} changing units: {np.mean([r[f"top_{k}"] == r["top_source"] for r in changed]):.2f}')


if __name__ == '__main__':
    main()
