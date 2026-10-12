import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Summary of per-supernode transfer rows: median and quartiles of the effect nRMSE of each target method '
               'over supernodes whose source deletion has a nonzero effect, the share below .5, and agreement of the '
               'top next token after deletion with the source dictionary.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path)
    parser.add_argument('--min-energy', type=float, default=1e-3)
    args = parser.parse_args()
    data = json.loads(args.rows.read_text())
    rows = [r for r in data['rows'] if r.get('source_energy', 0) > args.min_energy]
    print(args.rows.parent.name, 'supernodes with source effect:', len(rows), 'of', len(data['rows']))
    if 'heldout_text' in data:
        print('held-out text nRMSE:', {k: round(v, 3) for k, v in data['heldout_text'].items()})
    if 'site_residual' in rows[0]:
        v = np.array([r['site_residual'] for r in rows if r['site_residual'] is not None])
        print(f'edit-site relative residual median {np.median(v):.3f} [{np.quantile(v, .25):.3f}, {np.quantile(v, .75):.3f}]')
        if rows[0].get('site_residual_random') is not None:
            v = np.array([r['site_residual_random'] for r in rows if r['site_residual_random'] is not None])
            print(f'random direction of equal norm, same active features: median {np.median(v):.3f} '
                  f'[{np.quantile(v, .25):.3f}, {np.quantile(v, .75):.3f}]')
    keys = [k for k in rows[0] if k.startswith('nrmse')]
    for k in keys:
        v = np.array([r[k] for r in rows if r[k] is not None])
        method = k[6:] or 'one_to_one'
        top = [r for r in rows if f'top_{method}' in r or 'top_target' in r]
        agree = np.mean([r.get(f'top_{method}', r.get('top_target')) == r['top_source'] for r in top])
        print(f'{k:20s} median {np.median(v):.3f} [{np.quantile(v, .25):.3f}, {np.quantile(v, .75):.3f}]  '
              f'share<.5 {np.mean(v < .5):.2f}  top-token agreement {agree:.2f}')
    changed = [r for r in rows if r['top_source'] != r['top_base']]
    print('supernodes whose source deletion changes the top token:', len(changed))
    for k in keys:
        method = k[6:] or 'one_to_one'
        agree = np.mean([r.get(f'top_{method}', r.get('top_target')) == r['top_source'] for r in changed]) if changed else float('nan')
        print(f'   {method}: same new top token {agree:.2f}')


if __name__ == '__main__':
    main()
