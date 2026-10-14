import argparse
import json

import numpy as np

from p1const import DATASETS, SECOND_SEED, run_dir, save_json

DESCRIPTION = ('P1 validity check from the fine-tunes alone (no forecasts): sigma_t per behaviour from the four two-'
               'seed sets, z = (seed-0 score - base score) / sigma_t, and the fraction of dataset x behaviour pairs '
               'with |z| > 2. The protocol requires at least 10% before the method is judged. Writes '
               'analysis/validity.json.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    rd = run_dir(args.run)
    cfg = json.loads((rd / 'config.json').read_text())
    names = [b['name'] for b in json.loads((rd / 'battery.json').read_text())['behaviours']]
    base = json.loads((rd / 'base_battery_eval.json').read_text())['scores']
    b0 = np.array([base[n] for n in names])

    def ft(d, s):
        sc = json.loads((rd / 'train' / f'{d}_s{s}' / 'battery.json').read_text())['scores']
        return np.array([sc[n] for n in names])

    S0 = np.stack([ft(d, 0) for d in DATASETS])
    sigma = np.sqrt(np.mean([(S0[DATASETS.index(d)] - ft(d, 1)) ** 2 / 2 for d in SECOND_SEED], 0))
    z = (S0 - b0) / sigma
    frac = float(np.mean(np.abs(z) > 2))
    out = dict(frac_pairs_abs_z_gt_2=frac, threshold=cfg['kill']['min_frac_significant'],
               valid=frac >= cfg['kill']['min_frac_significant'],
               per_dataset={d: float(np.mean(np.abs(z[i]) > 2)) for i, d in enumerate(DATASETS)},
               sigma_t_median=float(np.median(sigma)), abs_shift_median=float(np.median(np.abs(S0 - b0))))
    save_json(rd / 'analysis' / 'validity.json', out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
