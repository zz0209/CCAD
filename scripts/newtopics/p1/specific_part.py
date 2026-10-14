import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from p1const import save_json

DESCRIPTION = ('P1 diagnostic outside the frozen criteria: the set-specific part of each shift. Each set\'s z and each '
               'forecaster\'s leave-one-dataset-out forecast are centred by the mean over the other 11 sets, then '
               'compared by pooled Spearman, sign accuracy (|centred z| > 2) and per-set Spearman. Reads '
               'analysis/table*.json and writes analysis/specific_part*.json next to it.')


def loo_center(a):
    n = len(a)
    return np.stack([a[i] - a[[j for j in range(n) if j != i]].mean(0) for i in range(n)])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--table', type=Path, required=True)
    args = parser.parse_args()
    t = json.loads(args.table.read_text())
    zs = loo_center(np.array(t['z']))
    m = np.abs(zs) > 2
    out = {}
    for f, p in t['lodo_pred'].items():
        ps = loo_center(np.array(p))
        out[f] = dict(rho_specific=float(spearmanr(ps.ravel(), zs.ravel())[0]),
                      sign_acc_specific=float(np.mean(np.sign(ps[m]) == np.sign(zs[m]))),
                      per_dataset_rho_specific={d: float(spearmanr(ps[i], zs[i])[0])
                                                for i, d in enumerate(t['datasets'])})
    save_json(args.table.with_name(args.table.name.replace('table', 'specific_part')), out)
    print(json.dumps({f: round(v['rho_specific'], 3) for f, v in out.items()}))


if __name__ == '__main__':
    main()
