import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('H-inv: relation between how much of a published supernode another dictionary contains (edit-site '
               'residual and slice-deletion effect error of P1e) and how much of its causal effect carries over to '
               'concept-preserving prompt variants (P2). Supernodes whose deletion moves the answer log-probability '
               'by less than --min-drop on the published prompt are left out; retention is the median over variants '
               'of the variant drop divided by the published drop. Spearman correlations with a bootstrap over graphs.')


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--generalize', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, nargs='+', required=True)
    parser.add_argument('--min-drop', type=float, default=.1)
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    gen = json.loads(args.generalize.read_text())['rows']
    rng = np.random.default_rng(0)
    for path in args.oracle:
        orc = {(r['graph'], r['supernode']): r for r in json.loads(path.read_text())['rows']}
        rows = []
        for g in gen:
            o = orc.get((g['graph'], g['supernode']))
            if o is None or g['drop_original'] < args.min_drop or not g['variants'] or o.get('site_residual') is None:
                continue
            ret = np.median([v['drop'] / g['drop_original'] for v in g['variants']])
            act = np.mean([v['active'] for v in g['variants']])
            rows.append((g['graph'], 1 - o['site_residual'], o['nrmse_slice'], ret, act, g['drop_original']))
        graphs = sorted({r[0] for r in rows})
        cont, sl, ret, act, d0 = (np.array([r[i] for r in rows]) for i in range(1, 6))
        print(path.parent.name, 'supernodes', len(rows), 'graphs', len(graphs))
        print(f'   retention median {np.median(ret):.3f}; variant activity median {np.median(act):.3f}')
        for name, x in [('containment (1 - residual)', cont), ('slice-deletion effect error', sl),
                        ('variant activity', act), ('published drop', d0)]:
            r = spearman(x, ret)
            boot = []
            for _ in range(args.draws):
                pick = rng.choice(graphs, len(graphs))
                idx = np.concatenate([[i for i, row in enumerate(rows) if row[0] == g] for g in pick]).astype(int)
                if len(set(idx)) > 3:
                    boot.append(spearman(x[idx], ret[idx]))
            print(f'   spearman({name}, retention) {r:+.3f} [{np.quantile(boot, .025):+.3f}, {np.quantile(boot, .975):+.3f}]')
        hi = cont >= np.median(cont)
        print(f'   retention, upper half of containment {np.median(ret[hi]):.3f}, lower half {np.median(ret[~hi]):.3f}')


if __name__ == '__main__':
    main()
