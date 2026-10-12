import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Do the units of the CLT\'s own graphs that the PLT can also express generalize better than the CLT-specific '
               'ones? Joins the CLT unit generalization rows (P2 with --units-from) with the converse reconstruction rows '
               '(P1r) and relates each unit\'s retention on concept-preserving prompt variants to its PLT site residual, '
               'its read layer and the share of its write in later layers, with a bootstrap over graphs.')


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--generalize', type=Path, required=True)
    parser.add_argument('--reverse', type=Path, required=True)
    parser.add_argument('--min-drop', type=float, default=.1)
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    rev = {(r['graph'], r['unit']): r for r in json.loads(args.reverse.read_text())['rows']}
    rows = []
    for g in json.loads(args.generalize.read_text())['rows']:
        r = rev.get((g['graph'], g['supernode']))
        if r is None or g['drop_original'] < args.min_drop or not g['variants']:
            continue
        own = [(res, n) for k, res, n in r['residual_by_offset'] if k == 0]
        later = sum(n ** 2 for k, _, n in r['residual_by_offset'] if k > 0)
        total = later + sum(n ** 2 for _, n in own)
        layer = int(g['supernode'][1:].split('F')[0])
        rows.append(dict(graph=g['graph'], unit=g['supernode'], layer=layer,
                         retention=float(np.median([v['drop'] / g['drop_original'] for v in g['variants']])),
                         own_residual=own[0][0] if own else float('nan'), later_share=later / total if total else float('nan'),
                         drop=g['drop_original']))
    rows = [r for r in rows if np.isfinite(r['own_residual'])]
    ret = np.array([r['retention'] for r in rows])
    rng = np.random.default_rng(0)
    graphs = sorted({r['graph'] for r in rows})
    print('units', len(rows), 'graphs', len(graphs), 'median retention', round(float(np.median(ret)), 3))
    for name in ['own_residual', 'layer', 'later_share', 'drop']:
        x = np.array([r[name] for r in rows])
        boot = []
        for _ in range(args.draws):
            pick = rng.choice(graphs, len(graphs))
            idx = np.concatenate([[i for i, r in enumerate(rows) if r['graph'] == g] for g in pick]).astype(int)
            boot.append(spearman(x[idx], ret[idx]))
        print(f'  spearman({name}, retention) {spearman(x, ret):+.3f} [{np.quantile(boot, .025):+.3f}, {np.quantile(boot, .975):+.3f}]')
    shared = np.array([r['own_residual'] < .4 for r in rows])
    early = np.array([r['layer'] <= 1 for r in rows])
    for name, mask in [('shared (own-layer PLT residual < .4)', shared), ('CLT-specific (>= .4)', ~shared),
                       ('read at layer 0 or 1', early), ('read at layer 2 or later', ~early)]:
        print(f'  {name:40s} n {int(mask.sum()):3d} median retention {np.median(ret[mask]):.3f} '
              f'[{np.quantile(ret[mask], .25):.3f}, {np.quantile(ret[mask], .75):.3f}]')


if __name__ == '__main__':
    main()
