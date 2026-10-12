import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Reverse-direction summary: reconstruction residual of CLT unit writes in the per-layer transcoder, '
               'split into the write into the unit\'s own layer and the writes into later layers (norm-weighted), and '
               'the share of each reconstruction carried by published supernode features of the same graph, layer and '
               'position, for single-feature units and for position groups.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path, nargs='+')
    args = parser.parse_args()
    for path in args.rows:
        rows = json.loads(path.read_text())['rows']
        own, later = [], []
        for r in rows:
            for k, res, norm in r.get('residual_by_offset', []):
                (own if k == 0 else later).append((res, norm))
        wavg = lambda xs: sum(r * n for r, n in xs) / max(sum(n for _, n in xs), 1e-12)
        print(path.parent.name, 'units', len(rows))
        print(f'   residual, own layer: median {np.median([r for r, _ in own]):.3f}, norm-weighted {wavg(own):.3f} ({len(own)} writes)')
        print(f'   residual, later layers: median {np.median([r for r, _ in later]):.3f}, norm-weighted {wavg(later):.3f} ({len(later)} writes)')
        norm_own = sum(n for _, n in own)
        norm_later = sum(n for _, n in later)
        print(f'   share of write norm into later layers {norm_later / (norm_own + norm_later):.3f}')
        for kind, sel in [('single features', [r for r in rows if r['features'] == 1]),
                          ('position groups', [r for r in rows if r['features'] > 1])]:
            sh = np.array([r['published_share'] for r in sel if r.get('published_share') is not None])
            if len(sh):
                print(f'   {kind}: published share median {np.median(sh):.3f}, mean {sh.mean():.3f}, '
                      f'>.5 {np.mean(sh > .5):.2f}, =0 {np.mean(sh == 0):.2f} (n {len(sh)})')
        # Published supernodes reached by any top CLT unit with at least 10% of its reconstruction.
        reached = {}
        for r in rows:
            for name, share in r.get('published_supernodes', {}).items():
                if share >= .1:
                    reached.setdefault(r['graph'], set()).add(name)
        print(f'   graphs with a published supernode carrying >=10% of some top CLT unit: {len(reached)}; '
              f'supernodes reached: {sum(len(v) for v in reached.values())}')


if __name__ == '__main__':
    main()
