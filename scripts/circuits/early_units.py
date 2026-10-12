import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

DESCRIPTION = ('Kinds of CLT units read at layers 0 and 1: for each unit, the fraction of its later writes that the '
               'model removes when the PLT units of the prompt at or before its token are removed, in excess of '
               'random PLT features (collapsed when above .3), and how often the same feature recurs among the top '
               'units of different graphs. Recurring features are listed with their share of the early units\' '
               'clamped write effect.')


def later(r, key):
    c = np.array(r['claim_energy'][1:])
    return float(np.sum(r[key][1:]) / c.sum())


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path, nargs='+')
    parser.add_argument('--min-graphs', type=int, default=5, help='a feature recurs when it is a top unit in this many graphs')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    summary = {}
    for path in args.rows:
        rows = [r for r in json.loads(path.read_text())['rows'] if r['layer'] <= 1 and len(r['claim_energy']) > 1]
        feats = Counter(r['unit'].split('P')[0] for r in rows)
        graphs = {f: len({r['graph'] for r in rows if r['unit'].split('P')[0] == f}) for f in feats}
        tested = [r for r in rows if 'removed_a' in r and 'removed_ra' in r]
        excess = np.array([later(r, 'removed_a') - later(r, 'removed_ra') for r in tested])
        effect = np.array([r['energy_w_clamped'] ** .5 for r in tested])
        collapsed = excess > .3
        recurring = {f for f, n in graphs.items() if n >= args.min_graphs}
        rec = np.array([r['unit'].split('P')[0] in recurring for r in tested])
        d = dict(units=len(rows), tested=len(tested), collapsed=int(collapsed.sum()),
                 collapsed_share=float(collapsed.mean()),
                 collapsed_effect_share=float(effect[collapsed].sum() / effect.sum()),
                 recurring_features={f: graphs[f] for f in sorted(recurring, key=lambda f: -graphs[f])},
                 recurring_units=int(rec.sum()), recurring_effect_share=float(effect[rec].sum() / effect.sum()),
                 recurring_collapsed=int((rec & collapsed).sum()),
                 median_excess_recurring=float(np.median(excess[rec])) if rec.any() else None,
                 median_excess_other=float(np.median(excess[~rec])) if (~rec).any() else None,
                 n_graphs=len({r['graph'] for r in rows}))
        summary[path.parent.name] = d
        print(path.parent.name, json.dumps(d))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
