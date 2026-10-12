import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Effects of deleting the writes of the always-active early CLT features, from the unit-test rows: for every '
               'top unit of a graph that is one of the listed features, the size of the change of the next-token '
               'distribution (probability-weighted norm of the change of log-probabilities at the final position) and '
               'the drop of the answer\'s log-probability when the unit\'s claimed writes are deleted with the later MLP '
               'outputs at its position clamped.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path)
    parser.add_argument('--features', nargs='+', default=['L0F7081', 'L1F3148', 'L1F3872'])
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.rows.read_text())['rows'] if r['unit'].split('P')[0] in args.features]
    effect = np.array([r['energy_w_clamped'] ** .5 for r in rows])
    drop = np.array([r['drop_w_clamped'] for r in rows])
    final = np.array([r['final'] for r in rows])
    out = dict(units=len(rows), graphs=len({r['graph'] for r in rows}), final_units=int(final.sum()),
               effect_median_all=float(np.median(effect)), effect_median_final=float(np.median(effect[final])),
               answer_drop_over_half=int((drop > .5).sum()), answer_drop_median=float(np.median(drop)),
               answer_drop_over_half_final=int((drop[final] > .5).sum()),
               answer_drop_median_final=float(np.median(drop[final])))
    print(json.dumps(out, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
