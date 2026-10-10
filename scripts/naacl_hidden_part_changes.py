from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

from naacl_analyze_e1_shift import PARTS, load

DESCRIPTION = ('For each SHIFT correspondence, the share of test biographies on which its complete deletion gives the '
               'source answer while at least one single-part deletion gives a different answer than the source.')
METHODS = ['hungarian_decoder', 'hungarian_activation', 'geometry', 'activation_top2', 'geometry_gain', 'native',
           'raw_reconstruction', 'whole', 'parts_relation', 'parts']


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = {m: dict(complete_agree=0, hidden=0) for m in METHODS}
    per_target = {}
    for run in args.runs:
        assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
        _, values, _, _ = load(run)
        answer = lambda m, q: values[(m, q)] > 0
        row = {}
        for m in METHODS:
            agree = answer(m, 'full') == answer('source', 'full')
            differs = np.any([answer(m, p) != answer('source', p) for p in PARTS], axis=0)
            counts[m]['complete_agree'] += int(agree.sum())
            counts[m]['hidden'] += int((agree & differs).sum())
            row[m] = float((agree & differs).sum() / agree.sum())
        per_target[run.name] = row
    summary = {m: c['hidden'] / c['complete_agree'] for m, c in counts.items()}
    for m, v in summary.items():
        print(f'{m:<22} {100 * v:5.1f}%  of {counts[m]["complete_agree"]} biographies with the source complete answer')
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                           runs=[str(r) for r in args.runs], summary=summary, counts=counts,
                                           per_target=per_target), indent=1))


if __name__ == '__main__':
    main()
