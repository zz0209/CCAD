import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Which units the unit test drops from the held-out CLT graphs of the filter-prediction runs. Per model, over '
               'prompts: the number of failing units among the tested pool, the share of failing units read at layers 0 '
               'and 1 and the share of pool units that fail at layers 0-1 and later (pooled over prompts and as medians '
               'over prompts), and the share of the test list that the layer list also contains.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--models', nargs='+', required=True, help='name=CLT_ROWS.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for spec in args.models:
        name, path = spec.split('=')
        rows = json.loads(Path(path).read_text())['rows']
        n_fail, fail_early, early_fail, later_fail, overlap = [], [], [], [], []
        tot = dict(fail=0, fail_early=0, early=0, later=0, later_fail=0)
        for r in rows:
            pool = r['pool']
            failing = [u for u in pool if not u['passes']]
            early = [u for u in pool if u['layer'] <= 1]
            later = [u for u in pool if u['layer'] >= 2]
            n_fail.append(len(failing))
            if failing:
                fail_early.append(sum(u['layer'] <= 1 for u in failing) / len(failing))
            if early:
                early_fail.append(sum(not u['passes'] for u in early) / len(early))
            if later:
                later_fail.append(sum(not u['passes'] for u in later) / len(later))
            test = {u['node'] for u in r['lists']['test']['units']}
            layer = {u['node'] for u in r['lists']['layer']['units']}
            if test:
                overlap.append(len(test & layer) / len(test))
            tot['fail'] += len(failing)
            tot['fail_early'] += sum(u['layer'] <= 1 for u in failing)
            tot['early'] += len(early)
            tot['later'] += len(later)
            tot['later_fail'] += sum(not u['passes'] for u in later)
        summary[name] = dict(
            prompts=len(rows), pool=len(rows[0]['pool']), failing_median=float(np.median(n_fail)),
            failing_early_share_pooled=tot['fail_early'] / tot['fail'],
            failing_early_share_median=float(np.median(fail_early)),
            early_failing_share_pooled=tot['fail_early'] / tot['early'],
            early_failing_share_median=float(np.median(early_fail)),
            later_failing_share_pooled=tot['later_fail'] / tot['later'],
            later_failing_share_median=float(np.median(later_fail)),
            test_in_layer_share_median=float(np.median(overlap)))
        print(name, json.dumps(summary[name]))
    args.out.write_text(json.dumps(dict(description=DESCRIPTION, models=summary), indent=1))


if __name__ == '__main__':
    main()
