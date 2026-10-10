from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = ('SAE-free behavior of the source and target models on the evaluation pairs: margin accuracy per task, '
               'correlation of unedited margins, and the size of the source explanation effect in the source model.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    for run in args.runs:
        values = dict(np.load(run / 'responses.npz'))
        panel = json.loads((run / 'panel.json').read_text())
        order = panel['query_order']
        tasks = np.array([r['task'] for r in panel['rows']])
        a, b = values['none_source'][0], values['none_target'][0]
        full = order.index('full')
        effect = values['source'][full]-values['none_source'][full]
        out = dict(run=run.name, correlation=float(np.corrcoef(a, b)[0, 1]))
        for task in np.unique(tasks):
            sel = tasks == task
            out[task[:14]] = dict(acc_source=float((a[sel] > 0).mean()), acc_target=float((b[sel] > 0).mean()),
                                  corr=float(np.corrcoef(a[sel], b[sel])[0, 1]),
                                  source_effect_rms=float(np.sqrt((effect[sel]**2).mean())),
                                  margin_rms=float(np.sqrt((a[sel]**2).mean())))
        print(json.dumps(out))


if __name__ == '__main__':
    main()
