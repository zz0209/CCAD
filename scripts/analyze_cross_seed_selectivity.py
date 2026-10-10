from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = ('Part-by-task selectivity of each method: the mean margin effect of deleting each named part on each '
               'task, compared with the same matrix of the source explanation in the source model.')
PARTS = ['verb', 'number', 'gender']


def matrix(values, order, tasks, task_names, method, base):
    effect = values[method]-values[base]
    return np.array([[effect[order.index(p)][tasks == t].mean() for t in task_names] for p in PARTS])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    for run in args.runs:
        values = dict(np.load(run / 'responses.npz'))
        panel = json.loads((run / 'panel.json').read_text())
        order, tasks = panel['query_order'], np.array([r['task'] for r in panel['rows']])
        task_names = list(dict.fromkeys(r['task'] for r in panel['rows']))
        src = matrix(values, order, tasks, task_names, 'source', 'none_source')
        print(run.name)
        print('  source', np.round(src, 2).tolist())
        for method in [k for k in values if k not in ('none_source', 'source', 'none_target')]:
            mat = matrix(values, order, tasks, task_names, method, 'none_target')
            rel = np.linalg.norm(mat-src)/np.linalg.norm(src)
            diag = np.trace(mat)/np.abs(mat).sum()
            print(f'  {method:<22} rel_error {rel:.3f}  diagonal share {diag:.3f}  scale {np.linalg.norm(mat)/np.linalg.norm(src):.2f}',
                  np.round(mat, 2).tolist())
        print(f"  source diagonal share {np.trace(src)/np.abs(src).sum():.3f}")


if __name__ == '__main__':
    main()
