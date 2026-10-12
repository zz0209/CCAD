import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('E1 pre-check: medians of the replacement and completeness scores of the CLT\'s and the PLT\'s own graphs '
               'on the same prompts, and the median of the paired CLT-minus-PLT difference with a bootstrap interval over '
               'prompts.')


def load(path):
    return {r['graph']: r for r in json.loads((path / 'GRAPH_SCORES.json').read_text())['rows']}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--pairs', nargs='+', required=True, help='name=CLT_RUN_DIR,PLT_RUN_DIR')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    for spec in args.pairs:
        name, dirs = spec.split('=')
        clt, plt = (load(Path(p)) for p in dirs.split(','))
        graphs = [g for g in clt if g in plt]
        d = dict(prompts=len(graphs))
        for score in ('replacement', 'completeness'):
            c = np.array([clt[g][score] for g in graphs])
            p = np.array([plt[g][score] for g in graphs])
            diff = c - p
            boot = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
            d[score] = dict(clt=float(np.median(c)), plt=float(np.median(p)), diff=float(np.median(diff)),
                            diff_ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                            clt_higher=int((diff > 0).sum()))
        d['features'] = dict(clt=float(np.median([clt[g]['features'] for g in graphs])),
                             plt=float(np.median([plt[g]['features'] for g in graphs])))
        out[name] = d
        print(name, json.dumps(d))
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
