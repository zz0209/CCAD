import argparse
import json
import time

import numpy as np
import torch

from battery import Battery
from p1common import DATASETS, LAYERS, load_model, load_tokenizer, run_dir, save_json

DESCRIPTION = ('P1 steering forecasters: for each training set, layer and strength, add alpha * v to the residual '
               'stream after that layer at every position and score the held-out battery. v is the self-contrast '
               "vector (dataset-response mean minus the model's own-response mean, forecaster sc) or the raw dataset-"
               'response mean (forecaster mdf). For each layer, alpha = c * (mean own-response token norm) / (median '
               'over the 12 sets of |v|), one global value per (forecaster, layer, c). Writes forecast/steer/.')


def vectors(sc_means, kind, name, layer):
    m = sc_means[name]
    if kind == 'sc':
        return m['data_mean'][layer] - m['own_mean'][layer]
    assert kind == 'mdf'
    return m['data_mean'][layer]


def alpha_table(sc_means, cfg, names):
    tab = dict(datasets=names)
    for kind in ['sc', 'mdf']:
        tab[kind] = {}
        for l in LAYERS:
            r = float(np.mean([sc_means[d]['own_token_norm'][l] for d in names]))
            med = float(np.median([float(vectors(sc_means, kind, d, l).norm()) for d in names]))
            tab[kind][str(l)] = dict(resid_norm=r, median_vec_norm=med,
                                     alphas=[c * r / med for c in cfg['alpha_c']])
    return tab


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', required=True)
    parser.add_argument('--kinds', nargs='+', default=['sc', 'mdf'])
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--layers', nargs='+', type=int, default=LAYERS)
    parser.add_argument('--cs', nargs='+', type=int, default=None, help='indices into alpha_c')
    parser.add_argument('--allow_partial', action='store_true', help='smoke tests: alphas from the sets present')
    args = parser.parse_args()
    rd = run_dir(args.run)
    cfg = json.loads((rd / 'config.json').read_text())
    sc_means = torch.load(rd / 'forecast' / 'sc_means.pt')
    names = [d for d in DATASETS if d in sc_means]
    assert args.allow_partial or names == DATASETS
    tab = alpha_table(sc_means, cfg, names)
    save_json(rd / 'forecast' / 'alpha_table.json', tab)
    cs = range(len(cfg['alpha_c'])) if args.cs is None else args.cs
    jobs = [(k, d, l, c) for k in args.kinds for d in args.datasets for l in args.layers for c in cs]
    todo = [j for j in jobs if not (rd / 'forecast' / 'steer' / j[0] / j[1] / f'l{j[2]}_c{j[3]}.json').exists()]
    print(f'{len(jobs)} steering evaluations, {len(jobs) - len(todo)} already done', flush=True)
    tok = load_tokenizer()
    bat = Battery(tok, json.loads((rd / 'battery.json').read_text()), 'eval')
    lm = load_model()
    t0 = time.time()
    for n, (k, d, l, c) in enumerate(todo):
        alpha = tab[k][str(l)]['alphas'][c]
        vec = (alpha * vectors(sc_means, k, d, l)).cuda()
        scores = bat.score(lm, l, vec)
        save_json(rd / 'forecast' / 'steer' / k / d / f'l{l}_c{c}.json',
                  dict(kind=k, dataset=d, layer=l, c=cfg['alpha_c'][c], alpha=alpha, vec_norm=float(vec.norm()),
                       scores=scores))
        el = time.time() - t0
        print(f'steer {n + 1}/{len(todo)} ({k} {d} l{l} c{c}) {el:.0f}s elapsed, eta {el / (n + 1) * (len(todo) - n - 1):.0f}s',
              flush=True)


if __name__ == '__main__':
    main()
