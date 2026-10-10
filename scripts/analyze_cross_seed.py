from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

DESCRIPTION = ('Part, complete and new-request errors of every method in cross_seed_transfer runs, as effect nRMSE '
               'pooled within each task and averaged over tasks, with a paired bootstrap over runs and pairs.')
FAMILIES = dict(complete=['full'], parts=['verb', 'number', 'gender'],
                new=['verb_number', 'verb_gender', 'number_gender', 'participation_a', 'participation_b',
                     'participation_c', 'half'])
FIXED = ('none_source', 'source', 'none_target')


def load(run):
    assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
    values = dict(np.load(run / 'responses.npz'))
    panel = json.loads((run / 'panel.json').read_text())
    order = panel['query_order']
    tasks = np.array([r['task'] for r in panel['rows']])
    natural = json.loads((run / 'NATURAL.json').read_text())
    return values, order, tasks, natural


def errors(values, order, tasks, method, weights):
    # Effects are measured against each model's own unedited margins.
    out = {}
    src = values['source']-values['none_source']
    eff = values[method]-values['none_target']
    for family, queries in FAMILIES.items():
        ix = [order.index(q) for q in queries]
        per_task = []
        for task in np.unique(tasks):
            sel = tasks == task
            num = (((eff[ix][:, sel]-src[ix][:, sel])**2).sum(0)*weights[..., sel]).sum(-1)
            den = ((src[ix][:, sel]**2).sum(0)*weights[..., sel]).sum(-1)
            per_task.append(np.sqrt(num/den))
        out[family] = np.mean(per_task, axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    loaded = [load(r) for r in args.runs]
    tasks = loaded[0][2]
    methods = [m for m in loaded[0][0] if m not in FIXED]
    point = {m: {f: float(np.mean([errors(v, o, t, m, np.ones(len(t)))[f] for v, o, t, _ in loaded])) for f in FAMILIES}
             for m in methods}
    per_run = {m: [{f: float(errors(v, o, t, m, np.ones(len(t)))[f]) for f in FAMILIES} for v, o, t, _ in loaded]
               for m in methods}
    rng = np.random.default_rng(20261010)
    item_w = np.zeros((args.draws, len(tasks)))
    for task in np.unique(tasks):
        idx = np.flatnonzero(tasks == task)
        picks = rng.choice(idx, size=(args.draws, len(idx)))
        for d in range(args.draws):
            np.add.at(item_w[d], picks[d], 1.)
    run_w = np.stack([np.bincount(rng.integers(0, len(loaded), len(loaded)), minlength=len(loaded))
                      for _ in range(args.draws)])/len(loaded)
    boot = {m: {f: (np.stack([errors(v, o, t, m, item_w)[f] for v, o, t, _ in loaded], 1)*run_w).sum(1) for f in FAMILIES}
            for m in methods}
    contrasts = {}
    for a in methods:
        for b in methods:
            if a < b:
                contrasts[f'{a}-{b}'] = {f: dict(point=point[a][f]-point[b][f],
                                                 interval=np.quantile(boot[a][f]-boot[b][f], [.025, .975]).tolist())
                                         for f in FAMILIES}
    natural = {m: float(np.mean([n[m] for _, _, _, n in loaded])) for m in loaded[0][3]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                           runs=[str(r) for r in args.runs], point=point, per_run=per_run,
                                           intervals={m: {f: np.quantile(boot[m][f], [.025, .975]).tolist() for f in FAMILIES}
                                                      for m in methods},
                                           contrasts=contrasts, natural=natural, draws=args.draws), indent=1))
    for m in methods:
        print(f'{m:<22}', '  '.join(f'{f} {point[m][f]:.3f}' for f in FAMILIES), f"  natural {natural.get(m, float('nan')):.3f}")


if __name__ == '__main__':
    main()
