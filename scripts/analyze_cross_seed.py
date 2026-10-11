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
    parser.add_argument('--merge-runs', nargs='*', type=Path, default=[],
                        help=('runs on the same targets and pairs, one group per pass over --runs in the same order, whose '
                              'methods are added under the suffix @merged, @merged2 and so on by group'))
    args = parser.parse_args()
    loaded = [load(r) for r in args.runs]
    for i, run in enumerate(args.merge_runs):
        values, order, tasks, natural = load(run)
        base = loaded[i % len(loaded)]
        suffix = '@merged' if i < len(loaded) else f'@merged{i // len(loaded) + 1}'
        assert order == base[1] and np.array_equal(tasks, base[2])
        # Runs on different GPUs agree to float rounding (about 1e-4 on margins of a few nats).
        assert all(np.allclose(values[k], base[0][k], atol=1e-3) for k in FIXED)
        base[0].update({f'{k}{suffix}': v for k, v in values.items() if k not in FIXED})
        base[3].update({f'{k}{suffix}': v for k, v in natural.items()})
    tasks = loaded[0][2]
    # Methods that every run carries; runs with extra families still contribute the shared ones.
    methods = [m for m in loaded[0][0] if m not in FIXED and all(m in v for v, _, _, _ in loaded)]
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
    natural = {m: float(np.mean([n[m] for _, _, _, n in loaded])) for m in loaded[0][3] if all(m in n for _, _, _, n in loaded)}
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
