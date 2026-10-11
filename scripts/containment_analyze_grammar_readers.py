from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
DESCRIPTION = ('Held-out grammar readers over all directions: effect nRMSE of every map on the margins of each held-out '
               'BLiMP paradigm, on the next-token distribution of their sentences and of generic text, and E2, with a '
               'bootstrap that resamples directions and pairs within paradigms, shared across maps.')


def load(run):
    assert json.loads((run / 'status.json').read_text())['status'] == 'PASS', run
    z = dict(np.load(run / 'responses.npz'))
    families = json.loads((run / 'requests.json').read_text())['families']
    return z, families


def errors(z, families, name, wp, wd):
    req = list(z['request_names'])
    tasks = z['tasks']
    base = z['pair|none|margin'][0]
    out = {}
    for fam, reqs in families.items():
        rows = [req.index(r) for r in reqs]
        src = z['pair|source|margin'][rows] - base
        mine = z[f'pair|{name}|margin'][rows] - base
        cell = {}
        for p in sorted(set(tasks)):
            sel = tasks == p
            # A paradigm whose resampled pairs all have zero source effect has no defined error in that draw.
            den = ((src[:, sel] ** 2).sum(0) * wp[..., sel]).sum(-1)
            num = (((mine - src)[:, sel] ** 2).sum(0) * wp[..., sel]).sum(-1)
            cell[f'margin_{p}'] = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0))
        cell['heldout_next_token'] = np.sqrt((z[f'pair|{name}|nt_err'][rows].sum(0) * wp).sum(-1) /
                                             (z[f'pair|{name}|nt_eff'][rows].sum(0) * wp).sum(-1))
        cell['fineweb_next_token'] = np.sqrt((z[f'doc|{name}|nt_err'][rows].sum(0) * wd).sum(-1) /
                                             (z[f'doc|{name}|nt_eff'][rows].sum(0) * wd).sum(-1))
        cell['E2'] = np.nanmean([cell[k] for k in cell if k.startswith('margin_')] + [cell['heldout_next_token']], 0)
        out[fam] = cell
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loaded = [load(RUNS / r) for r in args.runs]
    names = [k.split('|')[1] for k in loaded[0][0] if k.startswith('pair|') and k.endswith('|margin') and '|none|' not in k
             and '|source|' not in k]
    rng = np.random.default_rng(20261011)
    tasks = loaded[0][0]['tasks']
    n_pairs, n_docs = len(tasks), loaded[0][0]['doc|source|nt_err'].shape[1]
    wp = np.zeros((args.draws, n_pairs))
    for p in np.unique(tasks):
        idx = np.flatnonzero(tasks == p)
        picks = rng.choice(idx, size=(args.draws, len(idx)))
        for d in range(args.draws):
            np.add.at(wp[d], picks[d], 1.)
    wd = np.zeros((args.draws, n_docs))
    picks = rng.choice(n_docs, size=(args.draws, n_docs))
    for d in range(args.draws):
        np.add.at(wd[d], picks[d], 1.)
    run_w = np.stack([np.bincount(rng.integers(0, len(loaded), len(loaded)), minlength=len(loaded))
                      for _ in range(args.draws)]) / len(loaded)
    point, boot = {}, {}
    for name in names:
        per_run = [errors(z, f, name, np.ones(n_pairs), np.ones(n_docs)) for z, f in loaded]
        per_boot = [errors(z, f, name, wp, wd) for z, f in loaded]
        point[name] = {fam: {k: float(np.mean([r[fam][k] for r in per_run])) for k in per_run[0][fam]} for fam in per_run[0]}
        boot[name] = {}
        for fam in per_boot[0]:
            boot[name][fam] = {}
            for k in per_boot[0][fam]:
                vals = np.stack([r[fam][k] for r in per_boot], 1)
                weight = run_w * ~np.isnan(vals)
                boot[name][fam][k] = np.nansum(vals * run_w, 1) / weight.sum(1)
    contrasts = {}
    for a in names:
        for b in names:
            if a < b:
                contrasts[f'{a}-{b}'] = {fam: {k: dict(point=point[a][fam][k] - point[b][fam][k],
                                                       interval=np.quantile(boot[a][fam][k] - boot[b][fam][k], [.025, .975]).tolist())
                                               for k in ['E2', 'heldout_next_token', 'fineweb_next_token']}
                                         for fam in point[a]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                           runs=args.runs, draws=args.draws, point=point, contrasts=contrasts), indent=1))
    for fam in ['parts', 'random']:
        print(fam)
        for name in names:
            p = point[name][fam]
            print(f'  {name:<22} E2 {p["E2"]:.3f}  heldout_nt {p["heldout_next_token"]:.3f}  fineweb {p["fineweb_next_token"]:.3f}  '
                  + '  '.join(f'{k[7:17]} {v:.3f}' for k, v in p.items() if k.startswith('margin_')))


if __name__ == '__main__':
    main()
