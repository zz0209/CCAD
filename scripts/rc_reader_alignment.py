from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
DESCRIPTION = ('Alignment of each held-out scalar reader with the fitted classifier, measured as the correlation of their '
               'source effects over biographies and part requests, against the error of a classifier fit relative to the '
               'state fit on that reader. Coverage predicts that a classifier fit protects the readers whose effects '
               'follow the fitted classifier.')


def nrmse(z, name, reader, rows, clean):
    src = z[f'bio|source|logit_{reader}'][rows] - clean
    mine = z[f'bio|{name}|logit_{reader}'][rows] - clean
    return float(np.sqrt(((mine - src) ** 2).sum() / (src ** 2).sum()))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--classifier', nargs='+', required=True, help='classifier-fit map name per run, without seed')
    parser.add_argument('--state', nargs='+', required=True, help='state-fit map name per run, without seed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows_out = []
    for run, cls, st in zip(args.runs, args.classifier, args.state):
        z = dict(np.load(RUNS / run / 'responses.npz'))
        record = json.loads((RUNS / run / 'READERS.json').read_text())
        fams = json.loads((RUNS / run / 'requests.json').read_text())['families']
        req = list(z['request_names'])
        rows = [req.index(r) for r in fams['parts']]
        clean = {r: z[f'bio|none|{r}'][0] for r in record['readers']}
        effect = {r: (z[f'bio|source|logit_{r}'][rows] - clean[r]).ravel() for r in record['readers']}
        seeds = lambda base: sorted(n for n in record['summary'] if n.startswith(base + '_s'))
        for r in record['readers']:
            if r == 'original':
                continue
            err_c = np.mean([nrmse(z, n, r, rows, clean[r]) for n in seeds(cls)])
            err_s = np.mean([nrmse(z, n, r, rows, clean[r]) for n in seeds(st)])
            rows_out.append(dict(run=run, reader=r, effect_corr=float(np.corrcoef(effect[r], effect['original'])[0, 1]),
                                 classifier_fit=float(err_c), state_fit=float(err_s), log_ratio=float(np.log(err_c / err_s))))
    corr = np.array([r['effect_corr'] for r in rows_out])
    lr = np.array([r['log_ratio'] for r in rows_out])
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, rows=rows_out,
               spearman=float(spearmanr(np.abs(corr), lr).statistic), pearson=float(np.corrcoef(np.abs(corr), lr)[0, 1]))
    args.output.write_text(json.dumps(out, indent=1))
    for r in sorted(rows_out, key=lambda r: abs(r['effect_corr'])):
        print(f"{r['run'][-34:]:<34} {r['reader']:<28} corr {r['effect_corr']:+.3f}  classifier {r['classifier_fit']:.3f}  "
              f"state {r['state_fit']:.3f}  log ratio {r['log_ratio']:+.3f}")
    print(f"Spearman |corr| vs log ratio {out['spearman']:+.3f}, Pearson {out['pearson']:+.3f}, readers {len(rows_out)}")


if __name__ == '__main__':
    main()
