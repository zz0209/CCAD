import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTION = ('Summary of read-write agreement rows: by the earliest layer of each unit, the median answer drops when '
               'its writes are deleted and when its reads are removed, the median ratio of read to write effect size on '
               'the next-token distribution, the nRMSE between the two effects, the share of units whose write deletion '
               'moves the answer by more than 1 nat while read removal moves it by less than a quarter of that, and '
               'agreement of the new top token among units whose write deletion changes the answer.')


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows):
    dw = np.array([r['drop_write'] for r in rows])
    dr = np.array([r['drop_read'] for r in rows])
    ratio = np.sqrt(np.array([r['energy_read'] / max(r['energy_write'], 1e-12) for r in rows]))
    rvw = np.array([r['read_vs_write'] for r in rows])
    big = dw > 1.
    changed = [r for r in rows if r['top_write'] != r['top_base']]
    # Write-to-read effect size ratio among units whose write deletion has a substantial effect (size above .3).
    strong = [r for r in rows if r['energy_write'] ** .5 > .3]
    log_ratio = np.log10(np.array([(r['energy_write'] / max(r['energy_read'], 1e-12)) ** .5 for r in strong])) if strong else None
    return dict(units=len(rows), strong=len(strong),
                strong_log10_ratio=float(np.median(log_ratio)) if strong else None,
                strong_ratio_over_10=float(np.mean(log_ratio > 1)) if strong else None,
                drop_write=float(np.median(dw)), drop_read=float(np.median(dr)),
                size_ratio=float(np.median(ratio)), read_vs_write=float(np.median(rvw)),
                big_write=int(big.sum()), big_write_small_read=float(np.mean(dr[big] < .25 * dw[big])) if big.any() else None,
                spearman_drops=spearman(dw, dr) if len(rows) > 3 else None,
                changed=len(changed), same_new=float(np.mean([r['top_read'] == r['top_write'] for r in changed])) if changed else None)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('rows', type=Path, nargs='+')
    parser.add_argument('--min-energy', type=float, default=1e-3)
    args = parser.parse_args()
    for path in args.rows:
        rows = [r for r in json.loads(path.read_text())['rows'] if r['energy_write'] > args.min_energy]
        print(path.parent.name)
        for name, sel in [('all', rows), ('layers 0-1', [r for r in rows if min(r['layers']) <= 1]),
                          ('layers 2-12', [r for r in rows if 2 <= min(r['layers']) <= 12]),
                          ('layers 13+', [r for r in rows if min(r['layers']) >= 13])]:
            if sel:
                s = summarize(sel)
                print(f"  {name:12s} " + ' '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in s.items()))


if __name__ == '__main__':
    main()
