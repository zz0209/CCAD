import argparse
import json
from pathlib import Path

import numpy as np

from summarize_depth_referee import later

DESCRIPTION = ('Dose-response of the unit test: CLT units are binned by the fraction of their own-layer write that '
               'removing their input takes out (the dose), and for each bin and read-layer group (layers 0-1, later) '
               'the median fraction of their claimed later writes that the same removal takes out, with quartiles and '
               'unit counts. A removal that reaches the unit but not its later writes shows as a flat early curve.')
BINS = [(-np.inf, .25), (.25, .5), (.5, .75), (.75, np.inf)]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True, help='name=DEPTH_REFEREE_ROWS.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = {}
    for spec in args.runs:
        name, path = spec.split('=')
        rows = [r for r in json.loads(Path(path).read_text())['rows']
                if len(r['claim_energy']) > 1 and sum(r['claim_energy'][1:]) > 0 and r['claim_energy'][0] > 0]
        d = {}
        for group, keep in [('early', lambda r: r['layer'] <= 1), ('later', lambda r: r['layer'] >= 2)]:
            sel = [r for r in rows if keep(r)]
            own = np.array([r['removed_u'][0] / r['claim_energy'][0] for r in sel])
            lat = np.array([later(r, 'removed_u') for r in sel])
            d[group] = []
            for lo, hi in BINS:
                m = (own >= lo) & (own < hi)
                if m.sum() == 0:
                    d[group].append(dict(lo=lo, hi=hi, units=0))
                    continue
                d[group].append(dict(lo=lo, hi=hi, units=int(m.sum()), own=float(np.median(own[m])),
                                     later=[float(np.median(lat[m])), float(np.quantile(lat[m], .25)),
                                            float(np.quantile(lat[m], .75))]))
        out[name] = d
        print(name, json.dumps({g: [(b['units'], round(b['later'][0], 3) if b['units'] else None) for b in v]
                                for g, v in d.items()}))
    args.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
