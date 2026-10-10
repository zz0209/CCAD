from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'naacl_sprint_20261010' / 'e4_scale'
DESCRIPTION = ('Association between source-feature retention and the part error of re-applying the source edit across '
               'the 16 public SAEBench dictionaries, within each width, after regressing out width and log k, and over the '
               'eight configurations with the two training runs averaged.')


def partial(x, y, covariates):
    design = np.column_stack([np.ones(len(x))] + covariates)
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path, default=ART / 'RETENTION_PARTIAL.json')
    args = parser.parse_args()
    targets = json.loads((ART / 'TARGETS.json').read_text())['targets']
    fig = json.loads((ART / 'PUBLIC_FIGURE_DATA.json').read_text())
    seeds = sorted(targets, key=int)
    width = np.array([targets[s]['dict_size'] for s in seeds], dtype=float)
    k = np.array([targets[s]['k'] for s in seeds], dtype=float)
    x = np.array(fig['retained'])
    out = {}
    for key in ['raw_reconstruction', 'tangent_mixed', 'tangent_full']:
        y = np.array(fig['part_nrmse'][key])
        within = {str(int(w)): float(np.corrcoef(x[width == w], y[width == w])[0, 1]) for w in np.unique(width)}
        configs = sorted({(a, b) for a, b in zip(width, k)})
        cx = np.array([x[(width == a) & (k == b)].mean() for a, b in configs])
        cy = np.array([y[(width == a) & (k == b)].mean() for a, b in configs])
        out[key] = dict(pearson=float(np.corrcoef(x, y)[0, 1]), within_width=within,
                        partial_width_logk=partial(x, y, [width, np.log(k)]),
                        configurations=float(np.corrcoef(cx, cy)[0, 1]), n_configurations=len(configs))
        print(key, json.dumps(out[key]))
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), results=out), indent=1))


if __name__ == '__main__':
    main()
