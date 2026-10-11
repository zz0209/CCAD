from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIRROR = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010/mayo_mirror')
DESCRIPTION = ('Fraction of trained one-to-one gains that sit at their upper bound in the Gemma matched-gain runs, to '
               'decide whether the bound restricts the one-to-one control.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/naacl_sprint_20261010/e4_scale/GAIN_CLAMP.json')
    args = parser.parse_args()
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, runs={})
    for run in args.runs:
        cfg = json.loads((MIRROR / run / 'config.resolved.json').read_text())
        cap = float(cfg.get('matched_gain_max', 1.))
        maps = json.loads((MIRROR / run / 'maps.json').read_text())
        rows = {}
        for name, sites in maps.items():
            if 'matched' not in name:
                continue
            # Weights are stored source by target; each source feature keeps one nonzero gain on its match.
            gains = np.concatenate([np.asarray(s['weights'], dtype=float).max(1) for s in sites.values()])
            rows[name] = dict(features=int(gains.size), at_cap=float(np.mean(gains >= cap - 1e-3)),
                              at_zero=float(np.mean(gains <= 1e-3)), median=float(np.median(gains)),
                              quartiles=np.quantile(gains, [.25, .75]).tolist())
        at_cap = [r['at_cap'] for r in rows.values()]
        out['runs'][run] = dict(cap=cap, maps=rows, mean_at_cap=float(np.mean(at_cap)))
        print(run, cap, round(float(np.mean(at_cap)), 3), {k: round(v['median'], 3) for k, v in rows.items()})
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
