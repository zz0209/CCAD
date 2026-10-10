from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

MIRROR = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010/mayo_mirror')
ART = Path(__file__).resolve().parents[1] / 'artifacts' / 'naacl_sprint_20261010'
DESCRIPTION = ('How often the fixed Gemma correspondence clips a deletion fraction at one, that is how often min(1, Aq) '
               'is active, for every evaluated request, using the saved memberships and request masks.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', default=['NAACL_E4C_GEMMA2B_final_w65k_v2_20261010',
                                                      'NAACL_E4C_GEMMA2B_final_l0low16k_v2_20261010'])
    parser.add_argument('--output', type=Path, default=ART / 'e4_scale' / 'MIN_ACTIVE.json')
    args = parser.parse_args()
    out = {}
    for name in args.runs:
        run = MIRROR / name
        maps = json.loads((run / 'maps.json').read_text())
        req = json.loads((run / 'requests.json').read_text())
        sites = [f['site'] for f in req['features']]
        masks = {k: np.array(v, dtype=float) for k, v in req['masks'].items()}
        res = {}
        for method, per_site in maps.items():
            if '_sup_' not in method:
                continue
            clipped = members = 0
            by_family = {}
            for family, names in req['families'].items():
                c = m = 0
                for request in names:
                    q = masks[request]
                    for site, entry in per_site.items():
                        cols = [i for i, s in enumerate(sites) if s == site]
                        # Saved weights are source features by target features, so the load on each target is q A^T.
                        w = np.array(entry['weights'])
                        load = q[cols] @ w
                        active = load > 0
                        c += int((load[active] > 1).sum())
                        m += int(active.sum())
                by_family[family] = dict(clipped=c, members=m, share=c / max(m, 1))
                clipped += c
                members += m
            res[method] = dict(share=clipped / max(members, 1), by_family=by_family)
        out[name] = res
        for method, r in res.items():
            print(name[-28:], f'{method:<32}', f"{r['share']:.4f}",
                  {f: round(v['share'], 4) for f, v in r['by_family'].items()})
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
                                           runs=out), indent=1))


if __name__ == '__main__':
    main()
