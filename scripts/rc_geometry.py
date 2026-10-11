from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'containment'
READBACK = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010/mayo_mirror/NAACL_E10_GEMMA2B_readback_v1_20261010/READBACK.json')
DESCRIPTION = ('Advantage of the many-to-many state fit over its one-to-one control, and over Hungarian matching, across '
               'Gemma Scope targets, against the mean decoder cosine of the Hungarian matches of the explained features. '
               'All errors are on test biographies, part requests, from the reader analyses of the final and confirmation runs.')
# Reader-analysis groups for each source: (state fit, one-to-one state fit, Hungarian matching).
GROUPS = {'final': ('lw_subset_state_sup_hungarian', 'ext_matched_subset_state_sup_hungarian', 'hungarian_decoder'),
          'conf': ('conf_subset_state_sup_hungarian', 'conf_matched_subset_state_sup_hungarian', 'e10_hungarian_decoder')}
TARGET_OF = {'w65k': 'w65k_sfc', 'l0low16k': 'w16k_lowest'}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--final', type=Path, required=True, help='reader analysis of the final test runs')
    parser.add_argument('--conf', type=Path, required=True, help='reader analysis of the confirmation runs')
    parser.add_argument('--output', type=Path, default=ART / 'GEMMA_GEOMETRY.json')
    args = parser.parse_args()
    cos = {k: v['hungarian_cos_mean'] for k, v in json.loads(READBACK.read_text()).items()}
    rows = []
    for kind, path in [('final', args.final), ('conf', args.conf)]:
        rec = json.loads(path.read_text())
        state, matched, hungarian = GROUPS[kind]
        for run, r in rec['runs'].items():
            key = run.split('_readers_')[1].rsplit('_v1_', 1)[0]
            target = TARGET_OF.get(key, key)
            pt = r['point']
            rows.append(dict(target=target, source=kind, hungarian_cos=cos[target],
                             **{f'{reader}_{name}': pt[g][f'parts|{reader}'] for reader in ['original', 'E2']
                                for name, g in [('state', state), ('matched', matched), ('hungarian', hungarian)]}))
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, rows=rows, stats={})
    c = np.array([r['hungarian_cos'] for r in rows])
    for reader in ['original', 'E2']:
        for other in ['matched', 'hungarian']:
            ratio = np.array([r[f'{reader}_{other}'] / r[f'{reader}_state'] for r in rows])
            out['stats'][f'{reader}_{other}_over_state'] = dict(
                ratios=ratio.tolist(), state_lower=int((ratio > 1).sum()), targets=len(rows),
                geometric_mean=float(np.exp(np.log(ratio).mean())), spearman_with_cos=float(spearmanr(c, np.log(ratio)).statistic))
    args.output.write_text(json.dumps(out, indent=1))
    for r in sorted(rows, key=lambda r: r['hungarian_cos']):
        print(f"{r['target']:<14} cos {r['hungarian_cos']:.3f}  E2 state {r['E2_state']:.3f} matched {r['E2_matched']:.3f} "
              f"hungarian {r['E2_hungarian']:.3f}  original state {r['original_state']:.3f} matched {r['original_matched']:.3f}")
    for k, v in out['stats'].items():
        print(f"{k:<30} state lower {v['state_lower']}/{v['targets']}  geo mean {v['geometric_mean']:.3f}  "
              f"spearman with cos {v['spearman_with_cos']:+.2f}")


if __name__ == '__main__':
    main()
