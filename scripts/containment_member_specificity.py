from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
DESCRIPTION = ('Which target features carry each part: part purity of the target features used by each correspondence, '
               'and agreement between the carriers chosen by the task-free natural-text map and by part supervision, '
               'against the agreement between different parts.')
DEFAULT = ['CSM_X1_gpt2_s1_t2_v1_20261010', 'CSM_X1_gpt2_s1_t3_v1_20261010', 'CSM_X1_gpt2_s1_t4_v1_20261010',
           'CSM_X1_gpt2_s1_t5_v1_20261010', 'CSM_X1C_gpt2_s2_t3_v2_20261010', 'CSM_X1C_gpt2_s3_t4_v2_20261010',
           'CSM_X1C_gpt2_s4_t5_v2_20261010', 'CSM_X1C_gpt2_s5_t1_v2_20261010']
MAPS = ['hungarian_activation', 'matched_natural', 'membership_natural', 'membership_parts', 'membership_complete']
THRESHOLD = .05


def part_loads(weights, parts):
    # Total deletion weight that each target candidate receives from the members of each part.
    return np.stack([weights[:, parts == k].sum(1) for k in range(int(parts.max()) + 1)], 1)


def summary(loads):
    total = loads.sum(1)
    used = total > THRESHOLD
    purity = loads[used].max(1) / total[used]
    return dict(target_features=int(used.sum()), weighted_purity=float((purity * total[used]).sum() / total[used].sum()),
                features_per_part=[int((loads[:, k] > THRESHOLD).sum()) for k in range(loads.shape[1])])


def cosine(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', default=DEFAULT)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/containment/GPT2_MEMBER_SPECIFICITY.json')
    args = parser.parse_args()
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, threshold=THRESHOLD, runs={})
    same, cross = [], []
    for run in args.runs:
        z = np.load(RUNS / run / 'maps.npz')
        parts = z['parts']
        loads = {m: part_loads(z[m], parts) for m in MAPS if m in z.files}
        rec = {m: summary(v) for m, v in loads.items()}
        # Cosine between the per-part load vectors of the natural-text map and of part supervision, for the same part
        # and for different parts.
        a, b = loads['membership_natural'], loads['membership_parts']
        k = a.shape[1]
        agree = [[cosine(a[:, i], b[:, j]) for j in range(k)] for i in range(k)]
        rec['natural_vs_parts_load_cosine'] = agree
        # Both maps start from the Hungarian matches; their agreement with that starting point is the reference.
        h = loads['hungarian_activation']
        rec['load_cosine_with_hungarian'] = {m: [cosine(loads[m][:, i], h[:, i]) for i in range(k)]
                                             for m in ('membership_natural', 'membership_parts')}
        rec['natural_vs_parts_without_hungarian'] = [cosine(a[:, i]*(h[:, i] == 0), b[:, i]*(h[:, i] == 0)) for i in range(k)]
        same += [agree[i][i] for i in range(k)]
        cross += [agree[i][j] for i in range(k) for j in range(k) if i != j]
        out['runs'][run] = rec
        print(run, {m: (r['target_features'], round(r['weighted_purity'], 3)) for m, r in rec.items() if m in MAPS},
              'same part', np.round(np.diag(agree), 3).tolist())
    out['natural_vs_parts'] = dict(
        same_part_mean=float(np.mean(same)), different_part_mean=float(np.mean(cross)), same_part_min=float(np.min(same)),
        without_hungarian_mean=float(np.mean([v for r in out['runs'].values() for v in r['natural_vs_parts_without_hungarian']])),
        with_hungarian_mean={m: float(np.mean([v for r in out['runs'].values() for v in r['load_cosine_with_hungarian'][m]]))
                             for m in ('membership_natural', 'membership_parts')})
    for m in MAPS:
        vals = [r[m]['weighted_purity'] for r in out['runs'].values() if m in r]
        out.setdefault('mean_purity', {})[m] = float(np.mean(vals))
    print(out['natural_vs_parts'], out['mean_purity'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
