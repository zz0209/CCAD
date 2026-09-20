import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for source in range(1, 6):
        cfg = json.loads((ROOT/f'configs/final_science03_qwen_confirm_s{source}.json').read_text())
        spec = cfg['source_field_evaluation']
        target = spec['target_seeds'][0]
        reader_path = ROOT/spec['readout_run']/f'readout_s{source}_t{target}.npz'
        relation_path = ROOT/spec['relation_run']/f'relation_s{source}_t{target}.npz'
        with np.load(reader_path) as reader, np.load(relation_path) as relation:
            operations = []
            for operation in ['unit', 'tens']:
                si = reader[f'{operation}_source_indices']
                ti = relation[f'{operation}_target_indices']
                assert len(si) == 64 and len(ti) == 64
                assert np.array_equal(si, relation[f'{operation}_source_indices'])
                coefficients = reader[f'{operation}_full_activation']
                outside = np.ones(coefficients.shape[1], dtype=bool)
                outside[ti] = False
                own = reader[f'{operation}_target_indices']
                own_outside = np.ones(coefficients.shape[1], dtype=bool)
                own_outside[own] = False
                assert np.count_nonzero(coefficients[:, own_outside]) == 0
                assert np.isfinite(coefficients).all()
                operations.append(dict(operation=operation, source_members=len(si), target_members=len(ti),
                    discarded_readout_nonzero_coefficients=int(np.count_nonzero(coefficients[:, outside])),
                    shared_members=len(set(ti.tolist()) & set(own.tolist())),
                    target_bank_identical=set(ti.tolist()) == set(own.tolist()),
                    own_bank_contains_all_fitted_coefficients=True))
        records.append(dict(source=source, target=target, operations=operations,
            reader=dict(path=reader_path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(reader_path.read_bytes()).hexdigest()),
            relation=dict(path=relation_path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(relation_path.read_bytes()).hexdigest())))
    output = ROOT/'artifacts/final_science_20260920_round03/CONTROL_ASSET_CHECK.json'
    assert not output.exists()
    output.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(), records=records,
        scope='The recorded member intersections diagnose a readout-bank mismatch. Correct readout execution must select its own saved target indices. Primary profile and Euclidean outputs are unaffected. No confirmation predictions were read.'), indent=2)+'\n')
    print(json.dumps(dict(directions=len(records), operations=2*len(records),
        shared_members=[[o['shared_members'] for o in r['operations']] for r in records],
        status='READOUT_CORRECTION_REQUIRED')))


if __name__ == '__main__':
    main()
