from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np


def main():
    bulk=Path('D:/CCAD_Storage/runs/final_science_20260920_round02')
    old=bulk/'AGREEMENT_CONFIRM_T1_20260920'
    replay=bulk/'AGREEMENT_CODE_EQUIVALENCE_T1_20260920'
    records=[]
    with np.load(old/'responses.npz') as a, np.load(replay/'responses.npz') as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            assert a[key].shape==b[key].shape
            records.append(dict(key=key,shape=list(a[key].shape),identical=bool(np.array_equal(a[key],b[key])),
                                maximum_difference=float(np.max(np.abs(a[key]-b[key])))))
    assert all(r['identical'] for r in records)
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),old_run=str(old),replay_run=str(replay),
                all_prediction_arrays_identical=True,arrays=records,
                code_identities={str(p):json.loads((p/'code_hashes.json').read_text()) for p in [old,replay]},
                prediction_file_sha256={str(p):hashlib.sha256((p/'responses.npz').read_bytes()).hexdigest() for p in [old,replay]})
    output=Path('artifacts/final_science_20260920_round02/CODE_EQUIVALENCE.json')
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(arrays=len(records),identical=True,output=str(output))))


if __name__=='__main__':
    main()
