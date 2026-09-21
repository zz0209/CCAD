from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/final_science_20260921_round05')


def main():
    output=ROOT/'artifacts/final_science_20260921_round05/PROGRAM_ACTION_AUDIT.json'
    assert not output.exists()
    records=[]
    for objective in ['local','downstream','distribution']:
        run=BULK/f'FS05_GENERIC_{objective.upper()}_AUDIT_20260921'
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        original=BULK/f'FS05_GENERIC_{objective.upper()}_DEV_20260921'
        old=np.load(original/'responses.npz'); new=np.load(run/'responses.npz')
        assert set(old.files)==set(new.files)
        assert all(np.array_equal(old[k],new[k]) for k in old.files)
        rows=json.loads((run/'program_audit.json').read_text())['rows']
        for context in ['fit','held_context']:
            initial=[r for r in rows if r['context']==context and r['method']=='initial']
            trained=[r for r in rows if r['context']==context and r['method']=='trained']
            assert len(initial)==len(trained)==16
            for a,b in zip(initial,trained):
                for key in ['item','sequences','bank_indices','q','affected_token_fraction','source_hidden_energy','source_distribution_energy']:
                    assert a[key]==b[key], key
            record=dict(objective=objective,context=context,run=str(run),items=16,
                unique_sequences=len({v for r in initial for v in r['sequences']}),
                affected_token_fraction=float(np.mean([r['affected_token_fraction'] for r in initial])),
                exact_original_predictions=True)
            for metric in ['local_relative_mse','downstream_relative_mse','distribution_relative_kl']:
                a=np.array([r[metric] for r in initial]); b=np.array([r[metric] for r in trained])
                record[metric]=dict(initial=float(a.mean()),trained=float(b.mean()),
                    improved_items=int((b<a).sum()),paired_differences=(b-a).tolist())
            records.append(record)
    output.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Development diagnostic,16dependent actions per context population, one target. No population confidence interval. Original functional predictions are bitwise unchanged.',records=records),indent=2)+'\n')
    for r in records:
        print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='paired_differences'} if isinstance(v,dict) else v) for k,v in r.items()}))


if __name__=='__main__':
    main()
