from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BULK = Path('D:/CCAD_Storage/runs/final_science_20260921_round05')
OUT = ROOT/'artifacts/final_science_20260921_round05/TRAINING_TRACE_ANALYSIS.json'


def main():
    assert not OUT.exists()
    records = []
    source_energy = []
    bank_hashes = []
    for objective in ['local', 'downstream', 'distribution']:
        run = BULK/f'FS05_GENERIC_{objective.upper()}_DEV_20260921'
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        rows = [json.loads(line) for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
        fit = [r for r in rows if r['kind'] == 'training']
        assert [r['row_id'] for r in fit] == list(range(32, 513, 32))
        source_energy.append(np.array([r['energy'] for r in fit]))
        bank_hashes.append(hashlib.sha256((run/'bank_resid_4.json').read_bytes()).hexdigest())
        quality = [r for r in json.loads((run/'quality.json').read_text()) if r['site'] == 'resid_4']
        records.append(dict(objective=objective, run=str(run), sampled_steps=fit,
            first_four_program_mean=float(np.mean([r['program'] for r in fit[:4]])),
            last_four_program_mean=float(np.mean([r['program'] for r in fit[-4:]])),
            quality=quality))
    assert len(set(bank_hashes)) == 1
    assert all(np.array_equal(source_energy[0], x) for x in source_energy[1:])
    report = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Existing development traces, one target. Different objectives have different loss units. Early and late steps use different actions, so their averages are not a fixed-panel learning curve.',
        matched_bank=True, identical_logged_source_energies=True,
        source_energy_quantiles=dict(zip(['min','q25','median','q75','max'], np.quantile(source_energy[0],[0,.25,.5,.75,1]).tolist())),
        records=records)
    OUT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))
    for r in records:
        print(json.dumps({k:v for k,v in r.items() if k!='sampled_steps'}))


if __name__ == '__main__':
    main()
