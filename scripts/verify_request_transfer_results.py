from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from analyze_member_request_training import analyze
from analyze_program_generalization import read_run


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_science_20260921_round05'


def main():
    output=OUT/'NUMERICAL_REPLAY.json'
    assert not output.exists()
    checked=[]
    for setting in ['human','infinitive']:
        old=json.loads((OUT/f'{setting.upper()}_REQUEST_DEVELOPMENT.json').read_text())
        configs={m:[str(ROOT/f'configs/fs05_{setting}_{m}_dev{"_v2" if setting=="human" else ""}.json')] for m in ['semantic','balanced','independent']}
        result=analyze(setting,configs)
        for method,families in old['summary'].items():
            for family,values in families.items():
                for key in ['nrmse','interval','valid_draws']:
                    assert values[key]==result['summary'][method][family][key],(setting,method,family,key)
        assert result['per_query']==old['per_query']
        checked.append(setting)
    for kind in ['CROSS','GENERIC']:
        old=json.loads((OUT/f'{kind}_PROGRAM_GENERALIZATION.json').read_text())
        for value in old['records'].values():
            entries=[value] if kind=='CROSS' else list(value.values())
            for entry in entries:
                assert read_run(Path(entry['run']))==entry
                checked.append(entry['run'])
    for n in [4]:
        eigenvalues=np.linalg.eigvalsh(np.eye(n)/48+np.ones((n,n))*9/16)
        assert np.isclose(eigenvalues[-1]/eigenvalues[0],109.)
    output.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        status='PASS',checked=checked,condition_number=109.,
        scope='Recomputation from retained predictions and bootstrap definition. This uses the analysis implementation and is not an independent statistical audit.'),indent=2)+'\n')
    print(f'PASS {len(checked)} result collections and the request-conditioning calculation.')


if __name__=='__main__':
    main()
