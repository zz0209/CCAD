from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round02'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='RG02_FIELD_REALIZATION_DEVELOPMENT_20260921')
    parser.add_argument('--output',type=Path,default=OUT/'FIELD_REALIZATION_DIAGNOSTIC.json')
    args=parser.parse_args()
    output=args.output
    if output.exists():raise FileExistsError(output)
    run=BULK/args.run_id
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    c=json.loads((run/'config.resolved.json').read_text())
    reference=Path(c['reference_run'])
    membership=json.loads((run/'membership.json').read_text())
    prior=json.loads((reference/'membership.json').read_text())
    assert membership['evaluation']==prior['evaluation'] and membership['queries']==prior['queries']
    rows=membership['evaluation'];queries=membership['queries']
    source=np.stack([np.load(reference/f'source__{q}__pooled.npy') for q in queries]).astype(float)
    clean=np.load(reference/'clean__full__pooled.npy').astype(float)
    families={family:[i for i,q in enumerate(queries) if (q=='full' if family=='full' else
        q.startswith(('interior','boundary')) if family=='participation' else q.startswith('member'))]
        for family in ['full','participation','members']}
    metrics={}
    for seed in c['seeds']:
        for mode in c.get('execution_modes',['physical','native']):
            target=np.stack([np.load(run/f'{mode}_t{seed}__{q}__pooled.npy') for q in queries]).astype(float)
            scores=[]
            for task,pair in [('composer_surgeon_orientation0',(5,25)),('composer_surgeon_orientation1',(5,25)),
                              ('model_software_engineer_orientation0',(12,24)),('model_software_engineer_orientation1',(12,24))]:
                h=np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{task}__probe42.npz')['weight'].ravel()
                ix=np.array([r['profession'] in pair for r in rows])
                error=((target[:,ix]-source[:,ix])@h)**2
                energy=((source[:,ix]-clean[ix])@h)**2
                scores.append(np.sqrt(error.mean(-1)/np.maximum(energy.mean(-1),1e-12)))
            values=np.mean(scores,axis=0)
            metrics[f't{seed}/{mode}']={family:float(values[ix].mean()) for family,ix in families.items()}
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=str(run),metrics=metrics,
        evidence='Post-confirmation development diagnostic on32previously exposed biographies. Only embedding realization differs; the later native trajectory uses the same target dictionary.',
        replay=json.loads((run/'native_replay.json').read_text()),
        scope='Physical writing is an unrestricted diagnostic. Both variants use the same target2learned field; the target3native arm is checked against retained predictions.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
