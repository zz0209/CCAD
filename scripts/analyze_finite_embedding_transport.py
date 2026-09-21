from pathlib import Path
import argparse
import json

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    run=args.run
    if (run/'analysis.json').exists():raise FileExistsError(run/'analysis.json')
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    c=json.loads((run/'config.resolved.json').read_text())
    membership=json.loads((run/'membership.json').read_text())
    rows=membership['evaluation']; names=membership['queries']
    source=np.stack([np.load(run/f'source__{q}__pooled.npy') for q in names]).astype(float)
    clean=np.load(run/'clean__full__pooled.npy').astype(float)
    original=np.load(Path(c['frozen_source_run'])/'probe.npz')
    heads=[original['weight'].ravel()]
    cohorts=[np.ones(len(rows),bool)]
    tasks=['original','composer_surgeon_orientation0','composer_surgeon_orientation1',
           'model_software_engineer_orientation0','model_software_engineer_orientation1']
    for task,pair in zip(tasks[1:],[(5,25),(5,25),(12,24),(12,24)]):
        head=np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{task}__probe42.npz')
        heads.append(head['weight'].ravel())
        cohorts.append(np.array([r['profession'] in pair for r in rows]))
    families=dict(full=[i for i,q in enumerate(names) if q=='full'],
        endpoints=[i for i,q in enumerate(names) if not q.startswith(('interior','boundary','member')) and q!='full'],
        participation=[i for i,q in enumerate(names) if q.startswith(('interior','boundary'))],
        members=[i for i,q in enumerate(names) if q.startswith('member')])
    methods=c['baseline_methods']+list(c.get('transfer_fields',{}))+list(c.get('frozen_coefficients',{}))+[f'{arm}_{step}' for arm in c['arms'] for step in c['checkpoints']]
    result={}
    arrays={}
    for method in methods:
        target=np.stack([np.load(run/f'{method}__{q}__pooled.npy') for q in names]).astype(float)
        arrays[method]=target
        scores={}
        for task,head,cohort in zip(tasks,heads,cohorts):
            prediction_error=((target[:,cohort]-source[:,cohort])@head)**2
            source_energy=((source[:,cohort]-clean[cohort])@head)**2
            per_query=np.sqrt(prediction_error.mean(-1)/np.maximum(source_energy.mean(-1),1e-12))
            scores[task]={family:float(per_query[ix].mean()) for family,ix in families.items() if ix}
        result[method]=dict(original=scores['original'],later={family:float(np.mean([scores[t][family] for t in tasks[1:]]))
                           for family,ix in families.items() if ix},by_head=scores,
                           representation={family:float(np.sqrt(((target-source)**2).mean((1,2))/np.maximum(((source-clean)**2).mean((1,2)),1e-12))[ix].mean())
                                           for family,ix in families.items() if ix})
    value=dict(evidence=c.get('evaluation_description','Exposed development contexts; fixed dictionary and original explanation'),run=str(run),
               documents=len(rows),queries=names,families=families,metrics=result)
    (run/'analysis.json').write_text(json.dumps(value,indent=2)+'\n')
    for method,value in result.items():print(json.dumps(dict(method=method,original=value['original'],later=value['later'])))


if __name__=='__main__':
    main()
