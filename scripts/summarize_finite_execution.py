from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round02'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def main():
    output=OUT/'RESULTS_AND_CHECKS.json'
    if output.exists():raise FileExistsError(output)
    runs={name:BULK/f'RG02_{suffix}_20260921' for name,suffix in {
        'fixed_scaled':'FINITE_EMBEDDING_DEVELOPMENT_T2',
        'fixed_direct':'FIXED_DIRECT_DEVELOPMENT_T2',
        'sparse':'FINITE_SUPPORT_DEVELOPMENT_T2',
        'state':'FINITE_STATE_DEVELOPMENT_T2',
        'regrow':'REGROW_DEVELOPMENT_T2',
        'transfer':'FINITE_FIELD_TRANSFER_T3',
        'transfer_sparse':'SPARSE_FIELD_TRANSFER_T3'}.items()}
    studies={name:json.loads((run/'analysis.json').read_text()) for name,run in runs.items()}
    summaries={name:json.loads((run/'metrics.summary.json').read_text()) for name,run in runs.items()}
    assert all(s['status']=='PASS' and all(s['checks'].values()) for s in summaries.values())
    ref=runs['fixed_scaled']
    membership=json.loads((ref/'membership.json').read_text())
    queries=membership['queries'];rows=membership['evaluation']
    checks={}
    for name,run in runs.items():
        m=json.loads((run/'membership.json').read_text())
        assert m['evaluation']==rows and m['queries']==queries
        assert np.array_equal(np.load(run/'clean__full__pooled.npy'),np.load(ref/'clean__full__pooled.npy'))
        assert all(np.array_equal(np.load(run/f'source__{q}__pooled.npy'),np.load(ref/f'source__{q}__pooled.npy')) for q in queries)
        checks[name]=dict(source_arrays_equal=True,checks=summaries[name]['checks'],
            config_sha256=hashlib.sha256((run/'config.resolved.json').read_bytes()).hexdigest(),
            analysis_sha256=hashlib.sha256((run/'analysis.json').read_bytes()).hexdigest())
    checks['matched_64_steps']=all(np.array_equal(np.load(runs['fixed_direct']/f'fixed_direct_64__{q}__pooled.npy'),
                       np.load(runs['sparse']/f'sparse_64__{q}__pooled.npy')) for q in queries)
    assert checks['matched_64_steps']
    records=[]
    for name,study in studies.items():
        for method,metrics in study['metrics'].items():
            for endpoint in ['original','later','representation']:
                for family,value in metrics[endpoint].items():
                    records.append(dict(run=name,method=method,endpoint=endpoint,family=family,nrmse=value))
    selected=[q for q in queries if q.startswith(('interior','boundary'))]
    source=np.stack([np.load(ref/f'source__{q}__pooled.npy') for q in selected]).astype(float)
    clean=np.load(ref/'clean__full__pooled.npy').astype(float)
    cells=[np.array([i for i,r in enumerate(rows) if (r['profession'],r['gender'])==cell])
           for cell in sorted({(r['profession'],r['gender']) for r in rows})]
    heads=[]
    for task,pair in [('composer_surgeon_orientation0',(5,25)),('composer_surgeon_orientation1',(5,25)),
                      ('model_software_engineer_orientation0',(12,24)),('model_software_engineer_orientation1',(12,24))]:
        head=np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{task}__probe42.npz')['weight'].ravel()
        cohort=np.array([r['profession'] in pair for r in rows])
        heads.append((head,cohort))
    choices={'geometric':('fixed_scaled','geometric'),'gain':('fixed_scaled','gain_256'),
             'fixed':('fixed_direct','fixed_direct_256'),'sparse':('sparse','sparse_256'),
             'state_fixed':('state','fixed_direct_256'),'state_sparse':('state','sparse_256'),
             'regrow':('regrow','regrow_256'),
             'program':('sparse','program'),'readout':('fixed_scaled','readout')}
    errors={}
    energies=[]
    for head,_ in heads:energies.append(((source-clean)@head)**2)
    for name,(run_name,method) in choices.items():
        target=np.stack([np.load(runs[run_name]/f'{method}__{q}__pooled.npy') for q in selected]).astype(float)
        errors[name]=[((target-source)@head)**2 for head,_ in heads]
    def score(error,ix):
        values=[]
        for e,v,(_,cohort) in zip(error,energies,heads):
            take=ix[cohort[ix]]
            values.append(np.sqrt(e[:,take].mean(-1)/np.maximum(v[:,take].mean(-1),1e-12)).mean())
        return float(np.mean(values))
    points={name:score(e,np.arange(len(rows))) for name,e in errors.items()}
    for name,(run_name,method) in choices.items():
        assert abs(points[name]-studies[run_name]['metrics'][method]['later']['participation'])<1e-12
    rng=np.random.default_rng(921022)
    samples={name:[] for name in choices}
    for _ in range(2000):
        ix=np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in cells])
        for name,error in errors.items():samples[name].append(score(error,ix))
    contrasts={}
    for a,b in [('geometric','sparse'),('fixed','sparse'),('geometric','state_sparse'),('sparse','state_sparse'),('sparse','regrow')]:
        delta=np.asarray(samples[a])-samples[b]
        contrasts[a+'_minus_'+b]=dict(reduction=points[a]-points[b],interval95=np.quantile(delta,[.025,.975]).tolist())
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence='Exposed development biographies and requests; target2 fitting, target3 field transfer. All classifiers and dictionaries are fixed in resampling.',
        studies=studies,checks=checks,later_participation=points,development_contrasts=contrasts,
        uncertainty='2000 paired profession/gender-stratified biography resamples, fixed24participation requests, fixed4later classifiers. Conditional development uncertainty, not independent confirmation.',
        support_changes=json.loads((runs['sparse']/'support_changes.json').read_text()),records=records)
    output.write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'paper/data/finite_execution_development.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(points=points,contrasts=contrasts,checks=checks),indent=2))


if __name__=='__main__':
    main()
