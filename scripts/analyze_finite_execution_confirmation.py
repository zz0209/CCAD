from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round02'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def main():
    output=OUT/'CONFIRMATION_ANALYSIS.json'
    if output.exists():raise FileExistsError(output)
    freeze=json.loads((OUT/'FINITE_CONFIRMATION_FREEZE.json').read_text())
    for path,digest in freeze['identities'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    runs={seed:BULK/f'RG02_FINITE_CONFIRMATION_T{seed}_20260921' for seed in [2,4,5]}
    studies={seed:json.loads((run/'analysis.json').read_text()) for seed,run in runs.items()}
    membership=json.loads((runs[2]/'membership.json').read_text())
    rows=membership['evaluation'];queries=membership['queries']
    source=np.stack([np.load(runs[2]/f'source__{q}__pooled.npy') for q in queries]).astype(float)
    clean=np.load(runs[2]/'clean__full__pooled.npy').astype(float)
    for seed,run in runs.items():
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        m=json.loads((run/'membership.json').read_text())
        assert m['evaluation']==rows and m['queries']==queries
        assert all(np.array_equal(source[i],np.load(run/f'source__{q}__pooled.npy')) for i,q in enumerate(queries))
    head_specs=[('composer_surgeon_orientation0',(5,25)),('composer_surgeon_orientation1',(5,25)),
                ('model_software_engineer_orientation0',(12,24)),('model_software_engineer_orientation1',(12,24))]
    heads=[np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{task}__probe42.npz')['weight'].ravel() for task,_ in head_specs]
    cohorts=[np.array([r['profession'] in pair for r in rows]) for _,pair in head_specs]
    energy=np.stack([((source-clean)@h)**2 for h in heads])
    errors={}
    for seed,study in studies.items():
        for method in study['metrics']:
            target=np.stack([np.load(runs[seed]/f'{method}__{q}__pooled.npy') for q in queries]).astype(float)
            errors[seed,method]=np.stack([((target-source)@h)**2 for h in heads])
    parts=np.array([i for i,q in enumerate(queries) if q.startswith(('interior','boundary'))])
    cells=[np.array([i for i,r in enumerate(rows) if (r['profession'],r['gender'])==cell])
           for cell in sorted({(r['profession'],r['gender']) for r in rows})]
    def score(error,docs,requests):
        values=[]
        for h,cohort in enumerate(cohorts):
            take=docs[cohort[docs]]
            mse=error[h][np.ix_(requests,take)].mean(-1)
            var=energy[h][np.ix_(requests,take)].mean(-1)
            if not (var>1e-12).all():raise ValueError('Resample lacks source-response support')
            values.append(np.sqrt(mse/var).mean())
        return float(np.mean(values))
    points={key:score(value,np.arange(len(rows)),parts) for key,value in errors.items()}
    for (seed,method),value in points.items():assert abs(value-studies[seed]['metrics'][method]['later']['participation'])<1e-12
    rng=np.random.default_rng(921025)
    samples={key:[] for key in errors}
    for _ in range(2000):
        docs=np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in cells])
        requests=rng.choice(parts,len(parts),replace=True)
        for key,value in errors.items():samples[key].append(score(value,docs,requests))
    contrasts={}
    groups={'fitted_target2':[2],'held_targets4_5':[4,5]}
    for name,seeds in groups.items():
        selected='saved_sparse' if seeds==[2] else 'transfer_sparse'
        controls=['geometric','saved_fixed','program','readout'] if seeds==[2] else ['geometric','transfer_gain','program','readout']
        for control in controls:
            delta=np.mean([np.asarray(samples[s,control])-samples[s,selected] for s in seeds],axis=0)
            contrasts[name+'_'+control+'_minus_'+selected]=dict(
                reduction=float(np.mean([points[s,control]-points[s,selected] for s in seeds])),
                interval95=np.quantile(delta,[.025,.975]).tolist())
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),freeze=str(OUT/'FINITE_CONFIRMATION_FREEZE.json'),
        checks=dict(freeze_hashes_equal=True,source_predictions_equal=True,metric_replay_equal=True),
        documents=len(rows),participation_requests=len(parts),studies=studies,
        points={f't{s}/{m}':v for (s,m),v in points.items()},contrasts=contrasts,
        inference='2000paired profession/gender-stratified biography and participation-request resamples. Target2, held targets4/5, the source explanation and four heads are fixed. No seed-population interval is claimed.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(points=result['points'],contrasts=contrasts),indent=2))


if __name__=='__main__':
    main()
