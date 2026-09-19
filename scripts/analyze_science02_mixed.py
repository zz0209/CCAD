"""Evaluate mixed request training on the now-exposed development panel."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
BASE=Path('D:/CCAD_Storage/runs/science_upgrade_20260919')
OLD=BASE/'SCIENCE02_coverage_tasks_v1_20260919'
NEW=BASE/'SCIENCE02_mixed_training_v1_20260919'

def main():
    dest=OUT/'ROUND02_MIXED_ANALYSIS.json'
    if dest.exists():raise FileExistsError(dest)
    assert json.loads((NEW/'status.json').read_text())['status']=='PASS'
    assert json.loads((OLD/'evaluation_membership.json').read_text())==json.loads((NEW/'evaluation_membership.json').read_text())
    assert json.loads((OLD/'program_context_membership.json').read_text())==json.loads((NEW/'program_context_membership.json').read_text())
    panel=json.loads((OUT/'ROUND02_REQUESTS.json').read_text())
    families={'vertices':list(panel['calibration'])[:7],'interior':list(panel['evaluation'])[:12],'boundary':list(panel['evaluation'])[12:]}
    methods=['head_parts','pooled_parts','head_continuous','pooled_continuous','head_mixed','pooled_mixed','raw_reconstruction']
    rows=json.loads((NEW/'evaluation_membership.json').read_text())['rows']
    tasks=['composer_surgeon_orientation0','composer_surgeon_orientation1','model_software_engineer_orientation0','model_software_engineer_orientation1']
    weights=np.stack([np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz')['weight'].ravel() for t in tasks],1)
    clean=np.load(OLD/'none__full__pooled.npy').astype(float)
    assert np.allclose(clean,np.load(NEW/'none__full__pooled.npy'),atol=2e-5,rtol=1e-5)
    for q in sum(families.values(),[]):
        assert np.allclose(np.load(OLD/f'source__{q}__pooled.npy'),np.load(NEW/f'source__{q}__pooled.npy'),atol=2e-5,rtol=1e-5)
    rng=np.random.default_rng(2026091908);reps=1000;db=[]
    for profs in [(5,25),(12,24)]:
        rr=[r for r in rows if r['profession'] in profs]
        strata=[np.array([i for i,r in enumerate(rr) if r['profession']==p and r['gender']==g]) for p in profs for g in [0,1]]
        db.append(np.concatenate([rng.choice(s,(reps,len(s)),replace=True) for s in strata],1))
    summary=[];samples={};inputs={}
    for family,queries in families.items():
        source=np.stack([np.load(OLD/f'source__{q}__pooled.npy').astype(float) for q in queries])
        qb=np.tile(np.arange(len(queries)),(reps,1)) if family=='vertices' else rng.integers(0,len(queries),(reps,len(queries)))
        for m in methods:
            folder=NEW if m.endswith('_mixed') else OLD
            paths=[folder/f'{m}__{q}__pooled.npy' for q in queries]
            target=np.stack([np.load(p).astype(float) for p in paths])
            for p in paths:inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            values=[];bs=[]
            for h in range(4):
                profs=(5,25) if h<2 else (12,24)
                ix=np.array([r['profession'] in profs for r in rows]);w=weights[:,h]
                n=((target[:,ix]-source[:,ix])@w)**2;d=((source[:,ix]-clean[ix])@w)**2
                values.append(float(np.sqrt(n.mean(1)/d.mean(1)).mean()))
                perq=np.sqrt(n[:,db[h//2]].mean(2).T/d[:,db[h//2]].mean(2).T)
                bs.append(np.take_along_axis(perq,qb,axis=1).mean(1))
            samples[family,m]=np.mean(bs,0)
            summary.append(dict(family=family,method=m,nrmse=float(np.mean(values)),by_head=values))
    differences=[]
    for family in families:
        for response in ['head','pooled']:
            for reference in [response+'_parts',response+'_continuous']:
                delta=samples[family,response+'_mixed']-samples[family,reference]
                point=next(r['nrmse'] for r in summary if r['family']==family and r['method']==response+'_mixed')-next(r['nrmse'] for r in summary if r['family']==family and r['method']==reference)
                differences.append(dict(family=family,method=response+'_mixed',reference=reference,delta=point,interval=np.quantile(delta,[.025,.975]).tolist()))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),summary=summary,differences=differences,inputs=inputs,
                evidence='Adaptive development after coverage analysis, one controlled target seed; all evaluation requests now exposed; later heads excluded from fitting',
                statistics='1000 paired profession/gender document draws; request draws within interior/boundary, seven vertices fixed; fixed heads and target seed',
                identity='Source arrays and context membership agree across runs; every arm512updates; mixed training also supervises pair/full vertices')
    dest.write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'ROUND02_MIXED_ANALYSIS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['family','method','nrmse']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in summary)
    print(json.dumps({'summary':summary,'differences':differences},indent=2))

if __name__=='__main__':main()
