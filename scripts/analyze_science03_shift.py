"""Evaluate the same calibrated functional relation on later human-use readouts."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
BASE=Path('D:/CCAD_Storage/runs/science_upgrade_20260919')
RUN=BASE/'SCIENCE03_shift_execution_reform_v1_20260919'

def main():
    assert json.loads((RUN/'status.json').read_text())['status']=='PASS'
    old=BASE/'SCIENCE02_mixed_training_v1_20260919'
    assert json.loads((old/'evaluation_membership.json').read_text())==json.loads((RUN/'evaluation_membership.json').read_text())
    assert json.loads((old/'program_context_membership.json').read_text())==json.loads((RUN/'program_context_membership.json').read_text())
    panel=json.loads((OUT/'ROUND02_REQUESTS.json').read_text())
    families={'endpoints':list(panel['calibration'])[:7],'interior':list(panel['evaluation'])[:12],
              'boundary':list(panel['evaluation'])[12:],'held_requests':list(panel['evaluation'])}
    tasks=['composer_surgeon_orientation0','composer_surgeon_orientation1','model_software_engineer_orientation0','model_software_engineer_orientation1']
    weights=np.stack([np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz')['weight'].ravel() for t in tasks],1)
    rows=json.loads((RUN/'evaluation_membership.json').read_text())['rows']
    clean=np.load(RUN/'none__full__pooled.npy').astype(float)
    assert np.allclose(clean,np.load(old/'none__full__pooled.npy'),atol=2e-5,rtol=1e-5)
    methods={'initial':RUN,'head_mixed':old,'input_initial':RUN,'tangent_gain':RUN,'tangent_mixed':RUN,'raw_reconstruction':RUN}
    rng=np.random.default_rng(2026091910);reps=2000;db=[]
    for profs in [(5,25),(12,24)]:
        rr=[r for r in rows if r['profession'] in profs]
        strata=[np.array([i for i,r in enumerate(rr) if r['profession']==p and r['gender']==g]) for p in profs for g in [0,1]]
        db.append(np.concatenate([rng.choice(s,(reps,len(s)),replace=True) for s in strata],1))
    draws={'endpoints':np.tile(np.arange(7),(reps,1)),
           'interior':rng.integers(0,12,(reps,12)),'boundary':rng.integers(0,12,(reps,12))}
    draws['held_requests']=np.concatenate([draws['interior'],draws['boundary']+12],axis=1)
    summary={m:{} for m in methods};samples={}
    for family,queries in families.items():
        source=np.stack([np.load(RUN/f'source__{q}__pooled.npy').astype(float) for q in queries])
        for qi,q in enumerate(queries): assert np.allclose(source[qi],np.load(old/f'source__{q}__pooled.npy'),atol=2e-5,rtol=1e-5)
        for m,folder in methods.items():
            target=np.stack([np.load(folder/f'{m}__{q}__pooled.npy').astype(float) for q in queries])
            point=[];bs=[]
            for h in range(4):
                profs=(5,25) if h<2 else (12,24);ix=np.array([r['profession'] in profs for r in rows]);w=weights[:,h]
                num=((target[:,ix]-source[:,ix])@w)**2;den=((source[:,ix]-clean[ix])@w)**2
                point.append(float(np.sqrt(num.mean(1)/den.mean(1)).mean()))
                perq=np.sqrt(num[:,db[h//2]].mean(2).T/den[:,db[h//2]].mean(2).T)
                bs.append(np.take_along_axis(perq,draws[family],axis=1).mean(1))
            samples[m,family]=np.mean(bs,axis=0)
            summary[m][family]=dict(nrmse=float(np.mean(point)),by_head=point,
                                  interval=np.quantile(samples[m,family],[.025,.975]).tolist())
    differences=[]
    for m in ['tangent_gain','tangent_mixed']:
        for ref in methods:
            if m==ref:continue
            for family in families:
                differences.append(dict(method=m,reference=ref,family=family,
                    delta=summary[m][family]['nrmse']-summary[ref][family]['nrmse'],
                    interval=np.quantile(samples[m,family]-samples[ref,family],[.025,.975]).tolist()))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),summary=summary,differences=differences,
                evidence='Adaptive development, same one targetseed1 and exposed256later-task contexts; later heads/labels excluded from fitting',
                statistics='2000 paired profession/gender document and stratified query draws; fixed4heads/source/target; no seed-population inference',
                identity='Source outputs, clean outputs, fit/calibration/evaluation memberships agree with prior comparison',run=str(RUN))
    out=OUT/'ROUND03_SHIFT_ANALYSIS.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print([d for d in differences if d['family']=='held_requests'])

if __name__=='__main__':main()
