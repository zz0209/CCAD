"""Budget-matched prediction of unmeasured intervention fidelity.

Only candidate calibration responses enter selection. Evaluation responses are
loaded after designs and predictions are fixed. All contexts remain development.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
BASE=Path('D:/CCAD_Storage/runs/science_upgrade_20260919')
CAL=BASE/'SCIENCE02_coverage_cal_v2_20260919'
EVAL=BASE/'SCIENCE02_coverage_tasks_v1_20260919'

def distances(x):
    gram=x@x.T
    return np.maximum(np.diag(gram)[:,None]+np.diag(gram)[None,:]-2*gram,0)

def coverage_order(d,candidates,n=7):
    # Greedy minimax coverage of the requested coordinate/response domain.
    selected=[];nearest=np.full(len(d),np.inf)
    for _ in range(n):
        c=min((j for j in candidates if j not in selected),
              key=lambda j:(float(np.minimum(nearest,d[:,j]).max()),float(np.minimum(nearest,d[:,j]).mean()),j))
        selected.append(c);nearest=np.minimum(nearest,d[:,c])
    return selected

def main():
    output=OUT/'ROUND02_COVERAGE_ANALYSIS.json'
    if output.exists():raise FileExistsError(output)
    for folder in [CAL,EVAL]: assert json.loads((folder/'status.json').read_text())['status']=='PASS'
    panel=json.loads((OUT/'ROUND02_REQUESTS.json').read_text())
    c_names=list(panel['calibration']);e_names=list(panel['evaluation']);names=c_names+e_names
    methods=panel['methods'];coords=np.array(list(panel['calibration'].values())+list(panel['evaluation'].values()))
    cfg=json.loads((CAL/'config.resolved.json').read_text())
    inputs={}
    def load(folder,name,key=None):
        p=folder/name;inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        a=np.load(p);return np.asarray(a if key is None else a[key],dtype=np.float64)
    old=load(Path(cfg['frozen_source_run']),'probe.npz','weight').ravel()
    tasks=['composer_surgeon_orientation0','composer_surgeon_orientation1',
           'model_software_engineer_orientation0','model_software_engineer_orientation1']
    weights=np.stack([old]+[load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916',f'none__full__{t}__probe42.npz','weight').ravel() for t in tasks],axis=1)
    clean=load(CAL,'none__full__pooled.npy')
    source=np.stack([load(CAL,f'source__{q}__pooled.npy') for q in names])
    d_coord=distances(coords);d_source=distances((source-clean).reshape(len(names),-1)/np.sqrt(clean.size))
    orders={'endpoints':coverage_order(d_coord,list(range(7))),
            'coordinate_coverage':coverage_order(d_coord,list(range(len(c_names)))),
            'source_response_coverage':coverage_order(d_source,list(range(len(c_names))))}
    source_effect=(source[:len(c_names)]-clean)@weights
    denominator=np.mean(source_effect**2,axis=1)
    assert denominator.min()>1e-12
    scores=[]
    for m in methods:
        target=np.stack([load(CAL,f'{m}__{q}__pooled.npy') for q in c_names])
        scores.append(np.sqrt(np.mean(((target-source[:len(c_names)])@weights)**2,axis=1)/denominator))
    scores=np.array(scores) # method,calibration request,head
    predictions=[]
    rng=np.random.default_rng(2026091903)
    random_orders=[rng.permutation(len(c_names)).tolist() for _ in range(200)]
    for budget in panel['budgets']:
        for design,order in list(orders.items())+[(f'random_{i:03d}',o) for i,o in enumerate(random_orders)]:
            ix=np.array(order[:budget]);d=d_source if design=='source_response_coverage' else d_coord
            for information in ['old_scalar','full_response']:
                hix=[0]*4 if information=='old_scalar' else [1,2,3,4]
                training=scores[:,ix][:,:,hix]
                nearest=d[len(c_names):,ix].argmin(axis=1)
                pred=training[:,nearest,:]
                predictions.append(dict(design=design,budget=budget,information=information,predictor='nearest',
                                        queries=[c_names[j] for j in ix],values=pred,choices=pred.argmin(0)))
                if not design.startswith('random_'):
                    # Choose one adapted dictionary/relation for the whole
                    # subsequent request family, without switching per request.
                    constant=np.repeat(training.mean(1)[:,None,:],len(e_names),axis=1)
                    predictions.append(dict(design=design,budget=budget,information=information,predictor='global_choice',
                                            queries=[c_names[j] for j in ix],values=constant,choices=constant.argmin(0)))
                    length=np.median(d[:len(c_names),:len(c_names)][np.triu_indices(len(c_names),1)])
                    kernel=np.exp(-d/(2*max(length,1e-12)))
                    k=kernel[np.ix_(ix,ix)]+1e-4*np.eye(budget)
                    coeff=np.linalg.solve(k,training.transpose(1,0,2).reshape(budget,-1))
                    pred=np.maximum(kernel[len(c_names):,ix]@coeff,0).reshape(len(e_names),len(methods),4).transpose(1,0,2)
                    predictions.append(dict(design=design,budget=budget,information=information,predictor='rbf',
                                            queries=[c_names[j] for j in ix],values=pred,choices=pred.argmin(0)))
    freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),source_queries=len(names)*len(clean),
                calibration_documents=len(clean),orders={k:[c_names[i] for i in v] for k,v in orders.items()},
                predictions=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()} for p in predictions],
                policy='Written before opening task evaluation arrays; all target scores used here belong to candidate calibration queries')
    (OUT/'ROUND02_PREDICTIONS.json').write_text(json.dumps(freeze,indent=2)+'\n')
    # Target evaluation begins here, after all predictions have been materialized.
    rows=json.loads((EVAL/'evaluation_membership.json').read_text())['rows']
    cal_rows=json.loads((CAL/'evaluation_membership.json').read_text())['rows']
    assert not {r['document_sha256'] for r in rows}&{r['document_sha256'] for r in cal_rows}
    ec=load(EVAL,'none__full__pooled.npy')
    es=np.stack([load(EVAL,f'source__{q}__pooled.npy') for q in e_names])
    num=[];den=[];actual=[];external={}
    for m in methods+['raw_reconstruction']:
        et=np.stack([load(EVAL,f'{m}__{q}__pooled.npy') for q in e_names])
        nr=[];dr=[]
        for hi in range(4):
            professions=(5,25) if hi<2 else (12,24)
            ix=np.array([r['profession'] in professions for r in rows])
            w=weights[:,hi+1]
            nr.append(((et[:,ix]-es[:,ix])@w)**2)
            dr.append(((es[:,ix]-ec[ix])@w)**2)
        n=np.stack(nr,axis=-1);dd=np.stack(dr,axis=-1)
        value=np.sqrt(n.mean(1)/dd.mean(1))
        if m in methods: num.append(n);actual.append(value)
        else:external[m]=dict(mean_nrmse=float(value.mean()),interior=float(value[:12].mean()),boundary=float(value[12:].mean()))
        den=dd
    actual=np.array(actual);num=np.array(num)
    oracle=actual.min(0);summary=[]
    for p in predictions:
        values=np.take_along_axis(actual,p['choices'][None],axis=0)[0]
        p['evaluated_values']=values
        summary.append({k:p[k] for k in ['design','budget','information','predictor','queries']}|
                       dict(mean_nrmse=float(values.mean()),interior=float(values[:12].mean()),boundary=float(values[12:].mean()),
                            regret=float((values-oracle).mean()),prediction_mae=float(np.abs(p['values']-actual).mean()),
                            exact_choice=float((p['choices']==actual.argmin(0)).mean()),
                            choices={m:int((p['choices']==i).sum()) for i,m in enumerate(methods)}))
    # Random placement is an expectation over200 stored layouts, not200 independent studies.
    random=[]
    for budget in panel['budgets']:
        for info in ['old_scalar','full_response']:
            rr=[r for r in summary if r['design'].startswith('random_') and r['budget']==budget and r['information']==info]
            random.append(dict(design='random_mean_200',budget=budget,information=info,predictor='nearest',
                               **{k:float(np.mean([r[k] for r in rr])) for k in ['mean_nrmse','interior','boundary','regret','prediction_mae','exact_choice']}))
    fixed={m:dict(mean_nrmse=float(a.mean()),interior=float(a[:12].mean()),boundary=float(a[12:].mean())) for m,a in zip(methods,actual)}
    # Fixed-policy, paired document and request resampling. The shared orientations
    # use identical resampled documents and all methods share the same requests.
    rng=np.random.default_rng(2026091904);reps=1000
    qb=np.concatenate([rng.integers(0,12,(reps,12)),rng.integers(12,24,(reps,12))],axis=1)
    docboot=[]
    for professions in [(5,25),(12,24)]:
        rr=[r for r in rows if r['profession'] in professions]
        strata=[np.array([i for i,r in enumerate(rr) if r['profession']==pr and r['gender']==g]) for pr in professions for g in [0,1]]
        docboot.append(np.concatenate([rng.choice(s,(reps,len(s)),replace=True) for s in strata],axis=1))
    primary=[p for p in predictions if p['budget']==3 and p['predictor']=='nearest' and not p['design'].startswith('random_')]
    sampled={}
    for p in primary:
        scores=[]
        for h in range(4):
            picked=num[p['choices'][:,h],np.arange(len(e_names)),:,h]
            nn=picked[:,docboot[h//2]].mean(2).T
            dd=den[:,:,h][:,docboot[h//2]].mean(2).T
            perq=np.sqrt(nn/dd)
            scores.append(np.take_along_axis(perq,qb,axis=1).mean(1))
        sampled[p['design'],p['information']]=np.mean(scores,axis=0)
    intervals=[]
    for info in ['old_scalar','full_response']:
        for design in ['coordinate_coverage','source_response_coverage']:
            diff=sampled[design,info]-sampled['endpoints',info]
            intervals.append(dict(design=design,reference='endpoints',budget=3,information=info,
                                  interval=np.quantile(diff,[.025,.975]).tolist()))
    # Curvature of the executed source response relative to vertex interpolation.
    vertex={tuple(q):source[names.index(name)]-clean for name,q in panel['calibration'].items() if name in c_names[:7]}
    vertex[(0.,0.,0.)]=np.zeros_like(clean)
    residual=[]
    for i,q in enumerate(coords):
        approx=sum(np.prod(np.where(np.array(bits),q,1-q))*v for bits,v in vertex.items())
        residual.append(float(np.linalg.norm(source[i]-clean-approx)/max(np.linalg.norm(source[i]-clean),1e-12)))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),evidence='Frozen24 request coordinates; previously exposed contexts and one development target seed',
                evaluation_documents=len(rows),calibration_documents=len(cal_rows),fixed=fixed,external=external,
                oracle_nrmse=float(oracle.mean()),summary=[r for r in summary if not r['design'].startswith('random_')]+random,
                paired_intervals=intervals,source_interpolation_error=dict(zip(names,residual)),
                statistics='1000 paired profession/gender document and interior/boundary request resamples; fixed calibration/design/candidate set; no seed inference',
                inputs=inputs)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fixed':fixed,'external':external,'oracle':result['oracle_nrmse'],
                      'primary':[r for r in result['summary'] if r['budget']==3 and r['predictor']=='nearest'],
                      'intervals':intervals},indent=2))

if __name__=='__main__':main()
