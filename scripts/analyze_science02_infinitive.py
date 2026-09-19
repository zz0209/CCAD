"""Replay the frozen coverage comparison on a published linguistic function."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
from analyze_science02_coverage import distances,coverage_order

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
RUN=Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE02_infinitive_coverage_v1_20260919')

def main():
    dest=OUT/'ROUND02_INFINITIVE_ANALYSIS.json'
    if dest.exists():raise FileExistsError(dest)
    assert json.loads((RUN/'status.json').read_text())['status']=='PASS'
    panel=json.loads((OUT/'ROUND02_INFINITIVE_REQUESTS.json').read_text())
    index=json.loads((RUN/'INDEX.json').read_text())
    cn=list(panel['calibration']);en=list(panel['evaluation']);names=cn+en
    assert index['queries']==names
    rows=index['rows'];ci=np.array([r['coverage_split']=='calibration' for r in rows])
    ei=~ci
    assert not {r['pair'] for r,c in zip(rows,ci) if c}&{r['pair'] for r,e in zip(rows,ei) if e}
    methods=['native','geometry','geometry_gain','native_response_relation_8','native_tangent_relation_8']
    arrays=np.load(RUN/'responses.npz')['log_probability']
    clean=arrays[index['methods'].index('none'),0]
    source=arrays[index['methods'].index('source')]
    # Only source values and candidate-query calibration contexts enter selection.
    dc=distances(np.array(list(panel['calibration'].values())+list(panel['evaluation'].values())))
    ds=distances((source[:,ci]-clean[ci])/np.sqrt(ci.sum()))
    orders={'endpoints':coverage_order(dc,list(range(3)),3),
            'coordinate_coverage':coverage_order(dc,list(range(len(cn))),3),
            'source_response_coverage':coverage_order(ds,list(range(len(cn))),3)}
    targets=arrays[[index['methods'].index(m) for m in methods]]
    den=np.mean((source[:len(cn),ci]-clean[ci])**2,axis=1)
    assert den.min()>1e-12
    score=np.sqrt(np.mean((targets[:,:len(cn),ci]-source[:len(cn),ci])**2,axis=2)/den)
    rng=np.random.default_rng(2026091906)
    random=[rng.permutation(len(cn)).tolist() for _ in range(200)]
    policies=[]
    for budget in [1,3]:
        for design,order in list(orders.items())+[(f'random_{i:03d}',o) for i,o in enumerate(random)]:
            ix=np.array(order[:budget]);d=ds if design=='source_response_coverage' else dc
            pred=score[:,ix][:,d[len(cn):,ix].argmin(1)]
            policies.append(dict(design=design,budget=budget,predictor='nearest',queries=[cn[i] for i in ix],values=pred,choices=pred.argmin(0)))
            if not design.startswith('random_'):
                pred=np.repeat(score[:,ix].mean(1)[:,None],len(en),axis=1)
                policies.append(dict(design=design,budget=budget,predictor='global_choice',queries=[cn[i] for i in ix],values=pred,choices=pred.argmin(0)))
    freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),orders={k:[cn[i] for i in v] for k,v in orders.items()},
                policies=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()} for p in policies],
                information='Source responses on32 calibration contexts for32 queries; target responses on selected candidate queries and32 calibration contexts only',
                rule='Same nearest-query and global-choice rules as the human-explanation panel; written before evaluation slicing')
    (OUT/'ROUND02_INFINITIVE_PREDICTIONS.json').write_text(json.dumps(freeze,indent=2)+'\n')
    # Array materialization is logged separately from using its held-out slice.
    se=source[len(cn):,ei];ce=clean[ei]
    te=targets[:,len(cn):,ei]
    numerator=(te-se)**2;denominator=(se-ce)**2
    actual=np.sqrt(numerator.mean(2)/denominator.mean(1))
    oracle=actual.min(0)
    external=arrays[index['methods'].index('raw_reconstruction'),len(cn):,ei].T
    # NumPy advanced indexing places the context dimension first in this slice.
    assert external.shape==se.shape
    raw=np.sqrt(np.mean((external-se)**2,axis=1)/denominator.mean(1))
    summary=[]
    for p in policies:
        value=actual[p['choices'],np.arange(len(en))]
        summary.append({k:p[k] for k in ['design','budget','predictor','queries']}|
                       dict(mean_nrmse=float(value.mean()),interior=float(value[:12].mean()),boundary=float(value[12:].mean()),
                            regret=float((value-oracle).mean()),prediction_mae=float(np.abs(p['values']-actual).mean()),
                            exact_choice=float((p['choices']==actual.argmin(0)).mean()),
                            choices={m:int((p['choices']==i).sum()) for i,m in enumerate(methods)}))
    for budget in [1,3]:
        rr=[r for r in summary if r['design'].startswith('random_') and r['budget']==budget]
        summary.append(dict(design='random_mean_200',budget=budget,predictor='nearest',
                            **{k:float(np.mean([r[k] for r in rr])) for k in ['mean_nrmse','interior','boundary','regret','prediction_mae','exact_choice']}))
    er=[r for r,e in zip(rows,ei) if e];pairs=sorted({r['pair'] for r in er})
    groups=np.array([[i for i,r in enumerate(er) if r['pair']==p] for p in pairs])
    reps=1000;rng=np.random.default_rng(2026091907)
    db=groups[rng.integers(0,len(groups),(reps,len(groups)))].reshape(reps,-1)
    qb=np.concatenate([rng.integers(0,12,(reps,12)),rng.integers(12,24,(reps,12))],axis=1)
    sampled={}
    for p in policies:
        if p['budget']!=3 or p['predictor']!='nearest' or p['design'].startswith('random_'):continue
        nn=numerator[p['choices'],np.arange(len(en))]
        values=np.sqrt(nn[:,db].mean(2).T/denominator[:,db].mean(2).T)
        sampled[p['design']]=np.take_along_axis(values,qb,axis=1).mean(1)
    intervals={k:np.quantile(v-sampled['endpoints'],[.025,.975]).tolist() for k,v in sampled.items() if k!='endpoints'}
    vertex={tuple(v):source[names.index(k)]-clean for k,v in panel['calibration'].items() if k in cn[:3]}
    vertex[(0.,0.)]=np.zeros_like(clean)
    curvature=[]
    for i,q in enumerate(list(panel['calibration'].values())+list(panel['evaluation'].values())):
        q=np.array(q);approx=sum(np.prod(np.where(np.array(bits),q,1-q))*v for bits,v in vertex.items())
        curvature.append(float(np.linalg.norm((source[i]-clean-approx)[ci])/max(np.linalg.norm((source[i]-clean)[ci]),1e-12)))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),fixed={m:dict(mean_nrmse=float(a.mean()),interior=float(a[:12].mean()),boundary=float(a[12:].mean())) for m,a in zip(methods,actual)},
                external=dict(raw_reconstruction=float(raw.mean())),oracle=float(oracle.mean()),
                summary=[r for r in summary if not r['design'].startswith('random_') or r['design']=='random_mean_200'],
                intervals=intervals,source_interpolation_error=dict(zip(names,curvature)),
                statistics='1000 paired lexical-pair and interior/boundary query resamples; fixed calibration set and one target dictionary',
                evidence='Published4 features; researcher-authored previously exposed64 contexts;32 new request coordinates; split32cal/32eval by lexical pairs',
                sha256={str(RUN/f):hashlib.sha256((RUN/f).read_bytes()).hexdigest() for f in ['responses.npz','INDEX.json','config.resolved.json']})
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
