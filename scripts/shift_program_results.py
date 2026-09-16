"""Replay fixed-head program responses with paired target/document uncertainty."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=4000)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    arrays=[];identities=[];metadata=None
    for run in a.runs:
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
        rows=[r for r in rows if 'logit' in r and 'operation' in r]
        cfg=json.loads((run/'config.resolved.json').read_text())
        methods=cfg['methods'];queries=cfg['queries']
        docs=sorted({r['component'] for r in rows})
        cells={(r['method'],r['operation'],r['component']):r for r in rows}
        assert len(cells)==len(rows)
        labels=np.array([cells['none','full',d]['label'] for d in docs])
        gender=np.array([cells['none','full',d]['gender'] for d in docs])
        current=dict(methods=methods,queries=queries,documents=docs,labels=labels.tolist(),gender=gender.tolist())
        if metadata is not None:assert metadata==current
        metadata=current
        arrays.append(np.array([[[cells[m,'full' if m=='none' else q,d]['logit'] for d in docs] for q in queries] for m in methods]))
        identities.append(dict(run=str(run),target_seed=cfg['target_seed'],raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest()))
    logits=np.stack(arrays)
    si,ni=methods.index('source'),methods.index('none')
    for i in [si,ni]:assert np.max(np.abs(logits[:,i]-logits[0,i]))<2e-5
    source,clean=logits[0,si],logits[0,ni]
    change=(source>0)!=(clean>0)
    strata=[np.flatnonzero((labels==y)&(gender==g)) for y in [0,1] for g in [0,1]]
    families=dict(all=list(range(len(queries))),parts=[i for i,q in enumerate(queries) if q!='full' and '+' not in q],
                  unseen_combinations=[i for i,q in enumerate(queries) if q=='full' or '+' in q])
    assert all(families.values())

    def measure(targets,draw):
        z=logits[targets][...,draw];ref=source[:,draw];base=clean[:,draw]
        pred=z>0;agree=pred==(ref>0)
        changed=change[:,draw]
        b=[]
        for flag in [False,True]:
            ok=changed==flag;den=ok.sum(-1)
            assert np.all(den>0),'Undefined balanced stratum'
            b.append((agree*ok).sum(-1)/den)
        balanced=.5*(b[0]+b[1])
        correct=pred==labels[draw]
        groups=[np.flatnonzero((labels[draw]==y)&(gender[draw]==g)) for y in [0,1] for g in [0,1]]
        wg=np.stack([correct[...,ix].mean(-1) for ix in groups]).min(0)
        result=[]
        for ix in families.values():
            den=((ref[ix]-base[ix])**2).mean()
            assert den>0
            nrmse=np.sqrt(((z[:,:,ix]-ref[ix])**2).mean((2,3))/den).mean(0)
            result.append(np.stack([nrmse,balanced[:,:,ix].mean((0,2)),correct[:,:,ix].mean((0,2,3)),wg[:,:,ix].mean((0,2))],axis=-1))
        return np.stack(result,axis=1)
    point=measure(np.arange(len(logits)),np.arange(len(docs)))
    rng=np.random.default_rng(2026091604);samples=[]
    for _ in range(a.bootstrap):
        draw=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata])
        samples.append(measure(rng.integers(len(logits),size=len(logits)),draw))
    samples=np.stack(samples);metric=['response_nrmse','balanced_agreement','profession','worst_group']
    def summarize(value,sample):
        return {f:{k:dict(mean=float(value[j,l]),ci95=np.quantile(sample[:,j,l],[.025,.975]).tolist()) for l,k in enumerate(metric)} for j,f in enumerate(families)}
    result={m:summarize(point[i],samples[:,i]) for i,m in enumerate(methods)}
    contrasts={}
    if 'parts' in methods:
        i=methods.index('parts')
        for m in methods:
            if m in ['parts','none','source']:continue
            j=methods.index(m);contrasts['parts minus '+m]=summarize(point[i]-point[j],samples[:,i]-samples[:,j])
    output=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),inputs=identities,metadata=metadata,
                families={f:[queries[i] for i in ix] for f,ix in families.items()},results=result,contrasts=contrasts,
                bootstrap=a.bootstrap,source_changed=change.sum(-1).tolist(),
                inference='Joint target-seed and profession/gender-stratified document bootstrap, shared across methods and requests. Fixed source explanation and query family. nRMSE averages per-target normalized RMS errors; the denominator is the common source effect on each resample.')
    a.output.write_text(json.dumps(output,indent=2))
    np.savez_compressed(a.output.with_suffix('.npz'),logits=logits,labels=labels,gender=gender,methods=np.array(methods),queries=np.array(queries),documents=np.array(docs))
    print(json.dumps({m:{f:round(result[m][f]['response_nrmse']['mean'],4) for f in families} for m in methods}))


if __name__=='__main__':main()
