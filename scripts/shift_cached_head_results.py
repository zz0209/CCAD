"""Summarize every fixed classifier recipe on retained intervention states."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Keep retained results unchanged; choose a new output: {a.output}')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    index=json.loads((a.run/'HEAD_CHECK_INDEX.json').read_text())
    cfg=json.loads((a.run/'config.resolved.json').read_text())
    tasks=sorted({key.rsplit('/',1)[1] for key in index['files']})
    seeds,epochs=cfg['probe_seeds'],cfg['epochs']
    queries=['full','pronouns','names','associated_words']
    methods=['none','source','geometry','geometry_gain','native','raw']
    targets=cfg['seeds']
    values=np.empty((len(methods),4,len(targets),len(tasks),len(seeds),len(epochs),3))
    absent=[]
    for ti,target in enumerate(targets):
        for ki,task in enumerate(tasks):
            def load(method,query):
                t=targets[0] if method in ['none','source'] else target
                q='full' if method=='none' else query
                return np.load(a.run/index['files'][f'{t}/{method}__{q}/{task}'])
            baseline=load('none','full')
            clean=baseline['logits']>0
            labels,gender,documents=(baseline[k] for k in ['labels','gender','document_sha256'])
            for qi,query in enumerate(queries):
                source=load('source',query)['logits']>0
                changed=source!=clean
                for mi,method in enumerate(methods):
                    current=load(method,query)
                    assert np.array_equal(current['document_sha256'],documents)
                    assert np.array_equal(current['labels'],labels)
                    prediction=current['logits']>0
                    correct=prediction==labels
                    accuracy=correct.mean(-1)
                    group=np.stack([correct[...,(labels==y)&(gender==g)].mean(-1)
                                    for y in [0,1] for g in [0,1]])
                    agreement=prediction==source
                    strata=[]
                    for flag in [False,True]:
                        included=changed==flag
                        count=included.sum(-1)
                        strata.append(np.divide((agreement*included).sum(-1),count,
                            out=np.full(count.shape,np.nan),where=count>0))
                        if mi==0 and ti==0:
                            absent.extend(dict(task=task,query=query,source_changed=flag,
                                probe_seed=seeds[s],epochs=epochs[e]) for s,e in np.argwhere(count==0))
                    values[mi,qi,ti,ki]=np.stack([accuracy,group.min(0),.5*(strata[0]+strata[1])],-1)
    # Fixed cohorts: variation across head seeds is displayed, not treated as
    # independent test documents or substituted for the primary paired interval.
    summary=values.mean(axis=(2,3))
    names=['accuracy','worst_group','balanced_source_agreement']
    output={}
    for ei,epoch in enumerate(epochs):
        output[str(epoch)]={}
        for mi,method in enumerate(methods):
            output[str(epoch)][method]={}
            for qi,query in enumerate(queries+['parts_mean']):
                x=summary[mi,qi,:,ei] if qi<4 else summary[mi,1:,:,ei].mean(0)
                output[str(epoch)][method][query]={n:dict(mean=float(x[:,j].mean()),
                    by_probe_seed={str(seed):float(x[s,j]) for s,seed in enumerate(seeds)},
                    range=[float(x[:,j].min()),float(x[:,j].max())]) for j,n in enumerate(names)}
    payload=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=str(a.run),
        input_index_sha256=hashlib.sha256((a.run/'HEAD_CHECK_INDEX.json').read_bytes()).hexdigest(),
        probe_seeds=seeds,epochs=epochs,target_seeds=targets,tasks=tasks,results=output,
        undefined_balanced_strata=absent,primary_replay_max_logit_error=index['primary_replay_max_logit_error'],
        scope='Supplementary fixed-recipe sensitivity on the same confirmation cohort. All six initializations and both epoch counts retained; no recipe or relation selected by test performance.')
    # Preserve undefined scores explicitly rather than serializing nonstandard NaN.
    def finite(obj):
        if isinstance(obj,dict):return {k:finite(v) for k,v in obj.items()}
        if isinstance(obj,list):return [finite(v) for v in obj]
        if isinstance(obj,float) and not np.isfinite(obj):return None
        return obj
    a.output.write_text(json.dumps(finite(payload),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),undefined_strata=len(absent))))


if __name__=='__main__':
    main()
