"""Paired development analysis of input-conditioned human-part counterparts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--classifier',choices=['frozen','retrained'])
    p.add_argument('--queries',nargs='+')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    status=json.loads((a.run/'status.json').read_text())
    assert status['status']=='PASS',status
    cfg=json.loads((a.run/'config.resolved.json').read_text())
    raw=a.run/'metrics.raw.jsonl'
    rows=[json.loads(x) for x in raw.read_text().splitlines()]
    rows=[r for r in rows if 'operation' in r and 'logit' in r]
    if a.queries:rows=[r for r in rows if r['operation'] in a.queries]
    if a.classifier:rows=[r for r in rows if r.get('classifier')==a.classifier]
    docs=sorted({r['component'] for r in rows})
    cells={(r['method'],r['operation'],r['component']):r for r in rows}
    assert len(cells)==len(rows)
    ref=[cells['none','full',d] for d in docs]
    y=np.array([r['label'] for r in ref]);g=np.array([r['gender'] for r in ref])
    strata=[np.flatnonzero((y==yy)&(g==gg)) for yy in [0,1] for gg in [0,1]]
    methods=list(dict.fromkeys(r['method'] for r in rows if r['method']!='none'));queries=a.queries or cfg['queries']
    logits=np.array([[[cells[m,q,d]['logit'] for d in docs] for q in queries] for m in methods])
    clean=np.array([r['logit'] for r in ref])
    source=logits[methods.index('source')]
    pred=logits>0;truth=source>0;change=truth!=(clean>0)
    assert all(0<cc.sum()<len(docs) for cc in change)
    correct=pred==y;agree=pred==truth
    def measure(draw):
        groups=[draw[ix] for ix in strata]
        bp=[]
        for flag in [False,True]:
            eligible=(change==flag)[:,draw]
            bp.append((agree[:,:,draw]*eligible).sum(-1)/eligible.sum(-1).clip(1))
        denominator=((source[:,draw]-clean[draw])**2).mean()
        return dict(balanced_agreement=.5*(bp[0]+bp[1]),
                    profession=correct[:,:,draw].mean(-1),
                    worst_group=np.stack([correct[:,:,ix].mean(-1) for ix in groups]).min(0),
                    response_nrmse=np.sqrt(((logits[:,:,draw]-source[:,draw])**2).mean((1,2))/denominator))
    point=measure(np.arange(len(docs)))
    rng=np.random.default_rng(20260916)
    draws={k:[] for k in point}
    for _ in range(a.bootstrap):
        ix=np.arange(len(docs))
        for st in strata:ix[st]=rng.choice(st,len(st),replace=True)
        sample=measure(ix)
        for k,v in sample.items():draws[k].append(v)
    draws={k:np.array(v) for k,v in draws.items()}
    out={}
    for i,m in enumerate(methods):
        out[m]={}
        for k,v in point.items():
            if k=='response_nrmse':
                out[m][k]=dict(value=float(v[i]),ci=np.quantile(draws[k][:,i],[.025,.975]).tolist())
            else:
                out[m][k]={q:float(v[i,j]) for j,q in enumerate(queries)}
                parts=[j for j,q in enumerate(queries) if q!='full']
                out[m][k]['parts_mean']=float(v[i,parts].mean())
    differences={}
    for left,right in [('input_tangent_budget','native'),('input_finite_budget','native'),
                       ('input_tangent_budget','input_fixed_budget'),('input_finite','input_fixed'),
                       ('input_finite','raw'),('input_finite','raw_reconstruction'),
                       ('input_tangent_budget','geometry'),('input_tangent_budget','geometry_gain'),
                       ('input_tangent_budget','raw'),('input_tangent_budget','raw_reconstruction'),
                       ('parts','initial'),('parts','native'),('parts','whole'),
                       ('parts','parts_relation'),('parts','mse'),('parts','clean_e2e'),
                       ('parts','raw'),('parts','raw_reconstruction')]:
        if left not in methods or right not in methods:continue
        li,ri=methods.index(left),methods.index(right)
        differences[left+' - '+right]={}
        for k in point:
            if k=='response_nrmse':v=point[k][li]-point[k][ri];samples=draws[k][:,li]-draws[k][:,ri]
            else:
                parts=[j for j,q in enumerate(queries) if q!='full']
                v=(point[k][li,parts]-point[k][ri,parts]).mean()
                samples=(draws[k][:,li,parts]-draws[k][:,ri,parts]).mean(-1)
            differences[left+' - '+right][k]=dict(value=float(v),ci=np.quantile(samples,[.025,.975]).tolist())
    payload=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=str(a.run),
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),n_documents=len(docs),
        source_changed_by_query={q:int(change[j].sum()) for j,q in enumerate(queries)},
        statistics='Exploratory paired profession/gender-stratified document bootstrap; fixed source, target and query set. No independent confirmation.',
        source_effect_rms=float(np.sqrt(((source-clean)**2).mean())),methods=out,differences=differences,
        classifier=a.classifier,scope=cfg['scope'],status=status)
    a.output.write_text(json.dumps(payload,indent=2))
    np.savez_compressed(a.output.with_suffix('.npz'),logits=logits,clean=clean,labels=y,gender=g,
                        methods=np.array(methods),queries=np.array(queries),documents=np.array(docs))
    for m,r in out.items():
        print(m,'nRMSE',round(r['response_nrmse']['value'],4),'part agreement/acc/WG',
              *(round(r[k]['parts_mean']*100,2) for k in ['balanced_agreement','profession','worst_group']))


if __name__=='__main__':main()
