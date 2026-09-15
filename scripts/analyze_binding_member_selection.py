"""World-clustered utility of selecting members of a frozen binding-role update."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np


def analyze(run,forms=None):
    runs=[run] if isinstance(run,Path) else run
    records=[json.loads(x) for p in runs for x in (p/'metrics.raw.jsonl').read_text().splitlines()]
    rr=[r for r in records if r['kind']=='member_selection']
    if forms is not None:rr=[r for r in rr if r['template'] in forms]
    worlds=sorted({r['component'] for r in rr});wi={v:i for i,v in enumerate(worlds)}
    draws=np.random.default_rng(9571511).integers(len(worlds),size=(10000,len(worlds)))
    def stat(x):
        x=np.asarray(x,float)
        return dict(percent=float(100*x.mean()),ci95=(100*np.quantile(x[draws].mean(1),[.025,.975])).tolist())
    grouped=defaultdict(list)
    for r in rr:grouped[(r['method'],r['members'])].append(r)
    methods={};arrays={}
    for (method,k),rows in grouped.items():
        key=f'{method}/k{k}';variants=defaultdict(dict)
        for r in rows:variants[(r['target_seed'],r['component'],r['order'],r['template'])][r['operation'],r['query']]=r
        metrics={}
        for endpoint,ops in [('entity',['entity']),('restoration',['both']),('complete',['entity','both'])]:
            x=np.zeros(len(worlds));n=np.zeros(len(worlds))
            for (target,world,order,form),v in variants.items():
                assert all((op,q) in v for op in ops for q in (0,1))
                x[wi[world]]+=all(v[op,q]['correct'] for op in ops for q in (0,1));n[wi[world]]+=1
            arrays[key,endpoint]=x/n;metrics[endpoint]=stat(x/n)
        metrics['token_accuracy']={op:100*float(np.mean([r['correct'] for r in rows if r['operation']==op])) for op in ('entity','both')}
        metrics['mean_edit_norm']=float(np.mean([r['edit_norm'] for r in rows]))
        metrics['prediction_by_operation']={}
        for op in ('entity','both'):
            x=[r for r in rows if r['operation']==op and r['predicted_margin_gain'] is not None]
            if x:
                p=np.array([r['predicted_margin_gain'] for r in x]);a=np.array([r['actual_margin_gain'] for r in x])
                metrics['prediction_by_operation'][op]=dict(predicted_mean=float(p.mean()),actual_mean=float(a.mean()),mae=float(np.mean(abs(p-a))),pearson=float(np.corrcoef(p,a)[0,1]))
        methods[key]=metrics
    contrasts={}
    for k in sorted({r['members'] for r in rr}):
        selected=f'source_path/k{k}'
        if selected not in methods:continue
        for ref in ['geometry','source_cached','source_fullgrad','source_pooled','raw_path','target_path','source_selected']:
            base=f'{ref}/k{k}'
            if base in methods:contrasts[selected+' - '+base]={endpoint:stat(arrays[selected,endpoint]-arrays[base,endpoint]) for endpoint in ('entity','restoration','complete')}
    baseline=[r for r in records if r['kind']=='baseline']
    return dict(runs=[str(p) for p in runs],worlds=len(worlds),target_seeds=sorted({r['target_seed'] for r in rr}),context_variants=len({(r['component'],r['order'],r['template']) for r in rr}),
        baseline_token_accuracy=100*float(np.mean([r['correct'] for r in baseline])),methods=methods,contrasts=contrasts,
        primary='k16 complete: both cities correct for entity exchange and E+A restoration in each context; four answers.',
        inference='10000 paired world-cluster bootstrap resamples, retaining the fixed target cohort, orders, forms, queries and operations. Targets share source1; no independent seed-direction inference.',
        world_values={m:{ep:arrays[m,ep].tolist() for ep in ('entity','restoration','complete')} for m in methods},world_ids=worlds,
        prediction_scope='Each selector scores its own chosen set; correlation is within a fixed member budget and operation, not a cross-candidate ranking test.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path,nargs='+');p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    result=analyze(a.run);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('method/budget | entity pair | restoration pair | complete')
    for m,v in result['methods'].items():print(m,*(round(v[k]['percent'],2) for k in ('entity','restoration','complete')))
    print('Primary contrasts:')
    for m,v in result['contrasts'].items():
        if m.startswith('source_path/k16'):print(m,v['complete'])
