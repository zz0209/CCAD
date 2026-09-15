"""World-clustered analysis of single swaps and unfitted joint restoration."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np


def analyze(run):
    panel=json.loads((run/'panel.json').read_text()); rows=panel['rows']
    records=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()]
    dev=[x for x in records if x.get('kind')=='intervention']
    worlds=sorted({x['component'] for x in dev}); wi={c:i for i,c in enumerate(worlds)}
    rng=np.random.default_rng(9561599); draws=rng.integers(len(worlds),size=(10000,len(worlds)))
    def stat(x):
        x=np.asarray(x,float); boot=x[draws].mean(1)
        return dict(percent=float(x.mean()*100),ci95=(np.quantile(boot,[.025,.975])*100).tolist())
    groups=defaultdict(list)
    for r in dev:groups[r['method']].append(r)
    metrics={};arrays={};lookup={}
    for method,rr in groups.items():
        lookup[method]={(r['row_id'],r['operation']):r for r in rr}
        variants=defaultdict(dict)
        for r in rr:variants[(r['component'],r['order'],r['template'])][(r['operation'],r['query'])]=r
        all3=np.zeros(len(worlds));counts=np.zeros(len(worlds));summ={}
        for op in ['entity','attribute','both']:
            x=np.zeros(len(worlds));n=np.zeros(len(worlds))
            for (world,order,form),v in variants.items():
                assert (op,0) in v and (op,1) in v
                x[wi[world]]+=all(v[(op,q)]['correct'] for q in [0,1]);n[wi[world]]+=1
            summ[op+'_complete']=stat(x/n);arrays[(method,op)]=x/n
        joint_given_singles=[]
        for (world,order,form),v in variants.items():
            good=all(v[(op,q)]['correct'] for op in ['entity','attribute','both'] for q in [0,1])
            all3[wi[world]]+=good;counts[wi[world]]+=1
            if all(v[(op,q)]['correct'] for op in ['entity','attribute'] for q in [0,1]):
                joint_given_singles.append(all(v[('both',q)]['correct'] for q in [0,1]))
        summ['all_three_rules']=stat(all3/counts);arrays[(method,'all_three_rules')]=all3/counts
        summ['joint_given_both_singles_correct']=dict(n=len(joint_given_singles),percent=100*float(np.mean(joint_given_singles)) if joint_given_singles else None)
        summ['minimum_edit_norm']=min(r['edit_norm'] for r in rr)
        summ['token_rule_accuracy']={op:100*float(np.mean([r['correct'] for r in rr if r['operation']==op])) for op in ['entity','attribute','both']}
        metrics[method]=summ
    contrasts={}
    for method in ['response','gain','direct_fit','raw_response']:
        if method not in metrics:continue
        for ref in ['source_mean64','field','assignment','target_mean64']:
            if ref not in metrics:continue
            contrasts[method+' - '+ref]={op:stat(arrays[(method,op)]-arrays[(ref,op)]) for op in ['entity','attribute','both','all_three_rules']}
    baseline=[r for r in records if r['kind']=='baseline' and r['split']=='development']
    result=dict(run=str(run),worlds=len(worlds),context_variants=len({(x['component'],x['order'],x['template']) for x in dev}),
        baseline_token_accuracy=100*float(np.mean([r['correct'] for r in baseline])),methods=metrics,contrasts=contrasts,
        inference='10000 paired bootstrap samples of whole worlds; both orders/forms/queries retained. One development SAE direction; no seed-population inference.',
        endpoint='Both queried cities correct for each operation. All-three requires entity swap, attribute swap and unfit joint restoration in the same context variant.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    r=analyze(a.run);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('Method | entity pair | attribute pair | joint pair | all three rules')
    for m,v in r['methods'].items():print(m,*(round(v[k]['percent'],2) for k in ['entity_complete','attribute_complete','both_complete','all_three_rules']))
