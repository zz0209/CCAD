"""Aggregate the frozen five-SAE role confirmation without splitting seed dependence."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/correspondence_reform_20260913'
METHODS=['all_context','source64','target64','nearest64','direct64'];OPS=['entity','attribute','both']


def main():
    runs=[ROOT/f'runs/REFORM_R56_binding_roles_confirm_s{s}_v1_20260915' for s in range(1,6)]
    panels=[];metrics=[];identities=[];allrecords=[]
    for run in runs:
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        panels.append(json.loads((run/'panel.json').read_text()));rr=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()];allrecords.append(rr)
        metrics.append({(r['component'],r.get('template'),r.get('order'),r.get('query'),r['method'],r['operation']):r for r in rr if r['kind']=='intervention'})
        identities.append(dict(run=run.relative_to(ROOT).as_posix(),files={f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['config.resolved.json','panel.json','metrics.raw.jsonl','code_hashes.json','status.json']}))
    assert all(p==panels[0] for p in panels)
    worlds=sorted({r['component'] for r in panels[0]['rows'] if r['split']=='development'})
    rng=np.random.default_rng(9561599);draws=rng.integers(len(worlds),size=(10000,len(worlds)))
    def summary(v):
        x=v.mean(axis=tuple(range(1,v.ndim)));boot=x[draws].mean(1)
        return dict(percent=float(x.mean()*100),ci95=(np.quantile(boot,[.025,.975])*100).tolist(),by_seed=(v.mean(axis=(0,2,3))*100).tolist())
    groups={};values={}
    for group,forms in [('all_forms',range(4)),('known_forms',range(2)),('new_forms',range(2,4))]:
        groups[group]={}
        for method in METHODS:
            a=np.empty((len(worlds),5,len(forms),2,3,2),bool)
            for wi,w in enumerate(worlds):
                for si,lookup in enumerate(metrics):
                    for fi,f in enumerate(forms):
                        for order in range(2):
                            for oi,op in enumerate(OPS):
                                for query in range(2):a[wi,si,fi,order,oi,query]=lookup[(w,f,order,query,method,op)]['correct']
            pair=a.all(-1);joint=pair.all(-1);values[(group,method)]=joint.astype(float)
            groups[group][method]={op:summary(pair[:,:,:,:,oi].astype(float)) for oi,op in enumerate(OPS)}
            groups[group][method]['all_three']=summary(joint.astype(float))
    contrasts={g:{ref:summary(values[(g,'target64')]-values[(g,ref)]) for ref in ['nearest64','source64','direct64','all_context']} for g in groups}
    examples=[]
    for row in panels[0]['rows']:
        if row['split']!='development' or row['query']!=0:continue
        w,f,o=row['component'],row['template'],row['order']
        for si,lookup in enumerate(metrics):
            if all(lookup[(w,f,o,q,m,op)]['correct'] for m in ['source64','target64'] for op in OPS for q in range(2)) and not all(lookup[(w,f,o,q,'nearest64',op)]['correct'] for op in OPS for q in range(2)):
                examples.append(dict(row=row,source_seed=si+1,target_seed=(si+1)%5+1,
                    outputs={m:{op:[lookup[(w,f,o,q,m,op)]['answer'].strip() for q in range(2)] for op in OPS} for m in METHODS}))
    baseline={}
    for group,forms in [('all_forms',range(4)),('known_forms',range(2)),('new_forms',range(2,4))]:
        rows_by_id={r['row_id']:r for r in panels[0]['rows']}
        clean=[r['correct'] for r in allrecords[0] if r['kind']=='baseline' and rows_by_id[r['row_id']]['template'] in forms]
        baseline[group]=100*float(np.mean(clean))
    r=dict(worlds=len(worlds),seed_pairs=[[s,s%5+1] for s in range(1,6)],forms=['lives','works','staying','visited'],results=groups,contrasts=contrasts,baseline_token_accuracy=baseline,
        interval='10000 paired bootstrap samples of whole worlds, retaining the five dependent SAE-cycle directions, both orders, forms and queries. By-seed values descriptive.',
        primary='All-three: both cities correct under entity swap, attribute swap and unfitted joint restoration for the same context.',
        scope='Only pre14/pre24 switch to source/target SAE updates; other context layers retain the shared raw program.64 coefficients per changed state. Source response and target recoding use encoders at runtime. Direct retains access to the raw mean program.',
        runs=identities,example=examples[0] if examples else None,eligible_examples=len(examples),example_selection='First world/form/order/seed where all source and target rules hold and a nearest-decoder rule fails; descriptive only. All examples remain in aggregate.')
    (ART/'r56_role_confirmation_analysis.json').write_text(json.dumps(r,indent=2)+'\n')
    for g in groups:
        print(g)
        for m in METHODS:print(m,*(round(groups[g][m][op]['percent'],2) for op in OPS+['all_three']))
        print('target contrasts', {m:v['percent'] for m,v in contrasts[g].items()})


if __name__=='__main__':main()
