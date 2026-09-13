"""Aggregate source relevance and transfer of components and their union."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(run,out):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cells={};source={}
    for line in (run/'metrics.raw.jsonl').open(encoding='utf-8'):
        r=json.loads(line);key=tuple(r[k] for k in ['objective','seed','target_seed','task','split','operation','method'])
        if key not in cells:
            cells[key]=dict(zip(['objective','seed','target_seed','task','split','operation','method'],key),n=0,margin_error=0.,source_kl=0.,source_effect_kl=0.,source_decrement=0.,method_decrement=0.,clean_accuracy=0,source_accuracy=0,method_accuracy=0,source_changed=0,changed_retained=0,clean_margin_error=0.,edit_norm=0.)
        c=cells[key];c['n']+=1
        for k in ['margin_error','source_kl','source_effect_kl','edit_norm']:c[k]+=r[k]
        c['source_decrement']+=r['clean_margin']-r['source_margin'];c['method_decrement']+=r['clean_margin']-r['margin']
        c['clean_margin_error']+=abs(r['clean_margin']-r['source_margin'])
        c['clean_accuracy']+=r['clean_margin']>0;c['source_accuracy']+=r['source_margin']>0;c['method_accuracy']+=r['margin']>0
        c['source_changed']+=r['source_changed'];c['changed_retained']+=r['source_changed'] and r['source_decision_agreement']
        if r['method']=='source':
            sk=tuple(r[k] for k in ['objective','seed','task','split','row_id'])
            source.setdefault(sk,{})[r['operation']]=dict(effect=r['clean_margin']-r['source_margin'],norm=r['edit_norm'])
    for c in cells.values():
        for k in ['margin_error','source_kl','source_effect_kl','source_decrement','method_decrement','clean_accuracy','source_accuracy','method_accuracy','clean_margin_error','edit_norm']:c[k]/=c['n']
        c['changed_retention']=c['changed_retained']/c['source_changed'] if c['source_changed'] else None
    rows=list(cells.values())
    def save_csv(name,rr):
        if rr:
            with (out/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    save_csv('cells.csv',rows)
    groups=defaultdict(list)
    for c in rows:groups[c['objective'],c['split'],c['operation'],c['method']].append(c)
    aggregates=[]
    for key,rr in groups.items():
        item=dict(zip(['objective','split','operation','method'],key),cells=len(rr),pairs=sum(r['n'] for r in rr))
        for k in ['margin_error','source_kl','source_effect_kl','source_decrement','method_decrement','clean_accuracy','source_accuracy','method_accuracy','clean_margin_error','edit_norm']:item[k]=statistics.mean(r[k] for r in rr)
        item['source_changed']=sum(r['source_changed'] for r in rr);item['changed_retained']=sum(r['changed_retained'] for r in rr)
        item['changed_retention']=item['changed_retained']/item['source_changed'] if item['source_changed'] else None
        aggregates.append(item)
    save_csv('aggregates.csv',aggregates)
    own={'verb':'regular_plural_subject_verb_agreement_1','anaphor_number':'anaphor_number_agreement','anaphor_gender':'anaphor_gender_agreement'}
    primary=[]
    selections={
        'source_task_components':lambda r:r['split']=='development' and own.get(r['operation'])==r['task'],
        'new_agreement_component':lambda r:r['split']=='new' and r['operation']=='verb',
        'development_joint':lambda r:r['split']=='development' and r['operation']=='all_components',
        'new_joint':lambda r:r['split']=='new' and r['operation']=='all_components'}
    for name,choose in selections.items():
        for obj in ['topk','matryoshka']:
            for method in sorted({r['method'] for r in rows}):
                rr=[r for r in rows if r['objective']==obj and r['method']==method and choose(r)]
                if not rr:continue
                item=dict(consumer=name,objective=obj,method=method,cells=len(rr),pairs=sum(r['n'] for r in rr))
                for k in ['margin_error','source_kl','source_effect_kl','source_decrement','method_decrement','clean_accuracy','source_accuracy','method_accuracy','clean_margin_error','edit_norm']:item[k]=statistics.mean(r[k] for r in rr)
                item['source_changed']=sum(r['source_changed'] for r in rr);item['changed_retained']=sum(r['changed_retained'] for r in rr)
                item['changed_retention']=item['changed_retained']/item['source_changed'] if item['source_changed'] else None
                primary.append(item)
    save_csv('primary.csv',primary)
    composition=defaultdict(list)
    singleton=['verb','anaphor_number','anaphor_gender']
    for key,rr in source.items():
        if all(op in rr for op in singleton+['all_components']):
            composition[key[:4]].append(dict(coactive=sum(rr[k]['norm']>1e-6 for k in singleton),union_effect=rr['all_components']['effect'],sum_singleton_effect=sum(rr[k]['effect'] for k in singleton),nonadditivity=rr['all_components']['effect']-sum(rr[k]['effect'] for k in singleton)))
    comp=[]
    for key,rr in composition.items():
        comp.append(dict(zip(['objective','seed','task','split'],key),n=len(rr),at_least_two_active=sum(r['coactive']>=2 for r in rr)/len(rr),all_three_active=sum(r['coactive']==3 for r in rr)/len(rr),mean_union_effect=statistics.mean(r['union_effect'] for r in rr),mean_sum_singleton_effect=statistics.mean(r['sum_singleton_effect'] for r in rr),mean_nonadditivity=statistics.mean(r['nonadditivity'] for r in rr),mean_absolute_nonadditivity=statistics.mean(abs(r['nonadditivity']) for r in rr)))
    save_csv('composition.csv',comp)
    paths=[run/n for n in ['config.resolved.json','metrics.raw.jsonl','metrics.summary.json','fit_summary.json','source_selection.json','MAP_FREEZE.json','SOURCE_FREEZE.json','gradient_check.json','panel.json','raw_control_selection.json','CONTROL_FREEZE.json']]
    inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths if p.exists()]
    result=dict(run=str(run),status=json.loads((run/'status.json').read_text())['status'],cells=rows,aggregates=aggregates,primary=primary,composition=comp,inputs=inputs,scope='Equal-cell descriptive summaries; five shared cyclic seeds and original pairs are dependent. Source effects, actual changed decisions and coactive composition are distinct from unchanged agreement.')
    if (run/'raw_control_selection.json').exists():
        result['raw_control_selection']=json.loads((run/'raw_control_selection.json').read_text(encoding='utf-8'))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('out');a=ap.parse_args()
    result=analyze(a.run,a.out)
    for r in result['aggregates']:
        if r['operation'] in ['verb','all_components']:
            print(json.dumps({k:r[k] for k in ['objective','split','operation','method','margin_error','source_decrement','source_changed','changed_retention']}))
