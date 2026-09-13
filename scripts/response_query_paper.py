"""Export both the membership-query study and its response-conditioned confirmation."""
from pathlib import Path
import hashlib,json,statistics


def export(root,paper):
    b=root/'artifacts/final_value_five_20260913';reg=json.loads((paper/'reform_runs.json').read_text());inputs=[];studies={}
    for model in ['gpt2','pythia']:
        for stage,key,folder in [('membership','relational_exclusion_runs',f'r31_{model}_analysis'),('response','response_query_runs',f'r31_{model}_response_analysis')]:
            run=root/reg[key][model];assert json.loads((run/'status.json').read_text())['status']=='PASS'
            data=json.loads((b/folder/'summary.json').read_text());data['config']=json.loads((run/'config.resolved.json').read_text());data['run']=run.relative_to(root).as_posix();studies[model+'_'+stage]=data
            for p in [b/folder/'summary.json']+[run/n for n in ['config.resolved.json','QUERY_FREEZE.json','query_results.json','metrics.raw.jsonl','code_hashes.json','inputs.json','status.json']]:inputs.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for name in ['R31_INITIAL_FREEZE.json','R31_RESPONSE_PLAN.json','R31_RESPONSE_FREEZE.json','R31_PREFLIGHT.json','R31_RESPONSE_PREFLIGHT.json','r31_fresh_grammar/DATA_MANIFEST.json','r31_response_fresh_grammar/DATA_MANIFEST.json']:
        p=b/name;inputs.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    data=dict(studies=studies,inputs=inputs);(paper/'data/final_value_r31.json').write_text(json.dumps(data,indent=2)+'\n')
    conditions=[('gpt2','topk','GPT2 TopK'),('gpt2','matryoshka','GPT2 Matryoshka'),('pythia','topk','Pythia TopK')]
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Condition & Query ranking & $U$ & Requested & Preserved & Members\\',r'\midrule']
    for model,obj,name in conditions:
        d=studies[model+'_response']
        for st,label in [('conditional_path','Conditional response'),('conditional_clean','Conditional cached64'),('unconditional','Shared singleton'),('conditional_scalar','Conditional scalar')]:
            x=next(x for x in d['aggregates'] if x['objective']==obj and x['strategy']==st)
            lines.append(f"{name} & {label} & {100*x['selectivity']:.2f} & {100*x['requested_error_rate']:.2f} & {100*x['preserved_error_rate']:.2f} & {x['actual_members']:.1f}"+r'\\')
    lines+=[r'\bottomrule',r'\end{tabular}'];(paper/'tables/response_queries.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Condition & Comparator & Gain & Paired 95\% & Margin gain (nats)\\',r'\midrule']
    for model,obj,name in conditions:
        for x in studies[model+'_response']['contrasts']:
            if x['objective']!=obj:continue
            label={'conditional_clean':'Conditional cached64','unconditional':'Shared singleton','conditional_scalar':'Conditional scalar'}[x['comparator']];lo,hi=x['paired95']
            lines.append(f"{name} & {label} & {x['gain_points']:.2f} & [{lo:.2f},{hi:.2f}] & {x['margin_gain']:.3f}"+r'\\')
    lines+=[r'\bottomrule',r'\end{tabular}'];(paper/'tables/response_query_intervals.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Condition & Relation & Full & Own count-match & Membership subtraction & Set difference\\',r'\midrule']
    for model,obj,name in conditions:
        d=studies[model+'_membership']
        for family,label in [('shared_path_full','Shared'),('mean_path_full','Mean'),('cached64_full','Cached64')]:
            vals=[next(x['selectivity']*100 for x in d['aggregates'] if x['objective']==obj and x['family']==family and x['strategy']==st) for st in ['full','own_matched','soft_matched','difference']]
            lines.append(name+' & '+label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
    lines+=[r'\bottomrule',r'\end{tabular}'];(paper/'tables/membership_exclusion.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Condition & Query ranking & $0\to1$ & $0\to2$ & $1\to0$ & $1\to2$ & $2\to0$ & $2\to1$\\',r'\midrule']
    for model,obj,name in conditions:
        for x in studies[model+'_response']['aggregates']:
            if x['objective']!=obj:continue
            label={'conditional_path':'Conditional response','conditional_clean':'Conditional cached64','unconditional':'Shared singleton','conditional_scalar':'Conditional scalar'}[x['strategy']]
            vals=[100*x['by_query'][q]['selectivity'] for q in ['0>1','0>2','1>0','1>2','2>0','2>1']];lines.append(name+' & '+label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
    lines+=[r'\bottomrule',r'\end{tabular}'];(paper/'tables/response_queries_by_pair.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(paper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8,'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    d=json.loads((paper/'data/final_value_r31.json').read_text())['studies'];conditions=[('gpt2','topk','GPT2 TopK'),('gpt2','matryoshka','GPT2 Matryoshka'),('pythia','topk','Pythia TopK')]
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.65),sharey=True);comparators=[('conditional_clean','vs. conditional cached64'),('unconditional','vs. shared singleton'),('conditional_scalar','vs. conditional scalar')]
    allrows=[x for k,v in d.items() if k.endswith('_response') for x in v['contrasts']];lo=min(x['paired95'][0] for x in allrows);hi=max(x['paired95'][1] for x in allrows);lim=(np.floor(min(lo,0))-1,np.ceil(max(hi,0))+1)
    for ax,(comparator,title) in zip(axes,comparators):
        for y,(model,obj,name) in enumerate(conditions):
            x=next(x for x in d[model+'_response']['contrasts'] if x['objective']==obj and x['comparator']==comparator);color='#216b57' if model=='gpt2' else '#78517b'
            ax.plot(x['paired95'],[y,y],color=color,lw=1.3);ax.scatter(x['gain_points'],y,s=22,color=color,zorder=3)
        ax.axvline(0,color='#777777',lw=.7,ls=':');ax.set_xlim(*lim);ax.set_ylim(2.55,-.55);ax.set_yticks(range(3),[c[2] for c in conditions]);ax.set_title(title);ax.set_xlabel('Selective-query gain (points)');ax.tick_params(axis='y',length=0)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=.17,right=.98,bottom=.26,top=.84,wspace=.17)
    for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'response_queries.{ext}',dpi=220)
    plt.close(fig)
