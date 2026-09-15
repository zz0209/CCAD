"""Export the binding confirmation, all controls and a real factorial example."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/correspondence_reform_20260913';P=ROOT/'paper'


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    x=json.loads((ART/'r41_binding_query_confirmation.json').read_text())
    old=json.loads((ART/'r41_binding_transfer.json').read_text())
    cells={(r['method'],r['operation']):r for r in x['macro']}
    supplemental={}
    for name in ['r41_binding_capacity_control.json','r41_binding_matched_additive.json']:
        if (ART/name).exists():
            supplemental[name]=json.loads((ART/name).read_text())
            cells.update({(r['method'],r['operation']):r for r in supplemental[name]['macro']})
    run=ROOT/x['run'];panel=json.loads((run/'panel.json').read_text())
    raw=[json.loads(r) for r in (run/'metrics.raw.jsonl').read_text().splitlines()]
    rr=[r for r in raw if r['kind']=='intervention']
    lookup={(r['component'],r['task'],r['seed'],r['method'],r['operation'],r['query']):r for r in rr}
    eligible=[]
    for c in panel['contexts']:
        if c['split']=='fit':continue
        for seed in [1,2,3,4,5]:
            if all(lookup[c['context'],'template0',seed,m,o,q]['correct'] for m in ['country64','union_member_country'] for o in ['first','second','both'] for q in [0,1]):eligible.append((c,seed))
    c,seed=eligible[0];es=c['names'];places=c['places']
    labels={'sae64':'Source: fixed 64','country64':'Source: country 64','sae256':'Source: all 256',
        'member_country':'Member fields','query_member_country':'Query fields','union_member_country':'Shared participation',
        'assignment_country':'Assignment','target_country':'Direct target selection','raw_readout_country':'Raw readout',
        'code_readout_country':'Code readout','raw':'Raw replacement','member_parent':'Target parent 256',
        'member_static':'Member: fixed request','assignment_static':'Assignment: fixed request',
        'query_member_bank':'Query fields: bank 256','union_member_bank':'Shared: bank 256','union_wrong_query':'Shared: wrong request',
        'adam_capacity_country':'Additive: projected Adam','matched_additive_country':'Additive: matched penalty'}
    order=['sae64','country64','sae256','member_country','matched_additive_country','union_member_country',
           'union_wrong_query','assignment_country','target_country','raw_readout_country','code_readout_country','raw']
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8.5,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,3.05));ax=fig.add_axes([.015,.12,.37,.80]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
        ax.text(0,1,'Replacing one binding, then both',fontsize=10,va='top')
        for j in range(2):ax.text(0,.86-j*.095,f'{es[j]}: {places[j][0]} → {places[j+2][0]}')
        ax.text(0,.59,'Request',fontsize=8.5);ax.text(.58,.59,es[0],ha='center');ax.text(.89,.59,es[1],ha='center')
        ax.plot([0,1],[.55,.55],color='#555555',lw=.6)
        answerrows=[]
        for i,op in enumerate(['first','second','both']):
            ans=[lookup[c['context'],'template0',seed,'union_member_country',op,q]['answer'].strip() for q in [0,1]]
            answerrows.append(dict(operation=op,answers=ans))
            y=.45-.125*i;ax.text(0,y,['First person','Second person','Both'][i]);ax.text(.58,y,ans[0],ha='center');ax.text(.89,y,ans[1],ha='center')
        ax.text(0,.02,f'Source SAE {seed} and target SAE {seed%5+1}\ngive the same requested city pair.',fontsize=8)
        chart=fig.add_axes([.67,.15,.305,.76])
        for i,m in enumerate(order):
            r=cells[m,'macro'];v=100*r['complete_binding_accuracy'];lo,hi=r['interval_points'];color='#286956' if m in ['country64','union_member_country'] else '#555555'
            chart.plot([lo,hi],[i,i],color=color,lw=1)
            chart.plot(v,i,'o' if m in ['country64','union_member_country'] else 's',color=color,ms=3.7)
            chart.text(102,i,f'{v:.1f}',va='center',fontsize=7.6)
        chart.set(xlim=(0,104),ylim=(len(order)-.4,-.7),yticks=range(len(order)),yticklabels=[labels[m] for m in order],xticks=[0,50,100],xlabel='Complete binding accuracy (%)')
        chart.tick_params(axis='y',length=0,labelsize=8,pad=5);chart.spines[['top','right','left']].set_visible(False)
        chart.grid(axis='x',color='#dddddd',lw=.5);chart.set_axisbelow(True)
        for ext in ['pdf','svg','png']:fig.savefig(P/f'figures/binding_components.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & First & Second & Both & Macro \\',r'\midrule']
    for m in list(labels):
        if (m,'macro') in cells:table.append(labels[m]+' & '+' & '.join(f"{100*cells[m,o]['complete_binding_accuracy']:.2f}" for o in ['first','second','both','macro'])+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{All binding methods on the second frozen panel. Complete success requires both entity answers. Each request includes 64 contexts, two forms and five fixed SAE pairs, except raw replacement, which is shared across pairs.}',r'\label{tab:binding_complete}',r'\end{anchoredtable}']
    (P/'tables/binding_components.tex').write_text('\n'.join(table)+'\n')
    contrasts=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{llrr}',r'\toprule',r'Method & Reference & Difference & 95\% interval \\',r'\midrule']
    chosen=[]
    for r in x['macro_contrasts']:
        if r['operation']!='macro':continue
        if (r['method']=='country64' and r['reference'] in ['sae64','sae256']) or (r['method']=='union_member_country' and r['reference'] in ['member_country','query_member_country','assignment_country','target_country','raw_readout_country','code_readout_country']):
            lo,hi=r['interval_points'];chosen.append(r);contrasts.append(f"{labels[r['method']]} & {labels[r['reference']]} & {r['difference_points']:+.2f} & [{lo:.2f},{hi:.2f}]"+r' \\')
    if (ART/'r41_matched_additive_contrast.json').exists():
        matched=json.loads((ART/'r41_matched_additive_contrast.json').read_text());matched.update(method='union_member_country',reference='matched_additive_country',operation='macro')
        chosen.append(matched);lo,hi=matched['interval_points']
        contrasts.append(f"Shared participation & Additive: matched penalty & {matched['difference_points']:+.2f} & [{lo:.2f},{hi:.2f}]"+r' \\')
    contrasts += [r'\bottomrule\end{tabular}',r'\caption{Paired complete-binding differences in percentage points. Ten thousand context resamples retain both forms, all requests and the dependent five-SAE cohort. The matched-penalty comparison is a supplemental control on the same panel.}',r'\label{tab:binding_contrasts}',r'\end{anchoredtable}']
    (P/'tables/binding_contrasts.tex').write_text('\n'.join(contrasts)+'\n')
    preserved=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & First: modify & First: preserve & Second: modify & Second: preserve \\',r'\midrule']
    for m in order:
        preserved.append(labels[m]+' & '+' & '.join(f"{100*cells[m,o][k]:.2f}" for o in ['first','second'] for k in ['requested_accuracy','preserved_accuracy'])+r' \\')
    preserved += [r'\bottomrule\end{tabular}',r'\caption{Requested and preserved entity answers separately. All incorrect unedited answers remain in the denominator.}',r'\label{tab:binding_preserve}',r'\end{anchoredtable}']
    (P/'tables/binding_preserve.tex').write_text('\n'.join(preserved)+'\n')
    quality_run=ROOT/'runs/REFORM_R32_qwen_topk_five_seed_16m_v1_20260914'
    quality=[json.loads(line) for line in (quality_run/'metrics.raw.jsonl').read_text().splitlines()]
    quality=[r for r in quality if 'fve' in r]
    qt=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{rrrrrr}',r'\toprule',r'Tokens & Seed & FVE (\%) & CE recovery (\%) & $L_0$ & Alive / dead \\',r'\midrule']
    for r in quality:qt.append(f"{r['tokens']:,} & {r['seed']} & {100*r['fve']:.4f} & {100*r['ce_recovered']:.3f} & {r['l0']:.3f} & {r['alive']} / {r['dead']}"+r' \\')
    qt += [r'\bottomrule\end{tabular}',r'\caption{Retained natural-validation checkpoints for the five layer13 SAEs. All use the same ordered training stream. Decoder norm maximum errors are below $2.4\times10^{-7}$.}',r'\label{tab:binding_sae_quality}',r'\end{anchoredtable}']
    (P/'tables/binding_sae_quality.tex').write_text('\n'.join(qt)+'\n')
    example=dict(context=c,source_seed=seed,target_seed=seed%5+1,answers=answerrows,eligible=len(eligible),selection='First in context/seed order where source and shared-participation target both implement all three requests in prompt form0. Descriptive illustration; all cases remain in aggregate.')
    dest=P/'data/binding_components.json';dest.write_text(json.dumps(dict(confirmation=x,original_confirmation=old,supplemental=supplemental,example=example,contrasts=chosen),indent=2)+'\n')
    fp=P/'figures/FIGURE_MANIFEST.json';fm=json.loads(fp.read_text());paths=['figures/binding_components.'+e for e in ['pdf','svg','png']]
    fm['outputs']=[r for r in fm['outputs'] if r['path'] not in paths]+[dict(path=p,bytes=(P/p).stat().st_size,sha256=hashlib.sha256((P/p).read_bytes()).hexdigest()) for p in paths]
    fm.setdefault('additional_sources',{})['binding_components']=dict(path=dest.relative_to(P).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator='scripts/binding_components_paper.py');fp.write_text(json.dumps(fm,indent=2)+'\n')
    evidence=dict(id='semantic_binding_member_requests',paper='Known semantic requests and shared target participation',scope=x['scope'],
        evidence=[str((ART/f).relative_to(ROOT).as_posix()) for f in ['R41_BINDING_TRANSFER_FREEZE_v3.json','R41_QUERY_CONFIRMATION_FREEZE.json','r41_binding_transfer.json','r41_binding_query_confirmation.json']],
        raw_run=x['run'],result=chosen,example=example)
    evidence['evidence'] += [x['run']+'/'+name for name in ['metrics.raw.jsonl','config.resolved.json','code_hashes.json','panel.json','binding_relations.jsonl','status.json']]
    evidence['evidence'] += [str((quality_run/name).relative_to(ROOT).as_posix()) for name in ['metrics.raw.jsonl','config.resolved.json','checkpoints.json']]
    evidence['evidence'] += ['scripts/'+name for name in ['run_binding_components.py','binding_correspondence.py','analyze_binding_components.py','binding_components_paper.py']]
    for name,z in supplemental.items():
        evidence['evidence'] += ['artifacts/correspondence_reform_20260913/'+name]
        evidence['evidence'] += [z['run']+'/'+p for p in ['metrics.raw.jsonl','config.resolved.json','code_hashes.json','status.json']]
    (P/'data/binding_components_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(example=example,contrasts=chosen)))
    if (ART/'r42_binding_native_confirmation.json').exists():
        from binding_execution_paper import main as execution_paper
        execution_paper()


if __name__=='__main__':main()
