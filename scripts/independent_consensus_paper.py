"""Publication tables and figures for the independent functional consumer."""
import hashlib,json
from pathlib import Path

FAMILIES=[('shared_path_full','Shared source response'),('mean_path_full','Mean source response'),('single_path_full','Single-source response'),('shared_scalar_full','Shared scalar response'),('clean32_full','Clean credit / 32 pairs'),('cached64_full','Cached credit / 64 pairs'),('wrong_shared_path_full','Other-component paths')]
LABELS=[('1','Verb only'),('2','Number only'),('4','Gender only'),('3','Verb + number'),('5','Verb + gender'),('6','Number + gender'),('7','All three')]


def export(root,paper):
    run=root/json.loads((paper/'reform_runs.json').read_text())['independent_consensus_run']
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    base=root/'artifacts/final_value_five_20260913'
    summary=base/'r29_analysis/independent_summary.json';data=json.loads(summary.read_text())
    sourcepaths=[run/x for x in ['config.resolved.json','metrics.raw.jsonl','metrics.summary.json','PROPOSAL_FREEZE.json','SELECTION_FREEZE.json','STRUCTURE_FREEZE.json','consumer_results.json','structure_results.json','code_hashes.json','inputs.json']]
    unionpath=base/'r29_analysis/union_intervals.json'
    data['unions']=json.loads(unionpath.read_text())
    sourcepaths.extend([summary,unionpath,base/'R29_FINAL_FREEZE.json',base/'R29_PREFLIGHT.json',base/'R29_MATERIAL.json',base/'R29_ANALYSIS_SPEC.json',base/'r29_fresh_grammar/DATA_MANIFEST.json'])
    data['inputs']=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sourcepaths]
    (paper/'data/final_value_r29.json').write_text(json.dumps(data,indent=2)+'\n')
    lookup={(r['objective'],r['method'],r['validation_pairs_per_task']):r for r in data['aggregates']}
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'&\multicolumn{3}{c}{TopK}&\multicolumn{3}{c}{Matryoshka}\\',r'Member proposal & 0 pairs & 4 pairs & 16 pairs & 0 pairs & 4 pairs & 16 pairs\\',r'\midrule']
    for method,label in FAMILIES:
        values=[lookup[o,method,b]['selectivity']*100 for o in ['topk','matryoshka'] for b in [0,4,16]]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/independent_consensus.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'&\multicolumn{2}{c}{TopK}&\multicolumn{2}{c}{Matryoshka}\\',r'Member proposal & Requested & Collateral & Requested & Collateral\\',r'\midrule']
    for method,label in FAMILIES:
        values=[lookup[o,method,16][k]*100 for o in ['topk','matryoshka'] for k in ['requested_error_rate','collateral_error_rate']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/independent_consensus_absolute.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrr}',r'\toprule',r'SAE & Comparator & Gain (points) & Paired 95\% interval\\',r'\midrule']
    for r in data['contrasts']:
        if r['method']!='shared_path_full':continue
        label='Cached64' if r['comparator']=='cached64_full' else 'Shared scalar';ci=r['paired_target_and_within_grammar_percentile95']
        lines.append(('TopK' if r['objective']=='topk' else 'Matryoshka')+f" & {label} & {r['gain_points']:.2f} & [{ci[0]:.2f}, {ci[1]:.2f}]"+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/independent_consensus_intervals.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'SAE & Function & Shared & Mean & Scalar & Cached64\\',r'\midrule']
    for obj in ['topk','matryoshka']:
        for op,label in [('verb','Verb'),('anaphor_number','Anaphor number'),('anaphor_gender','Anaphor gender')]:
            vals=[next(r['selectivity'] for r in data['by_function'] if r['objective']==obj and r['operation']==op and r['method']==m)*100 for m in ['shared_path_full','mean_path_full','shared_scalar_full','cached64_full']]
            lines.append(('TopK' if obj=='topk' else 'Matryoshka')+' & '+label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/independent_consensus_by_function.tex').write_text('\n'.join(lines)+'\n')
    mat=json.loads((base/'R29_MATERIAL.json').read_text());lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Cohort & SAE & FVE & CE recovery & $L_0$ & Alive & Dead\\',r'\midrule']
    for r in mat['aggregates']:
        vals=[r[k] for k in ['fve','ce_recovered','l0','alive','dead']]
        lines.append(r['cohort'].title()+' & '+('TopK' if r['objective']=='topk' else 'Matryoshka')+' & '+' & '.join([f'{vals[0]:.4f}',f'{vals[1]:.4f}']+[f'{v:.1f}' for v in vals[2:]])+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/independent_material.tex').write_text('\n'.join(lines)+'\n')
    names={'3':'Verb + number','5':'Verb + gender','6':'Number + gender','7':'All three','3,5,6':'Mean of pair requests'}
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'SAE & Request & Shared & Cached64 & Gain [paired 95\%]\\',r'\midrule']
    for r in data['unions']['rows']:
        label=names[','.join(r['requests'])];ci=r['paired95']
        lines.append(('TopK' if r['objective']=='topk' else 'Matryoshka')+f" & {label} & {r['shared_points']:.2f} & {r['cached_points']:.2f} & {r['gain_points']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/functional_unions.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'SAE & Region & Members & Verb & Number & Gender\\',r'\midrule']
    tasks=json.loads((run/'config.resolved.json').read_text())['source_tasks']
    for obj in ['topk','matryoshka']:
        for label,name in LABELS:
            r=next(r for r in data['structure_aggregates'] if r['kind']=='region' and r['objective']==obj and r['family']=='shared_path_full' and r['label']==label)
            lines.append(('TopK' if obj=='topk' else 'Matryoshka')+' & '+name+f" & {r['mean_members']:.1f} & "+' & '.join(f"{r['task_margin_decrements'][t]:.3f}" for t in tasks)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/functional_regions.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(paper):
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/times.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads((paper/'data/final_value_r29.json').read_text())
    cfg=json.loads((paper.parent/json.loads((paper/'reform_runs.json').read_text())['independent_consensus_run']/'config.resolved.json').read_text())
    selected=json.loads((paper.parent/'artifacts/final_value_five_20260913/r29_analysis/summary.json').read_text())['selected']
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.12),sharex=True,sharey=True);fig.subplots_adjust(left=.28,right=.98,bottom=.17,top=.89,wspace=.16)
    for ax,obj,title in zip(axs,['topk','matryoshka'],['TopK','Matryoshka']):
        for y,(family,label) in enumerate(FAMILIES):
            rr=[r for r in selected if r['objective']==obj and r['method']==family and r['validation_pairs_per_task']==16]
            vals=[np.mean([r['selectivity'] for r in rr if r['seed']==s])*100 for s in cfg['seeds']]
            color='#216b57' if family=='shared_path_full' else ('#78517b' if family=='shared_scalar_full' else '#545454')
            ax.scatter(vals,y+np.linspace(-.13,.13,len(vals)),s=13,color=color,alpha=.55,linewidths=0)
            ax.scatter(np.mean(vals),y,s=30,marker='D',facecolor='white',edgecolor=color,linewidth=.9,zorder=5)
        ax.set_title(title);ax.set_yticks(range(len(FAMILIES)),[v[1] for v in FAMILIES]);ax.set_ylim(len(FAMILIES)-.5,-.5);ax.set_xlim(10,28)
        ax.set_xlabel('Requested − collateral disruption (points)');ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#ededed',lw=.5)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'independent_consensus.{ext}',dpi=220)
    plt.close(fig)
    task=cfg['source_tasks'];vals=[v for r in data['structure_aggregates'] if r['kind']=='region' for v in r['task_margin_decrements'].values()]
    cmap=LinearSegmentedColormap.from_list('functional_effect',['#78517b','#fbfaf7','#216b57']);norm=TwoSlopeNorm(vmin=-max(abs(v) for v in vals),vcenter=0,vmax=max(abs(v) for v in vals))
    fig,axs=plt.subplots(2,2,figsize=(7.05,4.95));fig.subplots_adjust(left=.21,right=.86,bottom=.09,top=.93,hspace=.32,wspace=.30)
    for row,obj in enumerate(['topk','matryoshka']):
        for col,family in enumerate(['shared_path_full','cached64_full']):
            ax=axs[row,col];rr=[next(r for r in data['structure_aggregates'] if r['objective']==obj and r['family']==family and r['kind']=='region' and r['label']==label) for label,_ in LABELS]
            matrix=np.asarray([[r['task_margin_decrements'][t] for t in task] for r in rr]);im=ax.imshow(matrix,cmap=cmap,norm=norm,aspect='auto')
            for i,r in enumerate(rr):
                for j,v in enumerate(matrix[i]):
                    ax.text(j,i,'—' if r['nonempty_targets']==0 else f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if v>max(vals)*.65 else '#202020')
            ax.set_xticks(range(3),['Verb','Number','Gender']);ax.set_yticks(range(len(LABELS)),[label for _,label in LABELS] if col==0 else []);ax.tick_params(length=0)
            ax.set_title(('TopK' if obj=='topk' else 'Matryoshka')+' / '+('Shared response' if col==0 else 'Cached credit'),fontsize=9)
            for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.89,.24,.015,.52]);cb=fig.colorbar(im,cax=cax);cb.set_label('Margin decrease (nats)',fontsize=8)
    for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'functional_regions.{ext}',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.50),sharey=True);fig.subplots_adjust(left=.22,right=.98,bottom=.24,top=.86,wspace=.16)
    querysets=[['3'],['5'],['6'],['3','5','6']]
    qlabels=['Verb + number','Verb + gender','Number + gender','Mean of pair requests']
    for ax,obj,title in zip(axs,['topk','matryoshka'],['TopK','Matryoshka']):
        for y,query in enumerate(querysets):
            r=next(r for r in data['unions']['rows'] if r['objective']==obj and r['requests']==query);ci=r['paired95'];color='#216b57' if len(query)>1 else '#545454'
            ax.plot(ci,[y,y],color=color,lw=1.2);ax.scatter(r['gain_points'],y,marker='D' if len(query)>1 else 'o',s=23,facecolor='white',edgecolor=color,zorder=3)
        ax.axvline(0,color='#888888',lw=.65,ls=':');ax.set_yticks(range(4),qlabels);ax.set_ylim(3.5,-.5);ax.set_xlim(-2,10);ax.set_title(title);ax.set_xlabel('Gain over cached credit (points)');ax.tick_params(axis='y',length=0)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'functional_unions.{ext}',dpi=220)
    plt.close(fig)
