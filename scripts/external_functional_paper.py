"""Export the Pythia functional-query study into the single manuscript."""
from pathlib import Path
import json,hashlib,statistics

FAMILIES=[('shared_path_full','Shared response'),('mean_path_full','Mean response'),('shared_scalar_full','Shared scalar'),('cached64_full','Cached credit')]
LABELS=[('1','Determiner only'),('2','Past form only'),('4','Gender only'),('3','Determiner + past'),('5','Determiner + gender'),('6','Past + gender'),('7','All three')]


def export(root,paper):
    run=root/json.loads((paper/'reform_runs.json').read_text())['external_functional_run'];base=root/'artifacts/final_value_five_20260913'
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    data=json.loads((base/'r30_analysis/independent_summary.json').read_text());data['unions']=json.loads((base/'r30_analysis/union_intervals.json').read_text())
    data['material']=json.loads((base/'R30_MATERIAL.json').read_text());cfg=json.loads((run/'config.resolved.json').read_text());data['config']=cfg
    source=root/cfg['path_ensemble_parent'];data['source_diagnostics']=json.loads((source/'source_path_diagnostics.json').read_text())
    paths=[run/n for n in ['config.resolved.json','metrics.raw.jsonl','metrics.summary.json','PROPOSAL_FREEZE.json','SELECTION_FREEZE.json','STRUCTURE_FREEZE.json','consumer_results.json','structure_results.json','code_hashes.json','inputs.json']]
    paths+=[base/n for n in ['r30_analysis/independent_summary.json','r30_analysis/union_intervals.json','R30_MATERIAL.json','R30_PREFLIGHT.json','R30_CONSUMER_FREEZE.json','r30_fresh_grammar/DATA_MANIFEST.json']]
    paths+=[source/n for n in ['source_path_diagnostics.json','config.resolved.json','source_selection.json','metrics.raw.jsonl','status.json']]
    data['inputs']=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (paper/'data/final_value_r30.json').write_text(json.dumps(data,indent=2)+'\n')
    lookup={(r['method'],r['validation_pairs_per_task']):r for r in data['aggregates']}
    allfamilies=[('shared_path_full','Shared response'),('mean_path_full','Mean response'),('single_path_full','Single source'),('shared_scalar_full','Shared scalar'),('cached64_full','Cached64'),('clean32_full','Clean32'),('wrong_shared_path_full','Other-component paths')]
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'&\multicolumn{3}{c}{Selective disruption}&\multicolumn{2}{c}{16-pair decomposition}\\',r'Proposal & 0 pairs & 4 pairs & 16 pairs & Requested & Collateral\\',r'\midrule']
    for family,name in allfamilies:
        vals=[lookup[family,b]['selectivity']*100 for b in [0,4,16]]+[lookup[family,16][k]*100 for k in ['requested_error_rate','collateral_error_rate']]
        lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(paper/'tables/external_choices.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Proposal & Det. + past & Det. + gender & Past + gender & Pair mean & Triple\\',r'\midrule']
    for family,name in FAMILIES:
        rr=[r for r in data['structure_aggregates'] if r['kind']=='union' and r['family']==family]
        vals=[next(r['selectivity'] for r in rr if r['label']==l)*100 for l in ['3','5','6','7']]
        lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in vals[:3]+[statistics.mean(vals[:3]),vals[3]])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(paper/'tables/external_unions.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Pair-mean comparator & Shared gain & Paired 95\% interval\\',r'\midrule']
    for comparator,name in [('cached64_full','Cached64'),('mean_path_full','Mean response'),('shared_scalar_full','Shared scalar')]:
        rr=data['unions']['rows'] if comparator=='cached64_full' else [r for r in data['unions']['additional_contrasts'] if r['comparator']==comparator]
        r=next(r for r in rr if r['requests']==['3','5','6']);ci=r['paired95'];lines.append(f"{name} & {r['gain_points']:.2f} & [{ci[0]:.2f}, {ci[1]:.2f}]"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(paper/'tables/external_union_intervals.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Exact region & Members & Determiner & Past form & Gender\\',r'\midrule']
    for label,name in LABELS:
        r=next(r for r in data['structure_aggregates'] if r['kind']=='region' and r['family']=='shared_path_full' and r['label']==label)
        lines.append(name+f" & {r['mean_members']:.1f} & "+' & '.join(f"{r['task_margin_decrements'][t]:.3f}" for t in cfg['source_tasks'])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(paper/'tables/external_regions.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(paper):
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8,'axes.titlesize':9,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads((paper/'data/final_value_r30.json').read_text());cfg=data['config']
    fig=plt.figure(figsize=(7.05,3.0));left=fig.add_axes([.17,.23,.30,.63]);right=fig.add_axes([.64,.19,.25,.69])
    allvals=[]
    for y,(family,name) in enumerate(FAMILIES):
        rows=[r for r in data['structures'] if r['kind']=='union' and r['family']==family and r['operation'] in ['3','5','6']]
        vals=[statistics.mean(r['selectivity'] for r in rows if r['source']==s)*100 for s in cfg['seeds']];allvals.extend(vals)
        color='#216b57' if family=='shared_path_full' else ('#78517b' if family=='shared_scalar_full' else '#555555')
        left.scatter(vals,y+np.linspace(-.1,.1,len(vals)),s=12,color=color,alpha=.65,linewidths=0)
        left.scatter(statistics.mean(vals),y,marker='D',s=28,facecolor='white',edgecolor=color,zorder=3)
    left.set_yticks(range(len(FAMILIES)),[name for _,name in FAMILIES]);left.set_ylim(3.5,-.5);left.set_xlabel('Pair-request selectivity (points)');left.set_title('Unfitted pair requests');left.tick_params(axis='y',length=0)
    left.set_xlim(np.floor(min(allvals)/5)*5-1,np.ceil(max(allvals)/5)*5+1);left.grid(axis='x',color='#eeeeee',lw=.5)
    for spine in ['top','right','left']:left.spines[spine].set_visible(False)
    rr=[next(r for r in data['structure_aggregates'] if r['kind']=='region' and r['family']=='shared_path_full' and r['label']==label) for label,_ in LABELS]
    m=np.asarray([[r['task_margin_decrements'][t] for t in cfg['source_tasks']] for r in rr]);v=max(.1,float(np.max(np.abs(m))));cmap=LinearSegmentedColormap.from_list('roles',['#78517b','#fbfaf7','#216b57'])
    im=right.imshow(m,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v))
    right.set_yticks(range(7),['D only','P only','G only','D + P','D + G','P + G','D + P + G']);right.set_xticks(range(3),['D','P','G']);right.tick_params(length=0);right.set_title('Shared-response regions')
    for i,r in enumerate(rr):
        for j,value in enumerate(m[i]):right.text(j,i,'—' if not r['nonempty_targets'] else f'{value:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(value)>.68*v else '#222222')
    for spine in right.spines.values():spine.set_visible(False)
    cb=fig.colorbar(im,cax=fig.add_axes([.915,.25,.012,.56]));cb.ax.tick_params(labelsize=7);cb.set_label('Margin decrease (nats)',fontsize=7)
    for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'external_functional_queries.{ext}',dpi=220)
    plt.close(fig)
