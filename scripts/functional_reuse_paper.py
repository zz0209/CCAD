"""Export the feature-choice experiment as source-backed tables and figures."""
import hashlib,json
from pathlib import Path
from analyze_functional_reuse import analyze

FAMILIES=[('fcc_members','Membership'),('fcc_functional_priority','Membership + effect'),('assignment','Assignment'),('raw_pullback','Raw coefficient projection'),('cached_gradient_pool','Cached gradient / pool'),('cached_gradient_full','Cached gradient / full')]


def export(root,paper):
    run=root/json.loads((paper/'reform_runs.json').read_text())['functional_reuse_run']
    data=analyze(run,paper/'data/functional_reuse')
    assert data['status']=='PASS'
    paths=[run/n for n in ['config.resolved.json','metrics.raw.jsonl','metrics.summary.json','PROPOSAL_FREEZE.json','SELECTION_FREEZE.json','consumer_results.json','code_hashes.json','inputs.json']]
    data['inputs']=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    lookup={(r['objective'],r['method'],r['validation_pairs_per_task']):r for r in data['aggregates']}
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'&\multicolumn{3}{c}{TopK}&\multicolumn{3}{c}{Matryoshka}\\',r'Member proposal & 0 pairs & 4 pairs & 16 pairs & 0 pairs & 4 pairs & 16 pairs\\',r'\midrule']
    for method,name in FAMILIES:
        values=[lookup[obj,method,b]['selectivity']*100 for obj in ['topk','matryoshka'] for b in [0,4,16]]
        lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/functional_reuse.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'&\multicolumn{2}{c}{TopK}&\multicolumn{2}{c}{Matryoshka}\\',r'Member proposal & Requested & Collateral & Requested & Collateral\\',r'\midrule']
    for method,name in FAMILIES:
        values=[lookup[obj,method,16][k]*100 for obj in ['topk','matryoshka'] for k in ['requested_error_rate','collateral_error_rate']]
        lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/functional_reuse_absolute.tex').write_text('\n'.join(lines)+'\n')
    (paper/'data/final_value_r27.json').write_text(json.dumps(data,indent=2)+'\n')
    return data


def plot(paper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    p=Path('C:/Windows/Fonts/times.ttf')
    if p.exists():font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':'Times New Roman' if p.exists() else 'STIXGeneral','mathtext.fontset':'stix','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads((paper/'data/final_value_r27.json').read_text());lookup={(r['objective'],r['method'],r['validation_pairs_per_task']):r for r in data['aggregates']}
    fig,axes=plt.subplots(1,2,figsize=(7.05,3.15),sharex=True,sharey=True);fig.subplots_adjust(left=.29,right=.98,bottom=.19,top=.80,wspace=.20)
    colors=['#216b57','#216b57','#707070','#76517c','#24465a','#24465a']
    allx=[r['selectivity']*100 for r in data['aggregates']];low=min(0,min(allx));high=max(allx);span=max(high-low,1)
    for col,(ax,obj) in enumerate(zip(axes,['topk','matryoshka'])):
        for y,((method,name),color) in enumerate(zip(FAMILIES,colors)):
            x=[lookup[obj,method,b]['selectivity']*100 for b in [0,16]]
            ax.plot(x,[y,y],color=color,lw=.9)
            ax.scatter(x[0],y,facecolor='white',edgecolor=color,s=28,marker='o',zorder=3)
            ax.scatter(x[1],y,color=color,s=22,marker='s',zorder=4)
        ax.axvline(0,color='#aaaaaa',lw=.7,linestyle=':');ax.set_title(['TopK','Matryoshka'][col]);ax.set_yticks(range(len(FAMILIES)),[v[1] for v in FAMILIES]);ax.invert_yaxis();ax.set_ylim(len(FAMILIES)-.5,-.5);ax.set_xlim(low-.1*span,high+.1*span)
        ax.set_xlabel('Requested − collateral disruption (points)');ax.tick_params(axis='y',length=0)
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],marker='o',mfc='white',mec='#333333',ls='none'),Line2D([],[],marker='s',color='#333333',ls='none')],['No target validation','16 validation pairs per task'],loc='upper center',bbox_to_anchor=(.62,.99),ncol=2,frameon=False,fontsize=8)
    for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'functional_reuse.{ext}',dpi=220)
    plt.close(fig)
