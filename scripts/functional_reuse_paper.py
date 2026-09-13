"""Export the feature-choice experiment as source-backed tables and figures."""
import hashlib,json
from pathlib import Path
from analyze_functional_reuse import analyze

FAMILIES=[('fcc_members','Membership'),('fcc_functional_priority','Membership + effect'),('assignment','Assignment'),('raw_pullback','Raw coefficient projection'),('cached_gradient_pool','Cached gradient / pool'),('cached_gradient_full','Cached gradient / full')]
PATH_FAMILIES=[('response_membership','Finite-response membership'),('response_clean_membership','Clean-response membership'),('source_path_gradient','Source-path credit'),('wrong_path_gradient','Wrong-path credit'),('clean_gradient_matched','Clean credit / 32 pairs'),('cached_gradient_pool','Cached credit / 64 pairs'),('fcc_members','Contribution membership')]
CONSENSUS_FAMILIES=[('consensus_mean_credit','Mean source credit'),('consensus_robust_credit','Worst-source credit'),('consensus_scale_credit','Scalar-only source credit'),('cached_gradient_pool','Cached credit / 64 pairs')]


def export(root,paper,*,key='functional_reuse_run',stem='functional_reuse',tag='final_value_r27',families=FAMILIES):
    run=root/json.loads((paper/'reform_runs.json').read_text())[key]
    data=analyze(run,paper/'data'/stem)
    assert data['status']=='PASS'
    paths=[run/n for n in ['config.resolved.json','metrics.raw.jsonl','metrics.summary.json','PROPOSAL_FREEZE.json','SELECTION_FREEZE.json','consumer_results.json','code_hashes.json','inputs.json']]
    if key=='functional_path_run':
        paths.extend(sorted(run.glob('*_path_relation.npz')));paths.extend(sorted(run.glob('*_path_fit.json')))
        data['path_fits']=[dict(file=p.name,**json.loads(p.read_text())) for p in sorted(run.glob('*_path_fit.json'))]
    if key=='functional_consensus_run':paths.extend(sorted(run.glob('*_consensus.npz')))
    data['inputs']=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    lookup={(r['objective'],r['method'],r['validation_pairs_per_task']):r for r in data['aggregates']}
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'&\multicolumn{3}{c}{TopK}&\multicolumn{3}{c}{Matryoshka}\\',r'Member proposal & 0 pairs & 4 pairs & 16 pairs & 0 pairs & 4 pairs & 16 pairs\\',r'\midrule']
    for method,name in families:
        values=[lookup[obj,method,b]['selectivity']*100 for obj in ['topk','matryoshka'] for b in [0,4,16]]
        lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables'/f'{stem}.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'&\multicolumn{2}{c}{TopK}&\multicolumn{2}{c}{Matryoshka}\\',r'Member proposal & Requested & Collateral & Requested & Collateral\\',r'\midrule']
    for method,name in families:
        values=[lookup[obj,method,16][k]*100 for obj in ['topk','matryoshka'] for k in ['requested_error_rate','collateral_error_rate']]
        lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables'/f'{stem}_absolute.tex').write_text('\n'.join(lines)+'\n')
    if key=='functional_path_run':
        import statistics
        methods=['source_path_gradient','clean_gradient_matched','cached_gradient_pool','wrong_path_gradient','response_membership']
        lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'SAE & Function & Path & Clean32 & Cached64 & Other path & Response fit\\',r'\midrule']
        for obj in ['topk','matryoshka']:
            for op,label in [('verb','Verb'),('anaphor_number','Anaphor number'),('anaphor_gender','Anaphor gender')]:
                values=[statistics.mean(r['selectivity'] for r in data['selected'] if r['objective']==obj and r['operation']==op and r['method']==m and r['validation_pairs_per_task']==16)*100 for m in methods]
                lines.append(('TopK' if obj=='topk' else 'Matryoshka')+' & '+label+' & '+' & '.join(f'{v:.2f}' for v in values)+r'\\')
        lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables'/f'{stem}_by_function.tex').write_text('\n'.join(lines)+'\n')
    (paper/'data'/f'{tag}.json').write_text(json.dumps(data,indent=2)+'\n')
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


def plot_path(paper):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    p=Path('C:/Windows/Fonts/times.ttf')
    if p.exists():font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':'Times New Roman' if p.exists() else 'STIXGeneral','mathtext.fontset':'stix','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads((paper/'data/final_value_r28.json').read_text())
    plot_families=PATH_FAMILIES
    if (paper/'data/final_value_r28_consensus.json').exists():
        other=json.loads((paper/'data/final_value_r28_consensus.json').read_text())
        data['aggregates']+= [r for r in other['aggregates'] if r['method']!='cached_gradient_pool']
        plot_families=[('consensus_robust_credit','Shared source credit'),('consensus_mean_credit','Mean source credit'),('source_path_gradient','Single-source path credit'),('consensus_scale_credit','Scalar-only mean credit'),('clean_gradient_matched','Clean credit / 32 pairs'),('cached_gradient_pool','Cached credit / 64 pairs'),('fcc_members','Contribution membership')]
    lookup={(r['objective'],r['method'],r['validation_pairs_per_task']):r for r in data['aggregates']}
    blocks=[]
    for obj in ['topk','matryoshka']:
        blocks.append(np.array([[lookup[obj,method,b]['selectivity']*100 for b in [0,4,16]] for method,_ in plot_families]))
    lo=min(0,min(float(b.min()) for b in blocks));hi=max(float(b.max()) for b in blocks)
    fig,axes=plt.subplots(1,2,figsize=(7.05,3.1),sharey=True);fig.subplots_adjust(left=.30,right=.90,bottom=.17,top=.90,wspace=.15)
    for i,(ax,block) in enumerate(zip(axes,blocks)):
        im=ax.imshow(block,cmap='Greens',vmin=lo,vmax=hi,aspect='auto')
        for y in range(len(plot_families)):
            for x in range(3):ax.text(x,y,f'{block[y,x]:.1f}',ha='center',va='center',color='white' if block[y,x]>(lo+hi)/2 else '#222222',fontsize=8)
        ax.set_xticks(range(3),['0','4','16']);ax.set_yticks(range(len(plot_families)),[n for _,n in plot_families]);ax.set_title(['TopK','Matryoshka'][i]);ax.set_xlabel('Validation pairs per task');ax.tick_params(length=0)
        for sp in ax.spines.values():sp.set_visible(False)
    ca=fig.add_axes([.925,.19,.014,.63]);cb=fig.colorbar(im,cax=ca);cb.set_label('Selectivity (points)',fontsize=8);cb.ax.tick_params(labelsize=7)
    for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'functional_path.{ext}',dpi=220)
    plt.close(fig)
