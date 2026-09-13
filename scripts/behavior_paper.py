"""Export all source-only behavioral groups, paired controls and examples."""
from pathlib import Path
import json,hashlib,statistics,contextlib,io
from collections import defaultdict
from analyze_response_groups import analyze


def export(root,paper):
    import numpy as np
    manifest=json.loads((paper/'reform_runs.json').read_text());run=root/manifest['behavior_run']
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    with contextlib.redirect_stdout(io.StringIO()):data=analyze(run,paper/'data/behavior_groups')
    keys=['assignment_one_scale','assignment_joint_refit','same_budget_group','target_group','target_group_rank1','target_group_rank2','target_pool_ridge','wrong_group_matched_norm','source_anchor']
    lookup={(r['query'],r['method'],r['stratum']):r for r in data['queries']};comparison={};geometry=[];seeds=[]
    for q in json.loads((run/'query_results.json').read_text())['queries']:
        p=run/(q['query']+'_functional_edits.npz')
        with np.load(p) as a:
            x=a['source_group'][a['strata']=='source_active'].astype('float64');sv=np.linalg.svd(x,compute_uv=False)
            geometry.append(dict(query=q['query'],objective=q['objective'],source_seed=q['source_seed'],rank1=float(sv[0]**2/(x*x).sum()),rank2=float((sv[:2]**2).sum()/(x*x).sum())))
        data['inputs'].append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for obj in ['topk','matryoshka']:
        comparison[obj]={}
        for control in ['assignment_joint_refit','assignment_one_scale','target_group_rank1','target_pool_ridge','wrong_group_matched_norm']:
            pairs=[(r,lookup[r['query'],control,'source_active']) for r in data['queries'] if r['objective']==obj and r['method']=='same_budget_group' and r['stratum']=='source_active']
            comparison[obj][control]={metric:dict(wins=sum(a[metric]<b[metric] for a,b in pairs),queries=len(pairs),mean_difference=statistics.mean(a[metric]-b[metric] for a,b in pairs)) for metric in ['relative_kl','future_relative_kl','nll_error']}
        for seed in sorted({r['source_seed'] for r in data['queries']}):
            for method in keys:
                rr=[r for r in data['queries'] if r['objective']==obj and r['source_seed']==seed and r['method']==method and r['stratum']=='source_active']
                seeds.append(dict(objective=obj,source_seed=seed,target_seed=rr[0]['target_seed'],method=method,queries=len(rr),relative_kl=statistics.mean(r['relative_kl'] for r in rr),future_relative_kl=statistics.mean(r['future_relative_kl'] for r in rr)))
    data.update(paired_comparisons=comparison,source_geometry=geometry,seed_summaries=seeds)
    (paper/'data/reform_r24.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'& \multicolumn{2}{c}{Edited position} & \multicolumn{2}{c}{Four tokens later}\\',r'Method & TopK & Matryoshka & TopK & Matryoshka\\',r'\midrule']
    names=[('source_anchor','Source anchor alone'),('assignment_one_scale','Assignment 64: one scale'),('assignment_joint_refit','Assignment 64: joint weights'),('same_budget_group','Group 64: joint weights'),('target_group','Group: up to 256 members'),('target_group_rank1','Group projected to rank 1'),('target_group_rank2','Group projected to rank 2'),('target_pool_ridge','Same-pool raw ridge'),('wrong_group_matched_norm','Wrong group, rescaled if active')]
    for method,label in names:
        vals=[data['aggregate'][obj]['source_active'][method][metric] for metric in ['relative_kl','future_relative_kl'] for obj in ['topk','matryoshka']]
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/behavior_native_groups.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(paper):
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads((paper/'data/reform_r24.json').read_text());fig,axes=plt.subplots(1,2,figsize=(6.8,3.5),sharey=True,sharex=True)
    methods=[('assignment_joint_refit','Assignment 64','#8B7861'),('same_budget_group','Group 64','#22685F'),('target_group_rank1','Group / rank 1','#75465B'),('target_pool_ridge','Raw ridge 256','#3F4E68')]
    for ax,obj in zip(axes,['topk','matryoshka']):
        for i,(method,label,color) in enumerate(methods):
            values=[r['relative_kl'] for r in data['queries'] if r['objective']==obj and r['method']==method and r['stratum']=='source_active']
            # The distribution contains all60 queries; diamonds expose the five
            # dependent source-edge means instead of treating60 as seed repeats.
            ax.boxplot([values],positions=[i],widths=.48,orientation='horizontal',patch_artist=True,showfliers=True,
                boxprops=dict(facecolor=color,alpha=.18,edgecolor=color),medianprops=dict(color=color,lw=1.2),
                whiskerprops=dict(color=color,lw=.7),capprops=dict(color=color,lw=.7),flierprops=dict(marker='.',markersize=2,markerfacecolor=color,markeredgecolor=color))
            for j,r in enumerate([r for r in data['seed_summaries'] if r['objective']==obj and r['method']==method]):
                ax.scatter(r['relative_kl'],i+(j-2)*.065,s=12,marker='D',color=color,zorder=4)
        ax.set_yticks(range(len(methods)),[v[1] for v in methods]);ax.set_ylim(3.55,-.55)
        ax.set_xscale('symlog',linthresh=.01,linscale=.4);ax.set_xlim(0,200)
        ax.set_xticks([0,.01,.1,1,10,100],['0','.01','.1','1','10','100']);ax.axvline(1,color='#888888',ls=':',lw=.7)
        ax.set_xlabel('Relative KL; linear below .01, log above')
        ax.set_title('TopK' if obj=='topk' else 'Matryoshka',loc='left',fontsize=10);ax.grid(axis='x',color='#ededed',lw=.5);ax.tick_params(axis='y',length=0)
    fig.text(.56,.96,'Boxes: all 60 groups per mechanism; diamonds: five shared-seed edge means',ha='center',fontsize=8)
    fig.subplots_adjust(left=.17,right=.99,bottom=.17,top=.86,wspace=.1)
    for ext in ['pdf','svg','png']:fig.savefig(paper/f'figures/behavior_native_groups.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];data=export(root,root/'paper');plot(root/'paper')
    print(json.dumps(dict(aggregate=data['aggregate'],comparisons=data['paired_comparisons']),indent=2))
