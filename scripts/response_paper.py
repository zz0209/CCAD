"""Export the response-fit experiment and its source-dimension diagnosis."""
from pathlib import Path
import json,hashlib
from analyze_response_groups import analyze


def export(root,paper):
    manifest=json.loads((paper/'reform_runs.json').read_text())
    v1=analyze(root/manifest['pointwise_run'],paper/'data/response_pointwise')
    v2=analyze(root/manifest['finite_run'],paper/'data/response_finite')
    dimension=root/manifest['source_dimension'];dim=json.loads(dimension.read_text())
    result=dict(pointwise=v1,finite=v2,source_dimension=dim,inputs=v1['inputs']+v2['inputs']+[
        dict(path=str(dimension.relative_to(root)),sha256=hashlib.sha256(dimension.read_bytes()).hexdigest())])
    (paper/'data/reform_r23.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'& \multicolumn{2}{c}{Edited position} & \multicolumn{2}{c}{Four tokens later}\\',
        r'Method & TopK & Matryoshka & TopK & Matryoshka\\',r'\midrule']
    for name,label in [('balanced_euclidean','Euclidean'),('balanced_fisher','Pointwise Fisher'),('finite_amplitude','Finite fit: one scale'),('finite_gates','Finite fit: all gates'),('target_group_rank1','Rank-one full-gate projection')]:
        vals=[v2['aggregate'][o]['source_active'][name][metric] for metric in ['relative_kl','future_relative_kl'] for o in ['topk','matryoshka']]
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/response_native_groups.tex').write_text('\n'.join(lines)+'\n')
    return result


def plot(paper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    data=json.loads((paper/'data/reform_r23.json').read_text());rows=data['finite']['queries'];lookup={(r['query'],r['method']):r for r in rows if r['stratum']=='source_active'}
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.8),sharex=True)
    for ax,obj in zip(axes,['topk','matryoshka']):
        qq=sorted([r for r in rows if r['objective']==obj and r['method']=='finite_gates' and r['stratum']=='source_active'],key=lambda r:r['anchor'])
        for y,r in enumerate(qq):
            base=lookup[r['query'],'finite_amplitude'];a=r['relative_kl']-base['relative_kl'];b=r['future_relative_kl']-base['future_relative_kl']
            ax.plot([a,b],[y,y],color='#cccccc',lw=.8)
            ax.scatter(a,y,s=19,color='#22685F',zorder=3)
            ax.scatter(b,y,s=23,marker='D',facecolor='white',edgecolor='#75465B',lw=.9,zorder=3)
        ax.axvline(0,color='#555555',lw=.7);ax.set_yticks(range(len(qq)),[str(r['anchor']) for r in qq]);ax.invert_yaxis()
        ax.set_title('TopK' if obj=='topk' else 'Matryoshka',loc='left',fontsize=10);ax.set_xlabel('Relative KL: full gates minus one scale');ax.set_ylabel('Source anchor');ax.grid(axis='x',color='#ededed',lw=.5)
        ax.tick_params(axis='y',length=0)
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,color='#22685F',label='Edited position',markersize=4),Line2D([],[],marker='D',lw=0,color='#75465B',markerfacecolor='white',label='Four tokens later',markersize=4)],loc='upper center',bbox_to_anchor=(.52,1),ncol=2,frameon=False)
    fig.subplots_adjust(left=.09,right=.99,bottom=.16,top=.86,wspace=.32)
    for ext in ['pdf','svg','png']:fig.savefig(paper/f'figures/response_native_groups.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];export(root,root/'paper');plot(root/'paper')
