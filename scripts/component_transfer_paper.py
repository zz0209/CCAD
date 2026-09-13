"""Views of contrast-defined component effects and their actual reuse."""
import json,statistics
from pathlib import Path
from analyze_component_transfer import analyze

METHODS=[('partition64','Partition 64'),('independent64_clipped','Independent / clip'),('assignment64','Assignment 64'),('raw_full','Raw full'),('raw_rank2','Raw rank 2'),('wrong_partition','Wrong component')]
TASKS=['regular_plural_subject_verb_agreement_1','anaphor_number_agreement','anaphor_gender_agreement']
OPS=['verb','anaphor_number','anaphor_gender']
NAMES=['Subject–verb','Anaphor number','Anaphor gender']
CONSUMERS=['source_task_components','new_agreement_component','development_joint','new_joint']


def export(root,paper):
    run=root/json.loads((paper/'reform_runs.json').read_text())['component_transfer_run']
    data=analyze(run,paper/'data/component_transfer')
    manifest=json.loads((paper/'reform_runs.json').read_text())
    if manifest.get('component_raw_control_run'):
        control=analyze(root/manifest['component_raw_control_run'],paper/'data/component_raw_controls')
        data['raw_controls']=control;data['inputs']+=control['inputs']
    for item in data['inputs']:item['path']=(root/Path(item['path'])).resolve().relative_to(root).as_posix()
    (paper/'data/reform_r26.json').write_text(json.dumps(data,indent=2)+'\n')
    lookup={(r['consumer'],r['objective'],r['method']):r for r in data['primary']}
    lines=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r'& \multicolumn{4}{c}{TopK} & \multicolumn{4}{c}{Matryoshka}\\',r'& Source tasks & New agreement & Dev. joint & New joint & Source tasks & New agreement & Dev. joint & New joint\\',r'\midrule']
    for method,name in [('source','No edit')]+METHODS:
        vals=[lookup[consumer,obj,method]['clean_margin_error' if method=='source' else 'margin_error'] for obj in ['topk','matryoshka'] for consumer in CONSUMERS]
        lines.append(name+' & '+' & '.join(f'{v:.3f}' for v in vals)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (paper/'tables/component_transfer.tex').write_text('\n'.join(lines)+'\n')
    if data.get('raw_controls'):
        control={(r['consumer'],r['objective'],r['method']):r for r in data['raw_controls']['primary']}
        lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'& \multicolumn{2}{c}{TopK} & \multicolumn{2}{c}{Matryoshka}\\',r'Method & Verb MAE & Joint MAE & Verb MAE & Joint MAE\\',r'\midrule']
        for method,name in [('partition64','Native partition'),('full_ridge_0.01','Raw ridge .01'),('full_ridge_0.1','Raw ridge .1'),('full_ridge_1','Raw ridge 1'),('rank8_ridge_0.001','Raw rank 8'),('selected_raw','Raw chosen on old development')]:
            source=lookup if method=='partition64' else control
            values=[source[c,obj,method]['margin_error'] for obj in ['topk','matryoshka'] for c in ['new_agreement_component','new_joint']]
            lines.append(name+' & '+' & '.join(f'{v:.3f}' for v in values)+r'\\')
        lines += [r'\bottomrule',r'\end{tabular}'];(paper/'tables/component_raw_controls.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(paper):
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,colors
    times=Path('C:/Windows/Fonts/times.ttf')
    if times.exists():font_manager.fontManager.addfont(str(times))
    plt.rcParams.update({'font.family':'Times New Roman' if times.exists() else 'STIXGeneral','mathtext.fontset':'stix','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads((paper/'data/reform_r26.json').read_text());rows=data['cells']
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(paper/'figures'/f'{name}.{ext}',dpi=220)
        plt.close(fig)
    arrays=[]
    for obj in ['topk','matryoshka']:
        for method in ['source','partition64']:
            arrays.append(np.array([[statistics.mean(r['method_decrement'] for r in rows if r['objective']==obj and r['method']==method and r['task']==task and r['operation']==op) for op in OPS] for task in TASKS]))
    limit=max(abs(a).max() for a in arrays);cmap=colors.LinearSegmentedColormap.from_list('signed_effect',['#24465a','#ffffff','#813b50'])
    fig,axes=plt.subplots(1,4,figsize=(7.05,2.7));fig.subplots_adjust(left=.15,right=.91,bottom=.28,top=.82,wspace=.12)
    for ix,(ax,a) in enumerate(zip(axes,arrays)):
        im=ax.imshow(a,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto');ax.set_xticks(range(3),['Verb','Number','Gender'],rotation=45,ha='right');ax.set_yticks(range(3),NAMES if ix==0 else [])
        ax.set_title(('TopK' if ix<2 else 'Matryoshka')+'\n'+('Source' if ix%2==0 else 'Transferred'))
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(a[i,j])>.63*limit else '#111111')
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0)
    ca=fig.add_axes([.93,.28,.014,.54]);fig.colorbar(im,cax=ca);ca.set_ylabel('Mean margin decrease (nats)',fontsize=8)
    save(fig,'component_effects')
    lookup={(r['consumer'],r['objective'],r['method']):r for r in data['primary']}
    fig,axes=plt.subplots(1,2,figsize=(7.05,3.1));fig.subplots_adjust(left=.20,right=.94,bottom=.25,top=.88,wspace=.25)
    names=['Source tasks','New agreement','Dev. joint','New joint']
    display=[('source','No edit')]+METHODS
    values=[]
    for obj in ['topk','matryoshka']:
        values.append(np.array([[lookup[c,obj,m]['clean_margin_error' if m=='source' else 'margin_error'] for c in CONSUMERS] for m,_ in display]))
    vmax=max(a.max() for a in values)
    for ix,(ax,a) in enumerate(zip(axes,values)):
        im=ax.imshow(a,cmap='Greys',vmin=0,vmax=vmax,aspect='auto');ax.set_xticks(range(4),names,rotation=25,ha='right');ax.set_yticks(range(len(display)),[v[1] for v in display] if ix==0 else []);ax.set_title(['TopK','Matryoshka'][ix]);ax.tick_params(length=0)
        for i in range(len(display)):
            for j in range(4):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]>.52*vmax else 'black')
        for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.20,.98,'Error relative to the same source operation (nats; lower is better)',fontsize=9,va='top')
    save(fig,'component_transfer_errors')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];export(root,root/'paper');plot(root/'paper')
