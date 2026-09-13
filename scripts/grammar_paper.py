"""Paper views of source effects, actual grammar transfer, and fixed probes."""
from pathlib import Path
import json, statistics, hashlib
from analyze_grammar_transfer import analyze

TASKS = ['regular_plural_subject_verb_agreement_1','anaphor_number_agreement','anaphor_gender_agreement']
SHORT = ['Subject–verb','Anaphor number','Anaphor gender']
METHODS = [('unchanged','No edit'),('assignment_one_scale','Assign. / scale'),('assignment64','Assignment 64'),('group64','Group 64'),('raw_full','Raw full'),('raw_rank1','Raw rank 1'),('raw_rank2','Raw rank 2')]


def export(root, paper):
    manifest=json.loads((paper/'reform_runs.json').read_text())
    run=root/manifest['grammar_run']
    data=analyze(run,paper/'data/grammar_transfer')
    if manifest.get('grammar_initial_run'):
        initial=analyze(root/manifest['grammar_initial_run'],paper/'data/grammar_initial')
        data['initial_aggregates']=initial['aggregates']
        data['probe_summary']=initial['probe_summary'];data['probe_rows']=initial['probe_rows']
        data['inputs']+=initial['inputs']
    for entry in data['inputs']:
        entry['path']=str((root/entry['path']).resolve().relative_to(root)).replace('\\','/')
    (paper/'data/reform_r25.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'& \multicolumn{2}{c}{Margin error (nats)} & \multicolumn{2}{c}{Changed decisions retained (\%)}\\',r'Method & TopK & Matryoshka & TopK & Matryoshka\\',r'\midrule']
    lookup={(r['objective'],r['method']):r for r in data['aggregates']}
    for method,name in METHODS:
        if method=='unchanged':
            vals=[lookup[obj,'source_group']['clean_margin_mae'] for obj in ['topk','matryoshka']]+[0.,0.]
        else:
            vals=[lookup[obj,method]['margin_mae'] for obj in ['topk','matryoshka']]+[100*lookup[obj,method]['changed_decision_agreement'] for obj in ['topk','matryoshka']]
        lines.append(name+' & '+' & '.join(f'{v:.3f}' if i<2 else f'{v:.1f}' for i,v in enumerate(vals))+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);(paper/'tables/grammar_transfer.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(paper):
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    data=json.loads((paper/'data/reform_r25.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.25),sharey=True,sharex=True)
    for ax,obj in zip(axes,['topk','matryoshka']):
        for j,task in enumerate(TASKS):
            rr=sorted([r for r in data['source_selections'] if r['objective']==obj and r['selection_task']==task],key=lambda r:r['source_seed'])
            for i,r in enumerate(rr):
                y=j+(i-2)*.12
                ax.plot([r['dev_mean_margin_decrement'],r['mean_margin_decrement']],[y,y],c='#96938D',lw=.8)
                ax.scatter(r['dev_mean_margin_decrement'],y,marker='o',s=20,facecolors='white',edgecolors='#806E58',zorder=3)
                ax.scatter(r['mean_margin_decrement'],y,marker='s',s=15,c='#22685F',zorder=4)
        ax.set_yticks(range(3),SHORT);ax.set_ylim(2.5,-.55);ax.axvline(0,c='#999999',ls=':',lw=.7)
        ax.set_title('TopK' if obj=='topk' else 'Matryoshka',loc='left',fontsize=10);ax.grid(axis='x',color='#eeeeee',lw=.5)
    fig.text(.57,.95,'Open: 64 source-selection pairs; filled: 936 held pairs; one row per seed',ha='center',fontsize=8)
    fig.supxlabel('Mean grammatical margin loss after source deletion (nats)',x=.58,y=.025,fontsize=9)
    fig.subplots_adjust(left=.18,right=.98,bottom=.17,top=.83,wspace=.16)
    for ext in ['pdf','svg','png']:fig.savefig(paper/f'figures/grammar_source_effects.{ext}',dpi=180)
    plt.close(fig)
    plot_probes(paper,data)
    plot_errors(paper,data)


def plot_probes(paper,data):
    import matplotlib.pyplot as plt
    categories=['contraction_curly','contraction_ascii','quote_curly','subject_pronoun','object_pronoun']
    names=['Curly continuation','ASCII continuation','Closing quotation','Subject pronoun','Object pronoun']
    methods=[('source_group','Source','#222222','o'),('group64','Group 64','#22685F','s'),('assignment64','Assignment 64','#8B7861','D'),('raw_full','Raw full','#3F4E68','x'),('raw_rank1','Raw rank 1','#75465B','^'),('raw_rank2','Raw rank 2','#A06A82','v')]
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.75),sharey=True,sharex=True)
    for ax,query,title in zip(axes,['topk_s1_t2_a1885_c8','topk_s1_t2_a5174_c3'],['Apostrophe hypothesis','Pronoun hypothesis']):
        for j,category in enumerate(categories):
            for i,(method,label,color,marker) in enumerate(methods):
                r=next(r for r in data['probe_summary'] if r['query']==query and r['category']==category and r['method']==method)
                ax.scatter(r['probability_decrement'],j+(i-2.5)*.115,s=16,c=color,marker=marker,label=label if j==0 else None,zorder=4)
        ax.set_yticks(range(5),names);ax.set_ylim(4.5,-.5);ax.axvline(0,c='#999999',ls=':',lw=.7)
        ax.set_xlim(-.009,.16);ax.set_xticks([0,.05,.10,.15])
        ax.set_title(title,loc='left',fontsize=10);ax.grid(axis='x',color='#eeeeee',lw=.5)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False,fontsize=8,handletextpad=.3,columnspacing=1)
    fig.supxlabel('Mean decrease in specified continuation probability',x=.59,y=.035,fontsize=9)
    fig.subplots_adjust(left=.2,right=.99,bottom=.17,top=.78,wspace=.12)
    for ext in ['pdf','svg','png']:fig.savefig(paper/f'figures/grammar_controlled_probes.{ext}',dpi=180)
    plt.close(fig)


def plot_errors(paper,data):
    import numpy as np
    import matplotlib.pyplot as plt
    values=[];labels=[]
    for obj in ['topk','matryoshka']:
        for task,label in zip(TASKS,SHORT):
            vals=[]
            for method,_ in METHODS:
                rr=[r for r in data['query_metrics'] if r['objective']==obj and r['selection_task']==task and r['split']=='positive' and r['method']==('source_group' if method=='unchanged' else method)]
                vals.append(statistics.mean(r['clean_margin_mae' if method=='unchanged' else 'margin_mae'] for r in rr))
            values.append(vals);labels.append(('TopK' if obj=='topk' else 'Matry.')+' / '+label)
    values=np.array(values)
    fig,ax=plt.subplots(figsize=(6.8,3.4));im=ax.imshow(values,cmap='Greys',vmin=0,vmax=values.max(),aspect='auto')
    for row in range(len(values)):
        for col in range(len(METHODS)):
            ax.text(col,row,f'{values[row,col]:.3f}',ha='center',va='center',color='white' if values[row,col]>values.max()*.6 else '#222222',fontsize=8.5)
    ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(len(METHODS)),[n.replace(' ','\n',1) for _,n in METHODS])
    ax.tick_params(length=0);ax.axhline(2.5,c='white',lw=2);ax.set_title('Absolute error in the source-edited grammatical margin (nats)',loc='left',fontsize=10)
    bar=fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
    bar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.29,right=.92,bottom=.19,top=.88)
    for ext in ['pdf','svg','png']:fig.savefig(paper/f'figures/grammar_transfer_errors.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];export(root,root/'paper');plot(root/'paper')
