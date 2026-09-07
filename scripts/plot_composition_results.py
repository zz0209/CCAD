"""R3 source-backed comparison matrix and a fixed compositional mechanism case."""
import argparse,json
from pathlib import Path
import numpy as np
from plot_factor_results import plt,save,BLUE,TEAL,ORANGE,GRAY,INK
from matplotlib.colors import LogNorm,LinearSegmentedColormap
from matplotlib.patches import Rectangle


METHODS=[('fcc_group','FCC, compact signed map'),('dense_select','Dense selection + refit'),('full_code_ridge','Full-code ridge'),('rrr_rank4','Reduced-rank ridge, r = 4'),('rrr_rank1','Reduced-rank ridge, r = 1'),('dsca_rank1','Conditional dSCA, r = 1'),('raw_native_units','Raw ridge, common units'),('raw_feature_rms','Raw ridge, feature RMS'),('same_members_native','FCC members, native'),('same_members_native_gain','FCC members, native + gain'),('same_members_projected','FCC members, projected'),('direct_target_native','Direct target selection'),('direct_target_native_gain','Direct target selection + gain'),('full_native','Whole target SAE, native'),('one_to_one','One-to-one assignment'),('single_atom','Best single atom'),('conditional_ot','Conditional OT'),('random_refit','Random members + refit'),('wrong_factor','Wrong factor / position')]


def plot_comparison(summary,out,behavior=None):
    rows=summary['rows']+(behavior['rows'] if behavior else []);sources=summary['source_rows'];methods=METHODS.copy()
    if behavior:
        for after,item in [('dsca_rank1',('das_style_raw_rank1','DAS-style raw interchange, r = 1')),('same_members_native_gain',('same_members_behavior_gain','FCC members, behavioral gain')),('direct_target_native_gain',('direct_target_behavior_gain','Direct target, behavioral gain'))]:
            index=next(i for i,m in enumerate(methods) if m[0]==after);methods.insert(index+1,item)
        methods=[(m,label.replace('native + gain','native + geometric gain').replace('selection + gain','selection + geometric gain')) for m,label in methods]
    fig=plt.figure(figsize=(178/25.4,143/25.4));gs=fig.add_gridspec(2,2,height_ratios=[1.05,len(methods)],left=.345,right=.985,bottom=.165,top=.915,hspace=.10,wspace=.15)
    cmap=LinearSegmentedColormap.from_list('error',['#f3f8fa','#b9dbe4','#549bad','#1d536e','#182837']);norm=LogNorm(vmin=1e-4,vmax=2)
    for col,part in enumerate(['known_cue','new_cue']):
        header=fig.add_subplot(gs[0,col]);header.set_xlim(-.5,4.5);header.set_ylim(-.5,.5);header.axis('off')
        vals=[]
        for seed in range(1,6):
            r=next(r for r in sources if r['factor']=='joint' and r['detail']==['seed_partition',seed,part]);vals.append(r['correct']/r['n']*100)
        for x,value in enumerate(vals):header.text(x,0,f'{value:.0f}%',ha='center',va='center',fontsize=7,color=INK)
        header.set_title('a  New nouns, known time cues' if col==0 else 'b  New nouns, new time cues',loc='left',fontsize=8,pad=15)
        if col==0:header.text(-.8,0,'Source joint transfer',ha='right',va='center',fontsize=7,color=GRAY)
        ax=fig.add_subplot(gs[1,col]);matrix=np.empty((len(methods),5))
        for i,(method,label) in enumerate(methods):
            for seed in range(1,6):
                detail='source_partition' if method=='das_style_raw_rank1' else 'edge_partition'
                cell=[r for r in rows if r['factor']=='joint' and r['method']==method and r['detail'][0]==detail and r['detail'][1]==seed and r['detail'][-1]==part]
                matrix[i,seed-1]=sum(r['kl_sum'] for r in cell)/sum(r['noop_kl_sum'] for r in cell)
        im=ax.imshow(matrix,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
        for i in range(len(methods)):
            for j in range(5):
                value=matrix[i,j];label=f'{value:.4f}' if value<.01 else f'{value:.3f}' if value<1 else f'{value:.2f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=5.9,color='white' if value>.09 else INK)
        ax.set_xticks(range(5),['1','2','3','4','5']);ax.set_xlabel('Source seed; four targets (DAS: one raw fit)',fontsize=6.3);ax.set_yticks(range(len(methods)),[label for _,label in methods] if col==0 else ['']*len(methods));ax.tick_params(length=0,pad=5,labelsize=6.7)
        for name in ['fcc_group','dsca_rank1','raw_feature_rms','direct_target_behavior_gain' if behavior else 'direct_target_native_gain','full_native']:
            boundary=next(i for i,m in enumerate(methods) if m[0]==name)+.5;ax.axhline(boundary,color='white',lw=1.8)
        ax.add_patch(Rectangle((-.49,-.49),4.98,.98,fill=False,edgecolor=TEAL,lw=1.2))
        for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.50,.062,.34,.014]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[.0001,.001,.01,.1,1]);cb.ax.set_xticklabels(['0.0001','0.001','0.01','0.1','1']);cb.set_label('Joint teacher KL / no-operation KL  ·  lower is better',fontsize=7,labelpad=5);cb.ax.tick_params(labelsize=6,length=2)
    save(fig,out,'figure_r3_comparison')


def plot_example(data,out):
    fig=plt.figure(figsize=(178/25.4,132/25.4));gs=fig.add_gridspec(2,2,height_ratios=[.31,1],width_ratios=[1.22,1],left=.08,right=.98,bottom=.12,top=.955,hspace=.31,wspace=.39)
    a=fig.add_subplot(gs[0,:]);a.axis('off');a.set_title('a  One fixed new-noun example and two intervention positions',loc='left',fontsize=8,pad=10)
    a.text(0,.72,'Right now,',color=ORANGE,fontsize=9);a.text(.15,.72,'the artist near the sailor',color=INK,fontsize=9)
    a.annotate('Back then',xy=(.065,.57),xytext=(.065,.13),ha='center',va='center',fontsize=8,color=ORANGE,arrowprops=dict(arrowstyle='-|>',lw=.8,color=ORANGE))
    a.annotate('artists',xy=(.236,.57),xytext=(.236,.13),ha='center',va='center',fontsize=8,color=BLUE,arrowprops=dict(arrowstyle='-|>',lw=.8,color=BLUE))
    a.text(.49,.73,'Source seed 1  →  target seed 2',fontsize=8)
    for y,factor,label,color in [(.40,'number','Number',BLUE),(.08,'time','Time',ORANGE)]:
        r=next(r for r in data['factors'] if r['factor']==factor);a.text(.49,y,f"{label}: {len(r['source_members'])} source → {len(r['target_members'])} target members",fontsize=7.3,color=color)
    bgrid=gs[1,0].subgridspec(2,2,wspace=.25,hspace=.40);baseline=next(r for r in data['raw_and_baseline'] if r['kind']=='baseline');heat=LinearSegmentedColormap.from_list('probability',['#f7f9fa','#9abdd1',BLUE]);order=['number','time','joint'];last=None
    for k,(method,label) in enumerate([('source_teacher','Source operations'),('fcc_group','FCC operations'),('same_members_native','FCC members, native'),('direct_target_native','Direct target groups')]):
        ax=fig.add_subplot(bgrid[k//2,k%2]);rr=[baseline]+[next(r for r in data['metrics'] if r['method']==method and r['factor']==factor) for factor in order];lp=np.array([r['label_logprobs'] for r in rr]);prob=np.exp(lp-lp.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);last=ax.imshow(prob,aspect='auto',vmin=0,vmax=1,cmap=heat)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{prob[i,j]*100:.0f}',ha='center',va='center',fontsize=6.4,color='white' if prob[i,j]>.65 else INK)
            ax.add_patch(Rectangle((np.argmax(prob[i])-.47,i-.47),.94,.94,fill=False,edgecolor=INK,lw=.65))
        ax.set_xticks(range(4),['is','are','was','were']);ax.set_yticks(range(4),['Base','N','T','N + T'] if k%2==0 else ['']*4);ax.tick_params(length=0,labelsize=6.7,pad=3);ax.set_title(('b  ' if k==0 else '')+label,fontsize=7.2,loc='left',pad=7)
        for spine in ax.spines.values():spine.set_visible(False)
    c=fig.add_subplot(gs[1,1]);r=next(r for r in data['factors'] if r['factor']=='time');all_points=[]
    for index,item in enumerate(r['cue_vectors']):
        q=np.array(item['source_pc']);qh=np.array(item['fcc_pc']);color=[BLUE,ORANGE,TEAL,GRAY][index];all_points.extend([q,qh]);c.annotate('',xy=q,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=1.1,color=color));c.plot([0,qh[0]],[0,qh[1]],color=color,lw=1,ls='--');c.scatter(*qh,s=24,marker='o',facecolors='white',edgecolors=color,lw=.9,zorder=3)
        c.text(.02,.97-index*.09,item['label'],transform=c.transAxes,color=color,fontsize=6.5,va='top')
    c.axhline(0,color='#c9d0d4',lw=.5,zorder=0);c.axvline(0,color='#c9d0d4',lw=.5,zorder=0);points=np.array(all_points);lo=points.min(0);hi=points.max(0);span=np.maximum(hi-lo,1);c.set_xlim(min(0,lo[0])-span[0]*.16,max(0,hi[0])+span[0]*.2);c.set_ylim(min(0,lo[1])-span[1]*.18,max(0,hi[1])+span[1]*.85)
    c.set_title('c  Time changes depend on the cue',fontsize=8,loc='left',pad=10);c.set_xlabel('Source-coordinate PC 1',fontsize=7);c.set_ylabel('Source-coordinate PC 2',fontsize=7);c.tick_params(labelsize=6.5);c.text(.03,.03,'Solid arrow: source\nDashed line / circle: FCC',transform=c.transAxes,fontsize=6.5,color=GRAY,va='bottom')
    fig.text(.08,.032,'Cells: probability (%) among the four displayed continuations. N + T uses the two separately fitted maps.',fontsize=6.8,color=GRAY)
    save(fig,out,'figure_r3_composition_example')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--behavior',action='store_true');args=ap.parse_args();summary=json.loads((args.out/'R3_COMPOSITION_SUMMARY.json').read_text());example=json.loads((args.out/'R3_FIXED_EXAMPLE.json').read_text());behavior=json.loads((args.out/'R3_BEHAVIOR_SUMMARY.json').read_text()) if args.behavior else None
    assert summary['completed_pairs']==20,'Main comparison figure requires all 20 completed pairs'
    if behavior:assert behavior['completed_gain_pairs']==20
    plot_comparison(summary,args.out,behavior);plot_example(example,args.out)
    (args.out/'R3_FIGURE_DATA.json').write_text(json.dumps(dict(sources=['R3_COMPOSITION_SUMMARY.json','R3_FIXED_EXAMPLE.json']+(['R3_BEHAVIOR_SUMMARY.json'] if behavior else []),comparison='Pooled teacher/noop KL for each source seed and four target seeds,19main plus3behaviorcontrols when enabled, separate known/new cues. DAS has one fitted raw direction per source/factor, no target fit; its cell is not four independent observations. Top row is source joint expected-label transfer, not fidelity. No independent-edge uncertainty.',example='Fixed source1/target2/first new noun, actual full-precision measured outputs; four-label probabilities renormalized for display. Time PC basis fitted on source discovery, new-cue projection can leave energy outside the plane.',sizes_mm=[[178,143],[178,132]]),indent=2)+'\n')


if __name__=='__main__':main()
