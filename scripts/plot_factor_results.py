"""Data-backed learning and correspondence figures at manuscript column sizes."""
import argparse,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.aris/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BLUE='#245881';TEAL='#087F82';ORANGE='#BC631F';GRAY='#7B858D';INK='#202C35'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.1,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'savefig.facecolor':'white'})


def save(fig,out,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=260)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    curves=json.loads((out/'functional_learning_curve.json').read_text())['rows'];directions=json.loads((out/'correspondence_directions.json').read_text())['rows'];macro=json.loads((out/'correspondence_macro.json').read_text())['rows'];training=[]
    for k,label in [(64,'five'),(128,'control')]:
        rr=ROOT/f'runs/SEVEN_R2_1b_train_k{k}_{label}_v1_20260906'
        for line in (rr/'metrics.raw.jsonl').read_text().splitlines():training.append(dict(k=k,**json.loads(line)))
    xs=np.array([262144,1048576,4194304])/1e6
    fig,axes=plt.subplots(2,2,figsize=(178/25.4,118/25.4));fig.subplots_adjust(left=.09,right=.98,bottom=.13,top=.94,wspace=.34,hspace=.52)
    a,b,c,d=axes.ravel()
    def envelope(ax,ys,label,color,marker='o'):
        ys=np.array(ys);ax.fill_between(xs,ys.min(0),ys.max(0),color=color,alpha=.12,lw=0);ax.plot(xs,np.mean(ys,axis=0),marker=marker,markersize=3.5,label=label,color=color)
    for k,color,label in [(64,BLUE,'k = 64, five seeds'),(128,ORANGE,'k = 128, seed 1')]:
        seeds=range(1,6) if k==64 else [1];ys=[[next(r['quality']['ce_recovered']*100 for r in training if r['k']==k and r['seed']==seed and r['step']==step) for step in [256,1024,4096]] for seed in seeds];envelope(a,ys,label,color)
        ys=[[next(r['mean_vector_error'] for r in curves if r['checkpoint']==f'k{k}_seed{seed}_step{step}' and r['factor']=='subject' and r['method']=='full_sae_delta' and r['split']=='held_lexical_development') for step in [256,1024,4096]] for seed in seeds];envelope(b,ys,label,color)
    a.set_ylabel('Validation CE recovered (%)');b.set_ylabel('Relative squared vector error');a.legend(frameon=False,fontsize=6.7,loc='lower right')
    for method,label,color,marker in [('full_sae_delta','Whole SAE',GRAY,'s'),('native_16','16 native members',BLUE,'o'),('projected_16','16 members, mean direction',ORANGE,'^')]:
        ys=[[next(r['correct']/r['n']*100 for r in curves if r['checkpoint']==f'k64_seed{seed}_step{step}' and r['factor']=='subject' and r['method']==method and r['split']=='held_lexical_development') for step in [256,1024,4096]] for seed in range(1,6)];envelope(c,ys,label,color,marker)
    raw=next(r for r in curves if r['checkpoint']=='raw' and r['factor']=='subject' and r['split']=='held_lexical_development');c.axhline(raw['correct']/raw['n']*100,color=INK,lw=.7,ls=':');c.set_ylabel('Correct donor transfers (%)');c.legend(frameon=False,fontsize=6.2,loc='lower right')
    for ax,title in [(a,'a  Whole-stream recovery'),(b,'b  Variable-change reconstruction'),(c,'c  Functional learning')]:
        ax.set_title(title,loc='left',fontweight='normal',pad=8);ax.set_xscale('log',base=2);ax.set_xticks(xs,['0.26','1.05','4.19']);ax.set_xlabel('SAE training tokens (millions)');ax.grid(axis='y',alpha=.15,lw=.5)
    rawrows=[json.loads(s) for s in (args.run/'metrics.raw.jsonl').read_text().splitlines()];source_methods=['source_pca_rank1','source_pca_rank4','source_native_teacher']
    for j,method in enumerate(source_methods):
        vals=[]
        for seed in range(1,6):
            rr=[r for r in rawrows if r.get('source_seed')==seed and r.get('target_seed')==seed and r['method']==method and r['factor']=='subject' and r['split']=='held_lexical_development'];vals.append(sum(r['donor_label_correct'] for r in rr)/len(rr)*100)
        d.scatter(j+np.linspace(-.11,.11,5),vals,c=BLUE,s=14,zorder=3);d.plot([j-.18,j+.18],[np.mean(vals)]*2,c=INK,lw=1)
    d.axhline(raw['correct']/raw['n']*100,color=INK,lw=.7,ls=':');d.set_xticks(range(3),['PCA, rank 1','PCA, rank 4','Complete span']);d.set_ylabel('Correct donor transfers (%)');d.set_title('d  Compressing the same 16-member teacher',loc='left',fontweight='normal',pad=8);d.grid(axis='y',alpha=.15,lw=.5)
    save(fig,out,'figure_r2_functional_learning')
    methods=[('raw_ridge','Raw ridge'),('full_code_native_units','Full-code ridge'),('single_atom','Best single atom'),('one_to_one','One-to-one, 16'),('conditional_ot16','Conditional OT, 16'),('dense_native_select16','Dense selection, 16'),('fcc_group16','Group, feature RMS, ≤16'),('fcc_native_units16','Group, native units, ≤16'),('fcc_native_units16_same_support_native','Same members, native'),('fcc_native_units16_same_support_projection','Same members, projected')]
    fig,axes=plt.subplots(1,2,figsize=(178/25.4,104/25.4),sharey=True,sharex=True);fig.subplots_adjust(left=.31,right=.98,bottom=.17,top=.9,wspace=.2)
    for ax,factor,title in zip(axes,['subject','distractor'],['a  Subject-number operation','b  Distractor-number operation']):
        for i,(method,label) in enumerate(methods):
            rr=sorted([r for r in directions if r['method']==method and r['factor']==factor and r['split']=='held_lexical_development'],key=lambda r:(r['source_seed'],r['target_seed']));values=np.array([r['kl_ratio'] for r in rr]);ax.scatter(values,i+np.linspace(-.16,.16,len(values)),s=8,alpha=.5,c=TEAL if method.startswith('fcc') else GRAY,lw=0)
            mr=next(r for r in macro if r['factor']==factor and r['method']==method);ax.scatter(mr['kl_ratio'],i,marker='D',s=18,c=BLUE if method.startswith('fcc') else INK,zorder=4)
        ax.axvline(1,color=ORANGE,ls='--',lw=.8);ax.set_xscale('log');ax.set_title(title,loc='left',fontweight='normal',pad=9);ax.set_yticks(range(len(methods)),[m[1] for m in methods]);ax.set_xlabel('Teacher KL / no-operation KL');ax.grid(axis='x',alpha=.15,lw=.5)
    axes[0].invert_yaxis()
    save(fig,out,'figure_r2_correspondence')
    (out/'figure_r2_data_manifest.json').write_text(json.dumps(dict(training_sources=[f'runs/SEVEN_R2_1b_train_k{k}_{label}_v1_20260906/metrics.raw.jsonl' for k,label in [(64,'five'),(128,'control')]],functional_source=str(args.run/'metrics.raw.jsonl'),summary_tables=['functional_learning_curve.csv','correspondence_directions.csv','correspondence_macro.csv'],learning_shading='min-max across five seeds; not confidence interval',correspondence_points='all 20 dependent directions; diamond pooled KL ratio; 1=no-op; log axes; stronger RMS-raw/native-unit full-code plus same-support native/projection controls foregrounded; complete variants retained in source tables',figure_sizes_mm=[[178,118],[178,104]]),indent=2)+'\n')


if __name__=='__main__':main()
