"""A fixed held lexical example with its actual signed relation and LM outputs."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.aris/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
import numpy as np
from plot_factor_results import ROOT,BLUE,TEAL,ORANGE,GRAY,INK,save


def main():
    out=ROOT/'artifacts/seven_round_rebuild_20260906';data=json.loads((out/'r2_relation_example.json').read_text());matrix=np.array(data['relation'])
    fig=plt.figure(figsize=(178/25.4,115/25.4));gs=fig.add_gridspec(2,2,height_ratios=[.28,1],width_ratios=[1.0,1.08],left=.09,right=.985,top=.945,bottom=.17,hspace=.34,wspace=.70)
    a=fig.add_subplot(gs[0,:]);a.axis('off');a.set_title('a  A subject-number change on held lexical material',loc='left',fontweight='normal',pad=8)
    a.text(0,.68,'Recipient',fontsize=7,color=GRAY);a.text(.16,.68,r'The $\mathbf{actor}$ near the painter',fontsize=8.5,color=INK)
    a.text(0,.19,'Donor',fontsize=7,color=GRAY);a.text(.16,.19,r'The $\mathbf{actors}$ near the painter',fontsize=8.5,color=INK)
    a.text(.71,.68,'Layer 3, subject position',fontsize=7,color=INK)
    a.text(.71,.19,f"Seed 1: {len(data['source_members'])} members\nSeed 2: {len(data['target_members'])} members",fontsize=7,color=INK,linespacing=1.6,va='center')
    b=fig.add_subplot(gs[1,0]);v=float(np.max(np.abs(matrix)));cmap=LinearSegmentedColormap.from_list('signed_operation',[ORANGE,'#FAFAFA',BLUE]);im=b.imshow(matrix,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v),interpolation='nearest')
    b.set_yticks(range(len(data['source_members'])),[str(i) for i in data['source_members']],fontsize=6.3);b.set_xticks(range(len(data['target_members'])),[str(i) for i in data['target_members']],rotation=90,fontsize=6.3);b.tick_params(length=0,pad=3);b.set_ylabel('Source feature ID');b.set_xlabel('Target feature ID');b.set_title('b  Signed contribution relation',loc='left',fontweight='normal',pad=8)
    cb=fig.colorbar(im,ax=b,orientation='horizontal',pad=.24,fraction=.035,aspect=26);cb.set_label('Contribution inner product',fontsize=6.5);cb.ax.tick_params(labelsize=6)
    c=fig.add_subplot(gs[1,1]);order=[('noop_teacher_reference','Unedited'),('source_native_teacher','Source teacher'),('fcc_native_units16','Group correspondence'),('fcc_native_units16_same_support_native','Same members, native'),('one_to_one','One-to-one'),('single_atom','Best single atom'),('raw_native_units','Raw ridge')];teacher=next(r['plural_margin'] for r in data['metrics'] if r['method']=='source_native_teacher')
    for i,(method,label) in enumerate(order):
        row=next(r for r in data['metrics'] if r['method']==method);value=row['plural_margin'];color=BLUE if method=='source_native_teacher' else TEAL if method=='fcc_native_units16' else GRAY;c.scatter(value,i,s=25 if method in ['source_native_teacher','fcc_native_units16'] else 17,c=color,zorder=3);c.text(value+.24,i,f'{value:.2f}',va='center',fontsize=6.7,color=color)
    c.axvline(0,c=INK,lw=.6);c.axvline(teacher,c=BLUE,lw=.65,ls=':');c.set_xlim(-7,5.1);c.set_ylim(6.6,-.6);c.set_yticks(range(len(order)),[label for _,label in order]);c.set_xticks([-6,-3,0,3]);c.set_xlabel('log P(are) − log P(is)');c.set_title('c  Actual continuation preference',loc='left',fontweight='normal',pad=8);c.grid(axis='x',alpha=.15,lw=.5)
    save(fig,out,'figure_r2_relation_example')


if __name__=='__main__':main()
