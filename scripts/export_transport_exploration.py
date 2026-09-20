"""Export fixed-support confirmation and its paired-seed research figure."""
from pathlib import Path
import json,csv,sys,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/transport_exploration_20260919'
sys.path.insert(0,str(ROOT/'.aris/plot_runtime_v1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    d=json.loads((OUT/'REFINEMENT_CONFIRMATION.json').read_text())
    shutil.copy2(OUT/'REFINEMENT_CONFIRMATION.json',ROOT/'paper/data/transport_refinement.json')
    table=[r'\begin{tabular}{lrr}',r'\toprule',r'Execution & Human three-member parts & Grammatical parts \\',r'\midrule']
    choices=[('Current active members','initial_active','native_tangent_relation_8'),('Initial common support','initial_open','transport_open_initial'),('Refined same support','refined_initial_open','transport_refined_open'),('Reconstruction readout','raw_reconstruction','raw_reconstruction')]
    rows=[]
    for label,h,g in choices:
        a=d['human']['summary'][h];b=d['grammar']['summary'][g]
        table.append(f"{label} & {a['nrmse']:.3f} & {b['nrmse']:.3f}"+r' \\')
        for study,m,v in [('human',h,a),('grammar',g,b)]:rows.append(dict(study=study,method=m,nrmse=v['nrmse'],lower=v['interval'][0],upper=v['interval'][1],by_seed=json.dumps(v['by_seed'])))
    table += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper/tables/transport_refinement.tex').write_text('\n'.join(table)+'\n')
    with (OUT/'CONFIRMATION_TABLE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','font.size':9,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.65))
    for ax,study,title in zip(axs,['human','grammar'],['Human parts / 6 target members','Grammatical parts / 8 target members']):
        s=d[study];a=s['summary'][s['reference']]['by_seed'];b=s['summary'][s['primary']]['by_seed']
        for seed,(x,y) in enumerate(zip(a,b),1):
            color='#286956' if y<x else '#82485d'
            ax.plot([0,1],[x,y],color=color,lw=.9,alpha=.85)
            ax.scatter([0,1],[x,y],color=color,s=14,zorder=3)
            dy=(-3 if seed==2 else 3 if seed==3 else 0) if study=='grammar' else 0
            ax.annotate(str(seed),(1,y),xytext=(5,dy),textcoords='offset points',va='center',fontsize=7)
        ax.set_xticks([0,1],['Initial coefficients','Refined coefficients']);ax.set_xlim(-.16,1.24)
        ax.set_ylim(bottom=0,top=max(a+b)*1.15);ax.set_title(title,fontsize=9)
        ax.set_ylabel('Response error (nRMSE)');ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y',color='#dddddd',linewidth=.45)
    fig.subplots_adjust(left=.085,right=.98,top=.84,bottom=.22,wspace=.35)
    fig.savefig(OUT/'fixed_support_refinement.pdf');fig.savefig(OUT/'fixed_support_refinement.png',dpi=220)


if __name__=='__main__':main()
