"""Export program-preservation results at manuscript size, retaining all methods."""
from pathlib import Path
import argparse,hashlib,json,csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True,type=Path)
    p.add_argument('--paper',type=Path,default=Path('paper'))
    p.add_argument('--stage',choices=['development','confirmation'],required=True)
    a=p.parse_args();d=json.loads(a.analysis.read_text())
    labels={'native':'Original member relation','whole':'Complete-program fit','random_parts':'Random-member fit',
            'parts_relation':'Part fit, fixed dictionary','parts':'Part fit, updated dictionary',
            'geometry':'Decoder geometry','geometry_gain':'Calibrated geometry','raw':'Source-direction readout',
            'raw_reconstruction':'Reconstruction readout'}
    order=[m for m in labels if m in d['results']]
    export=dict(analysis_path=str(a.analysis),analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),stage=a.stage,**d)
    (a.paper/'data/human_program_adaptation.json').write_text(json.dumps(export,indent=2))
    lines=[r'\begin{tabular}{lrrr}\toprule',r'& \multicolumn{2}{c}{Response nRMSE} & Part\\',
           r'Realization & All & Combinations & agreement\\\midrule']
    records=[]
    for m in order:
        row=d['results'][m];v=row['all']['response_nrmse']['mean'];u=row['unseen_combinations']['response_nrmse']['mean'];b=row['parts']['balanced_agreement']['mean']*100
        lines.append(f"{labels[m]} & {v:.3f} & {u:.3f} & {b:.2f}"+r'\\')
        for family in ['all','unseen_combinations']:
            r=row[family]['response_nrmse'];records.append(dict(method=m,family=family,value=r['mean'],lower=r['ci95'][0],upper=r['ci95'][1]))
    lines.append(r'\bottomrule\end{tabular}')
    (a.paper/'tables/human_program_adaptation.tex').write_text('\n'.join(lines)+'\n')
    with (a.paper/'data/human_program_adaptation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5}):
        fig,axes=plt.subplots(1,2,figsize=(6.75,2.65),sharey=True)
        fig.subplots_adjust(left=.31,right=.98,bottom=.20,top=.88,wspace=.22)
        upper=max(r['upper'] for r in records)*1.10
        for ax,fam,title in zip(axes,['all','unseen_combinations'],['(a) All seven requests','(b) Unfitted combinations']):
            for i,m in enumerate(order):
                r=d['results'][m][fam]['response_nrmse'];x=r['mean'];lo,hi=r['ci95']
                color='#216b57' if m=='parts' else ('#78517b' if m.startswith('raw') else '#636363')
                marker='s' if m=='parts' else ('v' if m.startswith('raw') else 'o')
                ax.errorbar(x,len(order)-i-1,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,ms=4,lw=.8,capsize=2)
            ax.set(xlim=(0,upper),ylim=(-.6,len(order)-.4),xlabel='Response nRMSE')
            ax.set_title(title,fontsize=9,pad=8)
            ax.set_yticks(range(len(order)-1,-1,-1),[labels[m] for m in order])
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
            ax.grid(axis='x',color='.92',lw=.5);ax.set_axisbelow(True)
        for ext in ['pdf','svg','png']:fig.savefig(a.paper/('figures/human_program_adaptation.'+ext),dpi=240)
        plt.close(fig)
    (a.paper/'data/human_program_adaptation_figure.json').write_text(json.dumps(dict(stage=a.stage,source=str(a.analysis),
        normalization='RMS target-source logit discrepancy divided by common source-effect RMS; per-target errors averaged.',
        intervals=d['inference'],unseen_combinations=d['families']['unseen_combinations'],
        note='Unfitted refers to semantic-singleton adaptation. The complete-program control fits full requests.',
        dimensions_inches=[6.75,2.65],font='Times New Roman/STIX',all_methods_retained=order),indent=2))


if __name__=='__main__':main()
