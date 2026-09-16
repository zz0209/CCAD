"""Export program-preservation results at manuscript size, retaining all methods."""
from pathlib import Path
import argparse,hashlib,json,csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True,type=Path)
    p.add_argument('--paper',type=Path,default=Path('paper'))
    p.add_argument('--stage',choices=['development','confirmation'],required=True)
    p.add_argument('--consumer-analysis',type=Path)
    p.add_argument('--material-run',type=Path)
    a=p.parse_args();d=json.loads(a.analysis.read_text())
    consumer=json.loads(a.consumer_analysis.read_text()) if a.consumer_analysis else None
    labels={'native':'Original member relation','whole':'Complete-program fit','random_parts':'Random-member fit',
            'parts_relation':'Part fit, fixed dictionary','parts':'Part fit, updated dictionary',
            'geometry':'Decoder geometry','geometry_gain':'Calibrated geometry','raw':'Source-direction readout',
            'raw_reconstruction':'Reconstruction readout'}
    order=[m for m in ['geometry','geometry_gain','native','raw','raw_reconstruction','whole','random_parts','parts_relation','parts'] if m in d['results']]
    export=dict(analysis_path=str(a.analysis),analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),stage=a.stage,**d)
    (a.paper/'data/human_program_adaptation.json').write_text(json.dumps(export,indent=2))
    lines=[r'\begin{tabular}{lrrrr}\toprule',r'& \multicolumn{3}{c}{Response nRMSE} & Part\\',
           r'Realization & All & Parts & Combinations & agreement\\\midrule']
    records=[]
    for m in order:
        row=d['results'][m];v=row['all']['response_nrmse']['mean'];u=row['unseen_combinations']['response_nrmse']['mean'];b=row['parts']['balanced_agreement']['mean']*100
        part=row['parts']['response_nrmse']['mean']
        lines.append(f"{labels[m]} & {v:.3f} & {part:.3f} & {u:.3f} & {b:.2f}"+r'\\')
        for family in ['all','parts','unseen_combinations']:
            r=row[family]['response_nrmse'];records.append(dict(method=m,family=family,value=r['mean'],lower=r['ci95'][0],upper=r['ci95'][1]))
    lines.append(r'\bottomrule\end{tabular}')
    (a.paper/'tables/human_program_adaptation.tex').write_text('\n'.join(lines)+'\n')
    with (a.paper/'data/human_program_adaptation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    if consumer:
        assert set(order)<=set(consumer['results'])
        (a.paper/'data/human_program_new_tasks.json').write_text(json.dumps(dict(analysis_path=str(a.consumer_analysis),analysis_sha256=hashlib.sha256(a.consumer_analysis.read_bytes()).hexdigest(),**consumer),indent=2))
        all_labels=dict(none='Unedited',source='Source explanation',**labels)
        lines=[r'\begin{tabular}{lrrrrr}\toprule',r'& \multicolumn{2}{c}{Full program} & \multicolumn{3}{c}{Parts}\\',r'Realization & Accuracy & Worst group & Accuracy & Worst group & Agreement\\\midrule']
        for m in ['none','source']+order:
            r=consumer['results'][m]
            values=[100*r[q][k]['mean'] for q,k in [('full','profession'),('full','worst_group'),('parts_mean','profession'),('parts_mean','worst_group'),('parts_mean','balanced_source_agreement')]]
            formatted=[f'{v:.2f}' for v in values]
            if m=='none':formatted[-1]='--'
            lines.append(all_labels[m]+' & '+' & '.join(formatted)+r'\\')
        lines.append(r'\bottomrule\end{tabular}')
        (a.paper/'tables/human_program_new_tasks.tex').write_text('\n'.join(lines)+'\n')
    if a.material_run:
        rows=[json.loads(s) for s in (a.material_run/'metrics.raw.jsonl').read_text().splitlines()]
        assert len(rows)==48 and {r['seed'] for r in rows}=={2,3,4,5}
        material_labels=dict(original='Original dictionary',whole=labels['whole'],random_parts=labels['random_parts'],parts=labels['parts'])
        lines=[r'\begin{tabular}{llrrrr}\toprule',r'Site & Training & FVE (\%) & CE recovery (\%) & $L_0$ & Alive / 8192\\\midrule']
        for site in ['embed','mlp_0','resid_0']:
            for m in material_labels:
                cells=[r for r in rows if r['site']==site and r['method']==m]
                assert len(cells)==4
                values=[]
                for k,scale,dec in [('fve',100,2),('ce_recovery',100,2),('l0',1,2),('alive',1,0)]:
                    v=[scale*r[k] for r in cells]
                    values.append(f'{sum(v)/4:.{dec}f} [{min(v):.{dec}f},{max(v):.{dec}f}]')
                lines.append(site.replace('_',r'\_')+' & '+material_labels[m]+' & '+' & '.join(values)+r'\\')
        lines.append(r'\bottomrule\end{tabular}')
        (a.paper/'tables/human_program_material.tex').write_text('\n'.join(lines)+'\n')
        (a.paper/'data/human_program_material.json').write_text(json.dumps(dict(run=str(a.material_run),raw_sha256=hashlib.sha256((a.material_run/'metrics.raw.jsonl').read_bytes()).hexdigest(),rows=rows),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8,'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5}):
        height=3.0 if len(order)>7 else 2.65
        fig,axes=plt.subplots(1,3 if consumer else 2,figsize=(6.75,height),sharey=True)
        fig.subplots_adjust(left=.31,right=.99,bottom=.18,top=.88,wspace=.30)
        upper=max(r['upper'] for r in records)*1.10
        panels=[('parts','(a) Semantic parts'),('unseen_combinations','(b) Combinations'),('consumer','(c) New-task use')] if consumer else [('all','(a) All seven requests'),('unseen_combinations','(b) Unfitted combinations')]
        for ax,(fam,title) in zip(axes,panels):
            for i,m in enumerate(order):
                r=consumer['results'][m]['parts_mean']['profession'] if fam=='consumer' else d['results'][m][fam]['response_nrmse']
                scale=100 if fam=='consumer' else 1
                x=scale*r['mean'];lo,hi=[scale*v for v in r['ci95']]
                color='#216b57' if m=='parts' else ('#78517b' if m.startswith('raw') else '#636363')
                marker='s' if m=='parts' else ('v' if m.startswith('raw') else 'o')
                ax.errorbar(x,len(order)-i-1,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,ms=4,lw=.8,capsize=2)
            if fam=='consumer':
                bounds=[100*v for m in order for v in consumer['results'][m]['parts_mean']['profession']['ci95']]
                ref=100*consumer['results']['source']['parts_mean']['profession']['mean']
                ax.set_xlim(min(bounds+[ref])-2,max(bounds+[ref])+2)
                ax.axvline(ref,color='.45',lw=.7,ls=':')
                ax.set_xlabel('Part accuracy (%)')
            else:
                bound=max(d['results'][m][fam]['response_nrmse']['ci95'][1] for m in order)*1.08
                ax.set(xlim=(0,bound),xlabel='Response nRMSE')
            ax.set_ylim(-.6,len(order)-.4)
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
        consumer_analysis=str(a.consumer_analysis) if consumer else None,
        dimensions_inches=[6.75,height],font='Times New Roman/STIX',all_methods_retained=order),indent=2))


if __name__=='__main__':main()
