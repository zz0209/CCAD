"""Export source-role and cross-seed functional results into the current paper."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/correspondence_reform_20260913';PAPER=ROOT/'paper'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    c=json.loads((ART/'r37_confirmation/confirmation.json').read_text())
    dev=json.loads((ART/'r37_development.json').read_text())
    labels=[('static_source','Constant source'),('role_source','Role source'),('role_field','Role field transfer'),('role_assignment','Role assignment')]
    data=dict(confirmation=c,development=dev,labels=labels)
    dest=PAPER/'data/arithmetic_source_roles.json';dest.write_text(json.dumps(data,indent=2)+'\n')
    def cell(method,op):return next(x for x in c['cells'] if x['initialization']==method and x['operation']==op)
    def mean(method):return 100*np.mean([cell(method,op)['exact_hybrid'] for op in ['unit','tens']])
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(7,2.7),sharey=True)
        fig.subplots_adjust(left=.19,right=.985,bottom=.31,top=.86,wspace=.14)
        categories=['Complete hybrid','Requested only','Preserved only','Neither']
        colors=['#286956','#785481','#b08b5a','#dddddd']
        for ax,op in zip(axes,['unit','tens']):
            for i,(method,label) in enumerate(labels):
                cc=cell(method,op);h,t,p=[cc[k] for k in ['exact_hybrid','target_digit_success','preserve_digit_success']]
                vals=100*np.array([h,t-h,p-h,1-t-p+h]);assert np.all(vals>=-1e-8)
                left=0
                for k,(value,color) in enumerate(zip(vals,colors)):
                    ax.barh(i,value,left=left,height=.58,color=color,edgecolor='white',linewidth=.4,label=categories[k] if i==0 else None)
                    left+=value
                if 100*h>10:ax.text(50*h,i,f'{100*h:.1f}',ha='center',va='center',color='white',fontsize=8)
            ax.set(yticks=range(4),yticklabels=[v for k,v in labels],xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Generated outcomes (%)')
            ax.set_title('Units replacement' if op=='unit' else 'Tens replacement',fontsize=10,fontweight='normal')
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',width=.6,length=3)
        axes[0].invert_yaxis();handles,names=axes[0].get_legend_handles_labels()
        fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.53,.0),ncol=4,frameon=False,handlelength=1.1,columnspacing=1.1,fontsize=8)
        for ext in ['pdf','svg','png']:fig.savefig(PAPER/f'figures/arithmetic_source_roles.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    lines=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrr}',r'\toprule',r'Prespecified contrast & Difference & 95\% interval \\',r'\midrule']
    display={'role_source':'Role source','static_source':'constant source','role_field':'Role field','role_assignment':'assignment'}
    for x in c['primary']:
        lo,hi=x['interval_points'];lines.append(f"{display[x['reference']]} versus {display[x['comparator']]} & {x['difference_points']:.2f} & [{lo:.2f},{hi:.2f}]"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{\textbf{Source-role learning and correspondence on unused questions.} Differences are percentage points of complete hybrid accuracy. Each contrast resamples the same 64 question-pair clusters, retaining both requests, two prompt forms and five fixed SAEs. Source learning uses the same 256-update schedule and a 64-member union in both conditions. Both translation methods use the same frozen role source and 64-member allowance.}',r'\label{tab:arithmetic_source_roles}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_source_roles.tex').write_text('\n'.join(lines)+'\n')
    a,b=c['primary'];alo,ahi=a['interval_points'];blo,bhi=b['interval_points']
    text=(f"Role-dependent source components reach {mean('role_source'):.2f}\\% complete hybrid accuracy,\n"
          f"compared with {mean('static_source'):.2f}\\% for the matched constant components.\n"
          f"The paired improvement is {a['difference_points']:.2f} points ([{alo:.2f},{ahi:.2f}]).\n"
          f"Field transfer reaches {mean('role_field'):.2f}\\%, and one-to-one role assignment reaches\n"
          f"{mean('role_assignment'):.2f}\\%. Their difference is {b['difference_points']:.2f} points\n"
          f"([{blo:.2f},{bhi:.2f}]). The source improvement survives the new questions;\n")
    text += ('both prespecified complete-answer contrasts have positive lower bounds.\n' if c['both_primary_comparisons_positive'] else 'the joint primary criterion is unmet.\n')
    text += r"Figure~\ref{fig:arithmetic_source_roles} separates complete changes from protected-only outcomes; Table~\ref{tab:arithmetic_source_roles} reports the paired contrasts."+'\n'
    (PAPER/'sections/arithmetic_role_findings.tex').write_text(text)
    runs=[];evidence=[]
    for run in sorted((ROOT/'runs').glob('REFORM_R37_*')):
        if not (run/'metrics.summary.json').exists():continue
        status=json.loads((run/'status.json').read_text());summary=json.loads((run/'metrics.summary.json').read_text())
        runs.append(dict(run=run.relative_to(ROOT).as_posix(),status=status,summary=summary))
        evidence += [p.relative_to(ROOT).as_posix() for p in run.iterdir() if p.is_file() and p.suffix in ['.json','.npz','.jsonl']]
    (ART/'R37_RUNS.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),runs=runs),indent=2)+'\n')
    evidence += ['artifacts/correspondence_reform_20260913/'+x for x in ['R37_CONFIRMATION_FREEZE.json','r37_development.json','r37_development_outcomes.npz','r37_confirmation/confirmation.json','r37_confirmation/cluster_outcomes.npz']]
    evidence += ['scripts/'+x for x in ['arithmetic_counterfactual_fit.py','arithmetic_relation_transfer.py','analyze_arithmetic_source_roles.py','analyze_arithmetic_reuse_confirmation.py','arithmetic_source_roles_paper.py','run_arithmetic_digit_components.py']]
    evidence += ['paper/'+x for x in ['data/arithmetic_source_roles.json','sections/arithmetic_role_learning.tex','sections/arithmetic_role_findings.tex','tables/arithmetic_source_roles.tex','figures/arithmetic_source_roles.pdf']]
    (PAPER/'data/arithmetic_source_roles_evidence.json').write_text(json.dumps(dict(id='role_conditioned_source_functions',paper='Source role participation and cross-seed functional reuse',result=c['primary'],scope=c['scope'],evidence=evidence),indent=2)+'\n')
    fp=PAPER/'figures/FIGURE_MANIFEST.json';fm=json.loads(fp.read_text());paths=['figures/arithmetic_source_roles.'+x for x in ['pdf','svg','png']]
    fm['outputs']=[x for x in fm['outputs'] if x['path'] not in paths]+[dict(path=p,bytes=(PAPER/p).stat().st_size,sha256=hashlib.sha256((PAPER/p).read_bytes()).hexdigest()) for p in paths]
    fm.setdefault('additional_sources',{})['arithmetic_source_roles']=dict(path=dest.relative_to(PAPER).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator='scripts/arithmetic_source_roles_paper.py')
    fp.write_text(json.dumps(fm,indent=2)+'\n')
    print(json.dumps(dict(means={m:mean(m) for m,l in labels},primary=c['primary'])))

if __name__=='__main__':main()
