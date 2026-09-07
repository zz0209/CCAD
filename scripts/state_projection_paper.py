"""Export retained finite-loss/state-projection results for the manuscript."""
import json,math
from pathlib import Path

BASE_METHODS=[('aggregate_ols','Shared OLS'),('dense_half','Dense half'),('matched_atoms','One-to-one'),('aggregate_full','Full codes'),('aggregate_raw','Raw residual')]
KEYS=[(f,c,r) for f in ['number','time'] for c in ['contrast','complete_removal'] for r in ['temporal','quoted']]


def export(root,out,read):
    base=Path('artifacts/extension_five_20260907/r12_output_loss')
    if not (root/base/'R12_STATE_CONFIRMATION_SUMMARY.json').is_file():return None
    summary=read(base/'R12_STATE_CONFIRMATION_SUMMARY.json');key=read(base/'R12_STATE_KEY_RESULTS.json')
    finite=read(base/'R12_FULL_DEV_SUMMARY.json');pilot=read(base/'R12_STATE_PILOT_SUMMARY.json');freeze=read(base/'STATE_FREEZE.json')
    data=dict(summary=summary,key=key,finite=finite,pilot=pilot,freeze=freeze,input_summary_paths=[(base/p).as_posix() for p in ['R12_STATE_CONFIRMATION_SUMMARY.json','R12_STATE_KEY_RESULTS.json','R12_STATE_PILOT_SUMMARY.json','R12_FULL_DEV_SUMMARY.json','R12_PILOT_SUMMARY.json','STATE_FREEZE.json','NNLS_SOURCE_REVIEW.json','SOURCE_REVIEW.json']])
    (out/'data/source_state_projection.json').write_text(json.dumps(data,indent=2)+'\n')
    def table(values,names,destination):
        lookup={(r['factor'],r['consumer'],r['role'],r['method']):r['kl'] for r in values if r['dose']==1 and r['mask_family']=='components'}
        lines=[label+' & '+' & '.join(f'{lookup[k+(name,)]:.5f}' for k in KEYS)+r' \\' for name,label in names]
        (out/'tables'/destination).write_text('\n'.join(lines)+'\n')
    names=[]
    for name,label in BASE_METHODS:
        names.extend([(name,label),(name+'_clip',label+' + clip'),(name+'_cone_half',label+' + cone')])
    table(summary['pooled'],names,'source_state_all.tex')
    table(finite['pooled'],[('aggregate_ols','Shared OLS'),('finite_ols','Finite OLS'),('finite_scalar','OLS + finite scalar'),('dense_half','Dense half'),('finite_dense','Finite dense'),('matched_atoms','One-to-one'),('finite_atoms','Finite atom scalars'),('aggregate_full','Full codes'),('finite_full','Finite full codes'),('aggregate_raw','Raw residual'),('finite_raw','Finite raw')],'finite_output_all.tex')
    lines=[]
    for i,(f,c,role) in enumerate(KEYS):
        vals=[]
        for method in ['aggregate_ols','aggregate_ols_clip','dense_half_cone_half','aggregate_raw_cone_half']:
            row=key['comparisons'][method][i]
            vals.append(f"{100*row['relative_reduction']:+.1f} ({row['directions_better']}/5; {row['incident_deletions_better']}/5)")
        lines.append(f.title()+' & '+('Contrast' if c=='contrast' else 'Removal')+' & '+role.title()+' & '+' & '.join(vals)+r' \\')
    (out/'tables/source_state_robustness.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for role in ['temporal','quoted']:
        for consumer in ['contrast','complete_removal']:
            for mask,label in [('rank_first_half','First half'),('rank_second_half','Second half'),('whole','Whole')]:
                vals=[]
                for method in ['source','aggregate_ols','aggregate_ols_clip','aggregate_ols_cone_half']:
                    row=next(r for r in summary['fixed_cases'] if r['role']==role and r['consumer']==consumer and r['mask']==mask and r['dose']==1 and r['method']==method)
                    vals.append(f"{100/(1+math.exp(-row['past_logodds'])):.1f}")
                lines.append(role.title()+' & '+('Contrast' if consumer=='contrast' else 'Removal')+' & '+label+' & '+' & '.join(vals)+r' \\')
    (out/'tables/source_state_cases.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(data,save):
    if not data:return
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.lines import Line2D
    key=data['key'];cases=key['case_states'];fig=plt.figure(figsize=(7,6.8))
    colors=ListedColormap(['#805285','#fbfaf6','#2d7158']);norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
    for i,case in enumerate(cases):
        ax=fig.add_axes([.14,.78-i*.18,.82,.10])
        values=np.array([case['states'][n] for n in ['source','aggregate_ols','aggregate_ols_cone_half']])
        ax.imshow(np.sign(values),aspect='auto',cmap=colors,norm=norm,interpolation='nearest')
        for row,col in zip(*np.where(values<0)):
            ax.text(col,row,'−',ha='center',va='center',color='white',fontsize=8)
        ax.set(yticks=[0,1,2],yticklabels=['True source','Linear map','Cone map'],xticks=[0,7,15,23,31],xticklabels=['1','8','16','24','32'])
        ax.tick_params(length=0,labelsize=7.5);ax.axvline(15.5,color='#555555',lw=.65,ls=':')
        ax.set_title(f"({'ab'[i]}) Fixed {case['role']} context: signs of absolute source-member activations",loc='left',pad=7,fontsize=9)
        if i==1:ax.set_xlabel('Original source rank (first half ends at 16)',fontsize=8,labelpad=3)
    fig.text(.14,.963,'Source 3 $\\to$ target 4, time group; the same frozen members in both contexts',fontsize=8.5)
    handles=[Line2D([],[],linestyle='none',marker='s',ms=6,markeredgecolor='#777777',markeredgewidth=.4,color=c,label=l) for c,l in [('#2d7158','Positive'),('#fbfaf6','Exact zero'),('#805285','Negative (−)')]]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.135,.927),ncol=3,frameon=False,fontsize=8,columnspacing=2)
    axs=[fig.add_axes([.275,.13,.29,.35]),fig.add_axes([.675,.13,.29,.35])]
    labels=[f'{f.title()} / '+('contrast' if c=='contrast' else 'removal')+' / '+('T' if r=='temporal' else 'Q') for f,c,r in KEYS]
    for index,(ax,method,title) in enumerate(zip(axs,['aggregate_ols','dense_half_cone_half'],['(c) Cone OLS vs. linear OLS','(d) Cone OLS vs. cone dense'])):
        vals=key['comparisons'][method]
        for i,row in enumerate(vals):
            xs=[100*r['relative_reduction'] for r in row['source_means']]
            ax.scatter(xs,i+np.linspace(-.16,.16,5),s=12,c='#999993',marker='o',alpha=.85,zorder=2)
            mean=100*row['relative_reduction'];ax.plot(mean,i,marker='D',color='#286956' if mean>=0 else '#805285',ms=4,zorder=3)
        ax.axvline(0,color='#5a5a55',ls=':',lw=.8);ax.set(ylim=(7.5,-.5),yticks=range(8),yticklabels=labels if index==0 else [],xlabel='KL reduction (%)')
        ax.tick_params(labelsize=7.5);ax.set_title(title,loc='left',fontsize=8.5,pad=8)
        for y in [1.5,3.5,5.5]:ax.axhline(y,color='#e5e5df',lw=.65,zorder=0)
    fig.text(.14,.033,'T: temporal; Q: quoted. Circles: five shared-seed directions. Diamonds: pooled comparison.',fontsize=7.5)
    fig.text(.14,.012,'Positive favors cone OLS. All six partial masks at dose 1; all 512 new inputs retained.',fontsize=7.5)
    save(fig,'source_state_projection')
