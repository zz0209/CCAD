"""Export source-component confirmation and plot its actual outputs."""
import json,math
from pathlib import Path
METHODS=[('aggregate_ols','Shared aggregate OLS'),('component_half','Half-family OLS'),('component_singleton','Singleton OLS'),('dense_half','Dense half-family'),('matched_atoms','One-to-one code match'),('aggregate_full','Aggregate full codes'),('full_half','Half-family full codes'),('aggregate_raw','Aggregate raw'),('raw_half','Half-family raw')]


def export(root,out,read):
    base=Path('artifacts/extension_five_20260907/r11_components')
    if not (root/base/'R11_CONFIRMATION_SUMMARY.json').is_file():return None
    summary=read(base/'R11_CONFIRMATION_SUMMARY.json');pilot=read(base/'R11_PILOT_SUMMARY.json');freeze=read(base/'FREEZE.json')
    panel=read(Path(summary['run'])/'panel.json') if (root/summary['run']/'panel.json').is_file() else read(Path('runs/EXT_R11_component_material_v1_20260907/panel.json'))
    data=dict(summary=summary,pilot=pilot,freeze=freeze,case_texts=[{key:r[key] for key in ['id','text','cue_role']} for r in panel['rows'] if r['id'] in [0,256]],input_summary_paths=[(base/p).as_posix() for p in ['R11_CONFIRMATION_SUMMARY.json','R11_PILOT_SUMMARY.json','FREEZE.json','operations/INDEX.json']])
    (out/'data/component_reuse.json').write_text(json.dumps(data,indent=2)+'\n')
    for consumer in ['contrast','complete_removal']:
        lines=[]
        for name,values in [('Pilot',pilot['pooled']),('Fresh',summary['pooled'])]:
            lookup={(r['factor'],r['role'],r['method']):r for r in values if r['consumer']==consumer and r['dose']==1 and r['mask_family']=='components'}
            if name=='Fresh':lines.append(r'\midrule')
            for method,label in METHODS:
                vals=[lookup[f,role,method]['kl'] for f in ['number','time'] for role in ['temporal','quoted']]
                lines.append(name+' & '+label+' & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\')
        (out/f'tables/component_{consumer}.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for f in ['number','time']:
        for c in ['contrast','complete_removal']:
            for role in ['temporal','quoted']:
                vals=[]
                for method in ['matched_atoms','dense_half','aggregate_full','aggregate_raw']:
                    r=next(x for x in summary['comparisons'] if x['factor']==f and x['consumer']==c and x['role']==role and x['method']==method and x['dose']==1 and x['mask_family']=='components')
                    improvement=r['primary_relative_reduction'];n=sum(x['primary_minus_comparator']<0 for x in r['leave_incident_seed_out'])
                    vals.append(f'{100*improvement:+.1f} ({n}/5)')
                lines.append(f.title()+' & '+('Contrast' if c=='contrast' else 'Removal')+' & '+role.title()+' & '+' & '.join(vals)+r' \\')
    (out/'tables/component_robustness.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for role in ['temporal','quoted']:
        for mask,label in [('rank_first_half','First half'),('rank_second_half','Second half'),('whole','Whole')]:
            for consumer,clabel in [('contrast','Contrast'),('complete_removal','Removal')]:
                vals=[]
                for method in ['source','aggregate_ols','matched_atoms']:
                    r=next(x for x in summary['fixed_cases'] if x['role']==role and x['mask']==mask and x['consumer']==consumer and x['dose']==1 and x['method']==method)
                    vals.append(f"{100/(1+math.exp(-r['past_logodds'])):.1f}")
                lines.append(role.title()+' & '+clabel+' & '+label+' & '+' & '.join(vals)+r' \\')
    (out/'tables/component_cases.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for factor in ['number','time']:
        for consumer in ['contrast','complete_removal']:
            for role in ['temporal','quoted']:
                rows=[r for r in summary['interactions'] if r['factor']==factor and r['consumer']==consumer and r['role']==role]
                which='number' if factor=='number' else 'time'
                vals=[]
                for method,key in [('source','absolute_source_'+which+'_interaction'),('aggregate_ols',which+'_error'),('matched_atoms',which+'_error'),('aggregate_full',which+'_error'),('aggregate_raw',which+'_error')]:
                    a=[r[key] for r in rows if r['method']==method];vals.append(f'{sum(a)/len(a):.4f}')
                lines.append(factor.title()+' & '+('Contrast' if consumer=='contrast' else 'Removal')+' & '+role.title()+' & '+' & '.join(vals)+r' \\')
    (out/'tables/component_interaction.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(data,save):
    if not data:return
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
    from matplotlib.lines import Line2D
    summary=data['summary'];cases=summary['fixed_cases'];fig=plt.figure(figsize=(7,6.15))
    axs=[fig.add_axes([.10,.62,.38,.26]),fig.add_axes([.59,.62,.38,.26])]
    styles=[('source','Source','o','#343434',0),('aggregate_ols','Shared map','s','#286956',-.15),('matched_atoms','One-to-one','x','#785481',.15)]
    masks=['rank_first_half','rank_second_half','whole']
    for ax,role,title in zip(axs,['temporal','quoted'],['(a) Temporal cue','(b) Same cue in a quoted title']):
        for method,label,marker,color,offset in styles:
            vals=[next(r for r in cases if r['role']==role and r['mask']==mask and r['consumer']=='contrast' and r['dose']==1 and r['method']==method) for mask in masks]
            ax.scatter(np.arange(3)+offset,[100/(1+math.exp(-r['past_logodds'])) for r in vals],marker=marker,color=color,s=22,zorder=3)
        base=vals[0]['past_logodds']-vals[0]['past_change'];ax.axhline(100/(1+math.exp(-base)),ls=':',lw=.8,color='#999999')
        ax.set(xticks=range(3),xticklabels=['First half','Second half','Whole'],ylim=(-3,103),yticks=[0,25,50,75,100],ylabel='Conditional past probability (%)' if role=='temporal' else '')
        ax.set_title(title,loc='left',pad=10);ax.grid(axis='y',color='#e9e9e5',lw=.5);ax.set_axisbelow(True)
    fig.text(.1,.967,'Frozen example: source 3 $\\to$ target 4, time contrast',fontsize=9)
    first=data['case_texts'][0]['text'].replace('<|endoftext|>','')
    fig.text(.1,.932,'“'+first+'”  /  “The title is ‘Right now’. Right now, …”',fontsize=8)
    handles=[Line2D([],[],linestyle='none',marker=m,color=c,ms=4,label=l) for _,l,m,c,_ in styles]+[Line2D([],[],ls=':',color='#999999',lw=.8,label='Unedited recipient')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.57),ncol=4,frameon=False,fontsize=8,columnspacing=1.3)
    ax=fig.add_axes([.29,.073,.60,.39]);cells=[(f,c,r) for f in ['number','time'] for c in ['contrast','complete_removal'] for r in ['temporal','quoted']]
    methods=['matched_atoms','dense_half','aggregate_full','aggregate_raw'];arr=np.empty((8,4))
    for i,(f,c,role) in enumerate(cells):
        for j,method in enumerate(methods):
            x=next(r for r in summary['comparisons'] if r['factor']==f and r['consumer']==c and r['role']==role and r['method']==method and r['dose']==1 and r['mask_family']=='components');arr[i,j]=x['reference_kl']/x['kl']
    cmap=LinearSegmentedColormap.from_list('fidelity',['#31765d','#f7f7f2','#815a86']);norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2)
    im=ax.imshow(np.log2(arr),cmap=cmap,norm=norm,aspect='auto')
    for (i,j),v in np.ndenumerate(arr):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(np.log2(v))>1.15 else '#242424')
    ax.set(xticks=range(4),xticklabels=['One-to-one','Dense half','Full codes','Raw residual'],yticks=range(8),yticklabels=[f'{f.title()} · '+('contrast' if c=='contrast' else 'removal')+' · '+('T' if r=='temporal' else 'Q') for f,c,r in cells])
    ax.tick_params(length=0,labelsize=8);ax.set_title('(c) Shared-map KL / comparator KL on all fresh inputs',loc='left',pad=12)
    for y in [1.5,3.5,5.5]:ax.axhline(y,color='white',lw=1)
    fig.text(.29,.017,'Six partial masks, dose 1.  Below 1 favors shared map.  T: temporal; Q: quoted.',fontsize=7.8)
    save(fig,'component_reuse')
    fig,axs=plt.subplots(1,2,figsize=(7,3.0));fig.subplots_adjust(left=.10,right=.985,bottom=.19,top=.77,wspace=.30)
    for ax,factor in zip(axs,['number','time']):
        for role,color,marker in [('temporal','#286956','o'),('quoted','#785481','s')]:
            for method,label,ls in [('aggregate_ols','Shared map','-'),('aggregate_full','Full codes','--')]:
                vals=[next(r['kl'] for r in summary['rows'] if r['factor']==factor and r['consumer']=='complete_removal' and r['role']==role and r['mask']=='rank_first_half' and r['dose']==dose and r['method']==method) for dose in [.5,1,1.5]]
                ax.plot([.5,1,1.5],vals,color=color,marker=marker,ls=ls,lw=.9,ms=3,label=role.title()+' / '+label)
        ax.set(xticks=[.5,1,1.5],xlabel='Removal dose',ylabel='Source-to-candidate KL (nat)' if factor=='number' else '',ylim=(0,None))
        ax.set_title(factor.title()+' / first source half',loc='left');ax.grid(axis='y',color='#e7e7e4',lw=.5)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=8)
    save(fig,'component_dose')
