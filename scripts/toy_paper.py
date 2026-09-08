"""Paper export and figures for frozen learned-superposition experiments."""
import json
from pathlib import Path

REGIMES=['independent','coactive','correlated','mixed']
REGIME_NAMES={'independent':'Independent','coactive':'Coactive','correlated':'Exact copies','mixed':'80% copies'}
METHODS=[('ols_code','OLS linear'),('ols_code_clip','OLS + clip'),('ols_affine_clip','OLS affine + clip'),('ols_preactivation','OLS preactivation'),('ols_rectified','OLS finite ReLU'),
         ('dense_code','Dense linear'),('dense_affine_clip','Dense affine + clip'),('dense_preactivation','Dense preactivation'),('dense_rectified','Dense finite ReLU'),
         ('atom1_code','Single-atom linear'),('atom1_affine_clip','Single-atom affine + clip'),('atom1_rectified','Single-atom finite ReLU'),
         ('random_code','Random linear'),('random_affine_clip','Random affine + clip'),('random_preactivation','Random preactivation'),('random_rectified','Random finite ReLU'),
         ('full_code','Full-code linear'),('full_affine_clip','Full-code affine + clip'),('full_preactivation','Full-code preactivation'),('full_rectified','Full-code finite ReLU'),
         ('raw_linear','Raw-hook linear'),('target_full_reencode','Full target re-encoding'),('source_encoder_oracle','Source-operation oracle'),('factor_truth_oracle','Factor-truth oracle')]

def export(root,out,read):
    base=Path('artifacts/inserted_superposition_20260907')
    if not (root/base/'KEY_RESULTS.json').is_file():return None
    key=read(base/'KEY_RESULTS.json');low=read(base/'CONFIRM_LOW_SUMMARY.json');high=read(base/'CONFIRM_HIGH_SUMMARY.json')
    pilot_low=read(base/'PILOT_STABLE_LOW_SUMMARY.json');pilot_high=read(base/'PILOT_STABLE_HIGH_SUMMARY.json');failed=read(base/'PILOT_HIGH_SUMMARY.json')
    paths=[base/p for p in ['KEY_RESULTS.json','CONFIRM_LOW_SUMMARY.json','CONFIRM_HIGH_SUMMARY.json','PILOT_STABLE_LOW_SUMMARY.json','PILOT_STABLE_HIGH_SUMMARY.json','PILOT_HIGH_SUMMARY.json','FREEZE.json','SOURCE_REVIEW.json']]
    traces={label:read(Path('runs')/summary['run_id']/'training_traces.json')['traces'] for label,summary in [('LOW',low),('HIGH',high)]}
    paths.extend(Path('runs')/summary['run_id']/'training_traces.json' for summary in [low,high])
    data=dict(key=key,low={k:low[k] for k in ['run_id','pooled','comparisons','source_quality','coverage','method_counts']},high={k:high[k] for k in ['run_id','pooled','comparisons','source_quality','coverage','method_counts']},traces=traces,freeze=read(base/'FREEZE.json'),input_summary_paths=[p.as_posix() for p in paths])
    (out/'data/learned_superposition.json').write_text(json.dumps(data,indent=2)+'\n')
    def save(name,lines):(out/'tables'/name).write_text('\n'.join(lines).replace('%',r'\%')+'\n')
    for label,summary in [('low',low),('high',high)]:
        lookup={(r['regime'],r['split'],r['family'],r['method']):r for r in summary['pooled']}
        lines=[]
        for method,title in METHODS:
            vals=[lookup[(regime,split,'all',method)]['function_nmse'] for regime in REGIMES for split in ['evaluation','shift']]
            lines.append(title+' & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\')
        save(f'toy_all_{label}.tex',lines)
    lines=[]
    for regime in REGIMES:
        for label,l1 in [('LOW',.05),('HIGH',.15)]:
            r=next(r for r in key['quality'] if r['material']==label and r['regime']==regime)
            truth=next(r for r in key['material_comparison'] if r['regime']==regime and r['split']=='evaluation')[('low' if label=='LOW' else 'high')+'_truth_nmse']
            lines.append(f"{REGIME_NAMES[regime]} & {l1:.2f} & {100*r['fve']['mean']:.2f} & {r['l0']['mean']:.2f} & {r['alive']['minimum']}--{r['alive']['maximum']} & {truth:.4f}"+r' \\')
    save('toy_quality.tex',lines)
    lines=[]
    for r in key['geometry']:
        lines.append(f"{REGIME_NAMES[r['regime']]} & {r['toy_seed']} & {r['pair_cosine']:.5f} & {r['pair_separation']:.5f} & {r['shift_pair_difference_nmse']:.5f}"+r' \\')
    save('toy_geometry.tex',lines)
    lines=[]
    for regime in REGIMES:
        for split in ['evaluation','shift']:
            cells=[]
            for comp in ['ols_affine_clip','dense_rectified','atom1_rectified','full_rectified']:
                r=next(r for r in high['comparisons'] if (r['regime'],r['split'],r['family'],r['comparator'])==(regime,split,'all',comp))
                good=sum(v['function_relative_reduction']>0 for v in r['incident_seed_deletions'])
                cells.append(f"{100*r['function_relative_reduction']:+.1f} ({good}/5)")
            lines.append(REGIME_NAMES[regime]+' & '+('Original' if split=='evaluation' else 'Independent')+' & '+' & '.join(cells)+r' \\')
    save('toy_robustness.tex',lines)
    lines=[]
    for r in key['material_comparison']:
        lines.append(f"{REGIME_NAMES[r['regime']]} & "+('Original' if r['split']=='evaluation' else 'Independent')+f" & {r['low_truth_nmse']:.4f} & {r['high_truth_nmse']:.4f} & {100*r['relative_reduction']:.1f} & {r['seeds_better']}/10"+r' \\')
    save('toy_truth.tex',lines)
    lines=[]
    for regime in ['independent','correlated','mixed']:
        for split in ['evaluation','shift']:
            values=[]
            for summary,method in [(failed,'full_rectified'),(pilot_high,'full_rectified'),(pilot_high,'ols_rectified'),(pilot_high,'ols_affine_clip')]:
                r=next(r for r in summary['pooled'] if (r['regime'],r['split'],r['family'],r['method'])==(regime,split,'all',method));values.append(r['function_nmse'])
            lines.append(REGIME_NAMES[regime]+' & '+('Original' if split=='evaluation' else 'Independent')+' & '+' & '.join(f'{v:.5f}' for v in values)+r' \\')
    save('toy_development_failure.tex',lines)
    e=key['examples'][0];lines=[]
    for col,member in enumerate(e['source_members']):
        vals=[row[col] for row in e['coefficient']]
        lines.append(str(member)+' & '+' & '.join(f'{v:+.5f}' for v in vals)+f" & {e['intercept'][col]:+.5f}"+r' \\')
    save('toy_example_map.tex',lines)
    lines=[]
    for e in key['examples']:
        lines.append(str(e['evaluation_row'])+' & '+f"{e['latent'][2]:.5f}"+' & '+' & '.join(f'{v:.5f}' for v in e['target_codes'])+r' \\')
    save('toy_example_inputs.tex',lines)
    return data


def plot(data,save):
    if not data:return
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
    from matplotlib.lines import Line2D
    green='#286956';purple='#785481';gray='#777773';key=data['key']
    fig=plt.figure(figsize=(7,5.35))
    ax=fig.add_axes([.16,.565,.34,.34]);bx=fig.add_axes([.61,.565,.35,.34])
    for yi,regime in enumerate(REGIMES):
        r=next(r for r in key['material_comparison'] if r['regime']==regime and r['split']=='evaluation')
        for j,base in enumerate([202,303]):
            points=[r for r in r['per_seed'] if r['toy_seed']==base];y=yi+(-.13 if j==0 else .13);marker='o' if j==0 else 's'
            means=[]
            for field,color,filled in [('low_error',purple,False),('high_error',green,True)]:
                values=[r[field] for r in points];mean=np.mean(values);means.append(mean)
                ax.plot([min(values),max(values)],[y,y],color=color,lw=.6,alpha=.6)
                ax.plot(mean,y,marker=marker,ms=4,mec=color,mfc=color if filled else 'white',mew=.8)
            ax.plot(means,[y,y],color=gray,lw=.5,zorder=0)
        for j,(comp,marker) in enumerate([('ols_affine_clip','o'),('dense_rectified','s'),('full_rectified','^')]):
            r=next(r for r in data['high']['comparisons'] if (r['regime'],r['split'],r['family'],r['comparator'])==(regime,'evaluation','all',comp))
            ratio=1-r['function_relative_reduction'];values=[1-v['function_relative_reduction'] for v in r['by_toy_seed']];y=yi+(j-1)*.20;color=green if ratio<1 else purple
            bx.plot([min(values),max(values)],[y,y],color=color,lw=.65)
            bx.plot(ratio,y,marker=marker,ms=4,color=color)
    labels=[REGIME_NAMES[r] for r in REGIMES]
    ax.set(ylim=(3.6,-.6),yticks=range(4),yticklabels=labels,xlim=(.25,.88),xlabel='Source-group truth error (NMSE)')
    ax.set_title('(a) A changed source grouping',loc='left',pad=9,fontsize=9)
    ax.tick_params(labelsize=7.5)
    bx.axvline(1,color=gray,ls=':',lw=.7);bx.set_xscale('log',base=2)
    bx.set(ylim=(3.6,-.6),yticks=range(4),yticklabels=[],xlim=(.25,16),xticks=[.25,.5,1,2,4,8,16],xticklabels=['¼','½','1','2','4','8','16'],xlabel='Compact error / comparator error')
    bx.set_title('(b) Same source operation, new fit',loc='left',pad=9,fontsize=9);bx.tick_params(labelsize=7.5)
    legend=[Line2D([],[],marker='o',ls='',mfc='white',mec=purple,label=r'$L_1=0.05$'),Line2D([],[],marker='o',ls='',color=green,label=r'$L_1=0.15$')]
    legend.extend(Line2D([],[],marker=m,ls='',color=gray,label=l) for m,l in [('o','Affine + clip (4)'),('s','Dense ReLU (4)'),('^','Full ReLU (24)')])
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.15,.993),ncol=5,frameon=False,fontsize=7.4,handletextpad=.3,columnspacing=.9)
    cx=fig.add_axes([.16,.11,.34,.29]);dx=fig.add_axes([.61,.11,.35,.29])
    for yi,regime in enumerate(REGIMES):
        for j,base in enumerate([202,303]):
            r=next(r for r in key['geometry'] if r['regime']==regime and r['toy_seed']==base);y=yi+(-.12 if j==0 else .12);marker='o' if j==0 else 's'
            cx.plot(r['pair_separation'],y,marker=marker,ms=4,color='#444440')
            dx.plot(r['shift_pair_difference_nmse'],y,marker=marker,ms=4,color='#444440')
    cx.set(ylim=(3.55,-.55),yticks=range(4),yticklabels=labels,xlim=(-.035,1.9),xticks=[0,.5,1,1.5],xlabel=r'$\|w_0-w_1\|\,/\,\|w_0+w_1\|$')
    cx.set_title('(c) Learned pair separation',loc='left',pad=9,fontsize=9);cx.tick_params(labelsize=7.5)
    dx.axvline(1,color=gray,ls=':',lw=.7);dx.set(ylim=(3.55,-.55),yticks=range(4),yticklabels=[],xlim=(0,1.09),xticks=[0,.25,.5,.75,1],xlabel='Independent-pair difference error')
    dx.set_title('(d) Difference readout after shift',loc='left',pad=9,fontsize=9);dx.tick_params(labelsize=7.5)
    for a in [ax,bx,cx,dx]:
        for y in [.5,1.5,2.5]:a.axhline(y,color='#e9e8e3',lw=.45,zorder=0)
    save(fig,'learned_toy_correspondence')

    examples=key['examples'];fig=plt.figure(figsize=(7,5.25))
    cmap=LinearSegmentedColormap.from_list('signed_toy',[purple,'#fbfaf6',green]);positive=LinearSegmentedColormap.from_list('positive_toy',['#fbfaf6',green])
    ax=fig.add_axes([.16,.645,.30,.24]);coef=np.asarray(examples[0]['coefficient']).T;limit=max(abs(coef).max(),1e-9)
    ax.imshow(coef,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(0,-limit,limit),interpolation='nearest')
    for (i,j),v in np.ndenumerate(coef):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=7.6,color='white' if abs(v)>.6*limit else '#262626')
    ax.set(yticks=range(coef.shape[0]),yticklabels=examples[0]['source_members'],xticks=range(coef.shape[1]),xticklabels=examples[0]['target_members'],xlabel='Target member ID',ylabel='Source member ID')
    ax.set_title('(a) One signed fitted map',loc='left',pad=9,fontsize=9);ax.tick_params(length=0,labelsize=7.5)
    bx=fig.add_axes([.65,.645,.30,.24]);states=np.asarray([state for e in examples for state in [e['source_state'],e['states']['ols_rectified']]])
    bx.imshow(states,aspect='auto',cmap=positive,vmin=0,vmax=max(states.max(),1e-9),interpolation='nearest')
    for (i,j),v in np.ndenumerate(states):bx.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7.2,color='white' if v>.4 else '#262626')
    bx.set(yticks=range(6),yticklabels=['11: source','11: map','6: source','6: map','0: source','0: map'],xticks=range(3),xticklabels=examples[0]['source_members'],xlabel='Source member ID')
    bx.set_title('(b) Actual and predicted states',loc='left',pad=9,fontsize=9);bx.tick_params(length=0,labelsize=7)
    for y in [1.5,3.5]:bx.axhline(y,color='white',lw=1.4)
    cx=fig.add_axes([.16,.17,.79,.34]);values=np.asarray([e['output_changes'][method] for e in examples for method in ['factor_truth','source','ols_rectified']]);norm=TwoSlopeNorm(0,-1,1)
    im=cx.imshow(values,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest');labels=[]
    for case in ['Alone','With others','Absent']:labels.extend([case+': truth','Source','Map'])
    cx.set(yticks=range(9),yticklabels=labels,xticks=range(12),xticklabels=range(12),xlabel='Toy output factor ID (removed factor: 2)')
    cx.set_title('(c) The same map can reproduce a source mistake',loc='left',pad=9,fontsize=9);cx.tick_params(length=0,labelsize=7.3)
    for y in [2.5,5.5]:cx.axhline(y,color='white',lw=1.6)
    for x in [1.5,2.5]:cx.axvline(x,color='#555550',ls=':',lw=.7)
    for i in range(9):cx.text(2,i,f'{values[i,2]:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(values[i,2])>.6 else '#262626')
    cax=fig.add_axes([.42,.070,.43,.023]);fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-1,-.5,0,.5,1],label='Change in decoder output')
    cax.tick_params(labelsize=7)
    save(fig,'toy_fixed_operations')

    fig,axes=plt.subplots(2,4,figsize=(7,3.45),sharex=True,sharey='row');fig.subplots_adjust(left=.095,right=.985,bottom=.14,top=.85,wspace=.23,hspace=.30)
    for j,regime in enumerate(REGIMES):
        for label,color in [('LOW',purple),('HIGH',green)]:
            rows=[r for r in data['traces'][label] if r['stage']=='sae' and r['regime']==regime]
            steps=sorted({r['step'] for r in rows})
            for i,field in enumerate(['fve','source_factor_error']):
                mean=[];lo=[];hi=[]
                for step in steps:
                    vals=[1-r[field] if field=='fve' else r[field] for r in rows if r['step']==step]
                    mean.append(np.mean(vals));lo.append(min(vals));hi.append(max(vals))
                ax=axes[i,j];x=np.array(steps)/1000;ax.fill_between(x,lo,hi,color=color,alpha=.10,linewidth=0);ax.plot(x,mean,color=color,marker='o',ms=2.3,lw=.85)
                ax.set_yscale('log');ax.set_xticks([0,2,4]);ax.tick_params(labelsize=7)
        axes[0,j].set_title(REGIME_NAMES[regime],fontsize=8.5,pad=6)
        axes[1,j].set_xlabel('Updates (thousands)',fontsize=7.5)
    axes[0,0].set_ylabel('Unexplained variance',fontsize=8);axes[1,0].set_ylabel('Factor deletion NMSE',fontsize=8)
    fig.legend(handles=[Line2D([],[],color=purple,label=r'$L_1=0.05$'),Line2D([],[],color=green,label=r'$L_1=0.15$')],loc='upper center',bbox_to_anchor=(.53,1.01),ncol=2,frameon=False,fontsize=8)
    save(fig,'toy_learning_curves')
