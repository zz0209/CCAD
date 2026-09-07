"""Source-backed data/table and figure extension for complete support selection."""
import json
from pathlib import Path

METHODS=[
    ('legacy_contrast','Original contrast map'),('old_support_balanced','Old members, balanced refit'),
    ('ols_same_count','Shared OLS, old count'),('ols_complete','Shared OLS, full budget'),
    ('lasso_complete','Group sparsity, complete'),('dense_complete','Dense selection, complete'),
    ('random_complete','Random active members'),('ols_contrast_only','OLS, contrast only'),
    ('full_code_balanced','Full codes, balanced'),('full_code_complete','Full codes, complete'),
    ('raw_balanced','Raw residual, balanced'),('raw_complete','Raw residual, complete'),
    ('ols_members_native','Selected members, native')]


def export(root,out,read):
    base=Path('artifacts/extension_five_20260907/r10_selection')
    if not (root/base/'R10_MATERIAL_SUMMARY.json').is_file():return None
    old=read(base/'R10_SUPPORT_SUMMARY.json');new=read(base/'R10_TARGET8M_SUMMARY.json');material=read(base/'R10_MATERIAL_SUMMARY.json')
    data=dict(old=old['rows'],continued=new['rows'],old_comparisons=old['comparisons'],continued_comparisons=new['comparisons'],material=material,
        fixed_examples_4m=old['fixed_examples'],fixed_examples_8m=new['fixed_examples'],
        input_summary_paths=[(base/p).as_posix() for p in ['R10_SUPPORT_SUMMARY.json','R10_TARGET8M_SUMMARY.json','R10_MATERIAL_SUMMARY.json']])
    (out/'data/complete_support.json').write_text(json.dumps(data,indent=2)+'\n')
    for consumer in ['contrast','complete_removal']:
        lines=[]
        for stage,rows in [('4M',data['old']),('8M',data['continued'])]:
            lookup={(r['factor'],r['role'],r['method']):r for r in rows if r['consumer']==consumer}
            if stage=='8M':lines.append(r'\midrule')
            for method,label in METHODS:
                vals=[lookup[f,role,method]['kl'] for f in ['number','time'] for role in ['temporal','quoted']]
                lines.append(stage+' & '+label+' & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\')
        (out/f'tables/support_{consumer}.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for r in material['quality_by_step']:
        vals=[]
        for key in ['fve','ce_recovered']:
            vals.append(f"{100*r[key]['mean']:.2f} ({100*r[key]['min']:.2f}--{100*r[key]['max']:.2f})")
        mantissa,exponent=f"{r['decoder_norm_max_error']:.2e}".split('e')
        vals+=[f"{r['actual_nonzero_l0']['mean']:.2f}",f"{r['alive_features']['min']}--{r['alive_features']['max']}",rf'${mantissa}\times10^{{{int(exponent)}}}$']
        lines.append(f"{r['step']:,} & "+' & '.join(vals)+r' \\')
    (out/'tables/continued_quality.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for step in [4096,6144,8192]:
        vals=[]
        for factor in ['number','time','joint']:
            r=next(x for x in material['source_function'] if x['step']==step and x['factor']==factor)
            a=[x['accuracy'] for x in r['source_seed_means']]
            vals.append(f"{100*r['accuracy']:.2f} ({100*min(a):.2f}--{100*max(a):.2f})")
        lines.append(f"{step:,} & "+' & '.join(vals)+r' \\')
    (out/'tables/continued_source_function.tex').write_text('\n'.join(lines)+'\n')
    import math
    lines=[]
    for row_id,role in [(0,'Temporal'),(256,'Quoted')]:
        for consumer,label in [('contrast','Contrast'),('complete_removal','Removal')]:
            cells=[]
            for stage,method in [('continued','source'),('old','ols_complete'),('continued','ols_complete'),('continued','dense_complete')]:
                examples=old['fixed_examples'] if stage=='old' else new['fixed_examples']
                r=next(x for x in examples if x['source_seed']==1 and x['factor']=='time' and x['row_id']==row_id and x['consumer']==consumer and x['method']==method)
                cells.append(f"{100/(1+math.exp(-r['past_logodds'])):.1f}")
            lines.append(role+' & '+label+' & '+' & '.join(cells)+r' \\')
    (out/'tables/support_fixed_cases.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(data,save):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    if not data:return
    lookups={k:{(r['factor'],r['consumer'],r['role'],r['method']):r for r in data[k]} for k in ['old','continued']}
    variants=[('old','legacy_contrast','Original contrast · 4M'),('old','dense_complete','Dense selection · 4M'),
              ('old','ols_complete','Shared OLS · 4M'),('continued','ols_complete','Shared OLS · 8M'),
              ('continued','full_code_complete','Full codes · 8M'),('continued','raw_complete','Raw residual')]
    fig,axs=plt.subplots(2,2,figsize=(7,5.8));fig.subplots_adjust(left=.225,right=.985,bottom=.1,top=.89,wspace=.20,hspace=.44)
    for ci,consumer in enumerate(['contrast','complete_removal']):
        for fi,factor in enumerate(['number','time']):
            ax=axs[ci,fi]
            for i,(stage,method,label) in enumerate(variants):
                for role,offset,color,marker in [('temporal',-.12,'#286956','o'),('quoted',.12,'#785481','s')]:
                    r=lookups[stage][factor,consumer,role,method];y=i+offset
                    ax.plot([r['min_seed_kl'],r['max_seed_kl']],[y,y],color=color,lw=.75,alpha=.7)
                    ax.plot(r['kl'],y,color=color,marker=marker,ms=3.5)
            ax.set_xscale('log');ax.set(ylim=(5.48,-.48),yticks=range(6),yticklabels=[v[2] for v in variants] if fi==0 else [],xlabel='Source-to-candidate KL (nat, log scale)')
            ax.set_xlim((.008,.08) if ci==0 else (.005,.25))
            ax.xaxis.set_major_locator(FixedLocator([.01,.02,.05] if ci==0 else [.01,.02,.05,.1,.2]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"({'abcd'[ci*2+fi]}) {factor.title()} / "+('contrast' if ci==0 else 'group removal'),loc='left',pad=10)
            ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=8)
            ax.grid(axis='x',which='major',color='#e6e6e3',lw=.5);ax.set_axisbelow(True)
    handles=[Line2D([],[],marker=m,color=c,lw=.7,label=l,ms=3.5) for l,c,m in [('Temporal cue','#286956','o'),('Quoted title','#785481','s')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.59,.98),ncol=2,frameon=False,fontsize=8.5)
    save(fig,'complete_support')
