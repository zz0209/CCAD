"""Keep quality, atom geometry, source function and fixed-teacher transfer distinct."""
import hashlib
import json
from pathlib import Path
import statistics

STEPS = [8192,16384,24576,32768]
BASE = Path('artifacts/final_five_research_20260908/training_material')


def export(root,out,read):
    if not (root/BASE/'MATERIAL_32M.json').is_file():
        return None
    material = read(BASE/'MATERIAL_32M.json')
    assert material['fixed_teacher_label_distributions_exact']
    sources = [(BASE/'MATERIAL_32M.json').as_posix()]
    transfers = {4096:read('artifacts/extension_five_20260907/r10_selection/R10_SUPPORT_SUMMARY.json'),
                 8192:read('artifacts/extension_five_20260907/r10_selection/R10_TARGET8M_SUMMARY.json')}
    sources += ['artifacts/extension_five_20260907/r10_selection/'+name for name in ['R10_SUPPORT_SUMMARY.json','R10_TARGET8M_SUMMARY.json']]
    for step in STEPS[1:]:
        filename = f'TARGET_{step//1024}M_SUMMARY.json'
        transfers[step] = read(BASE/filename)
        sources += [(BASE/filename).as_posix(),transfers[step]['run']+'/metrics.raw.jsonl']
        summary = read(transfers[step]['run']+'/metrics.summary.json')
        assert summary['status']=='PASS' and read(transfers[step]['run']+'/contract_validation.json')['ok']
        assert summary['metrics_raw_sha256']==transfers[step]['raw_sha256']
    for name in ['training','function_run']:
        obj=material[name]
        run=Path(obj['run'])
        assert read(run/'metrics.summary.json')['status']=='PASS' and read(run/'contract_validation.json')['ok']
        assert hashlib.sha256((root/run/'metrics.raw.jsonl').read_bytes()).hexdigest()==obj['raw_sha256']
        sources += [(run/'metrics.raw.jsonl').as_posix(),(run/'metrics.summary.json').as_posix()]
    atom_rows=[]
    for name in ['FINAL5_R14_atom_curve8_24m_v1_20260908','FINAL5_R14_atom_curve32m_v1_20260908']:
        run=Path('runs')/name
        summary=read(run/'metrics.summary.json')
        assert summary['status']=='PASS' and read(run/'contract_validation.json')['ok']
        raw=(root/run/'metrics.raw.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==summary['metrics_raw_sha256']
        atom_rows += [json.loads(line) for line in raw.splitlines()]
        sources += [(run/'metrics.raw.jsonl').as_posix(),(run/'metrics.summary.json').as_posix()]
    atoms=[]
    for step in STEPS:
        rows=[r for r in atom_rows if r['step']==step]
        assert len(rows)==10
        values=[r['mean_absolute_cosine'] for r in rows]
        leave=[]
        for seed in range(1,6):
            retained=[r['mean_absolute_cosine'] for r in rows if seed not in [r['source_seed'],r['target_seed']]]
            assert len(retained)==6
            leave.append(dict(omitted_seed=seed,mean=statistics.fmean(retained)))
        atoms.append(dict(step=step,mean=statistics.fmean(values),min=min(values),max=max(values),pairs=rows,leave_one_seed_out=leave))
    data=dict(quality=material['quality_by_step'],function=material['source_function'],atoms=atoms,
              quality_seed_rows=material['training']['rows'],
              transfers={str(k):v['rows'] for k,v in transfers.items()},phase=material['continuation'],
              input_summary_paths=sources,
              scope='Controlled dependent checkpoints and five SAE seeds. All ten unordered atom pairs share those seeds; fixed-teacher transfer uses five dependent cyclic directions. Source groups are reselected by one frozen rule, while only target dictionaries change in the transfer curve. All task data are exposed development.')
    lines=[]
    for step in STEPS:
        q=next(r for r in data['quality'] if r['step']==step)
        a=next(r for r in atoms if r['step']==step)
        fun=[next(r for r in data['function'] if r['step']==step and r['factor']==factor) for factor in ['number','time','joint']]
        def interval(values,scale=1,places=2):
            return f'{scale*statistics.fmean(values):.{places}f} ({scale*min(values):.{places}f}--{scale*max(values):.{places}f})'
        vals=[f'{step//1024}M',f"{100*q['fve']['mean']:.2f}",f"{100*q['ce_recovered']['mean']:.2f}",f"{q['actual_nonzero_l0']['mean']:.2f}",f"{q['alive_features']['min']}--{q['alive_features']['max']}",interval([r['mean_absolute_cosine'] for r in a['pairs']],places=3)]
        lines.append(' & '.join(vals)+r' \\')
    (out/'tables/training32_material.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for row in data['quality_seed_rows']:
        if row['step'] not in STEPS[1:]:continue
        q=row['quality'];mantissa,exponent=f"{row['decoder_norm_max_error']:.2e}".split('e')
        lines.append(f"{row['step']//1024}M & {row['seed']} & {100*q['fve']:.3f} & {100*q['ce_recovered']:.3f} & {q['actual_nonzero_l0']:.2f} & {q['alive_features']} & "+rf'${mantissa}\times10^{{{int(exponent)}}}$'+r' \\')
    (out/'tables/training32_quality_seeds.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for step in STEPS:
        vals=[f'{step//1024}M']
        for factor in ['number','time','joint']:
            r=next(r for r in data['function'] if r['step']==step and r['factor']==factor)
            values=[s['accuracy'] for s in r['source_seed_means']]
            vals.append(interval(values,100))
        lines.append(' & '.join(vals)+r' \\')
    (out/'tables/training32_function.tex').write_text('\n'.join(lines)+'\n')
    lookup={step:{(r['factor'],r['consumer'],r['role'],r['method']):r for r in obj['rows']} for step,obj in transfers.items()}
    # Compare every requested cell under fixed source teachers and unchanged input IDs.
    changes=[]
    lines=[]
    for factor in ['number','time']:
        for consumer in ['contrast','complete_removal']:
            for role in ['temporal','quoted']:
                key=(factor,consumer,role,'ols_complete')
                old=lookup[4096][key]
                vals=[]
                for step in [4096]+STEPS:
                    row=lookup[step][key]
                    vals.append(f"{row['kl']:.5f}")
                    if step!=4096:
                        paired=list(zip(old['source_seed_means'],row['source_seed_means']))
                        assert all(a['source_seed']==b['source_seed'] for a,b in paired)
                        changes.append(dict(step=step,factor=factor,consumer=consumer,role=role,
                                            relative_change=row['kl']/old['kl']-1,
                                            improved_directions=sum(a['kl']>b['kl'] for a,b in paired)))
                for method in ['dense_complete','full_code_complete','raw_complete']:
                    vals.append(f"{lookup[32768][factor,consumer,role,method]['kl']:.5f}")
                lines.append(' & '.join([factor.title(),'Contrast' if consumer=='contrast' else 'Removal',role.title()]+vals)+r' \\')
    data['transfer_changes']=changes
    (out/'tables/training32_transfer.tex').write_text('\n'.join(lines)+'\n')
    (out/'data/training32_curve.json').write_text(json.dumps(data,indent=2)+'\n')
    return data


def plot(data,save):
    if not data:return
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    from matplotlib.lines import Line2D
    # Learning curves are appropriate here: the x-axis is actual ordered exposure.
    fig,axes=plt.subplots(1,3,figsize=(7,2.7))
    fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.86,wspace=.49)
    quality=[r for r in data['quality'] if r['step'] in STEPS]
    x=np.array(STEPS)*1024/1e6
    for key,label,color,marker in [('fve','FVE','#286956','o'),('ce_recovered','CE recovered','#785481','s')]:
        axes[0].plot(x,[100*r[key]['mean'] for r in quality],color=color,marker=marker,ms=3,lw=.8,label=label)
        axes[0].fill_between(x,[100*r[key]['min'] for r in quality],[100*r[key]['max'] for r in quality],color=color,alpha=.15,lw=0)
    axes[0].set(ylabel='Validation (%)',ylim=(73,91));axes[0].legend(frameon=False,fontsize=7,loc='center right')
    for pair in [(a,b) for a in range(1,6) for b in range(a+1,6)]:
        y=[next(r['mean_absolute_cosine'] for r in a['pairs'] if (r['source_seed'],r['target_seed'])==pair) for a in data['atoms']]
        axes[1].plot(x,y,color='#8d8d87',alpha=.6,lw=.5)
    axes[1].plot(x,[r['mean'] for r in data['atoms']],color='#282824',marker='o',ms=3,lw=1)
    axes[1].set(ylabel='Matched atom cosine',ylim=(.30,.38))
    for factor,label,color,marker in [('number','Number','#286956','o'),('time','Time','#785481','s'),('joint','Joint','#555550','^')]:
        rows=[next(r for r in data['function'] if r['step']==step and r['factor']==factor) for step in STEPS]
        axes[2].plot(x,[100*r['accuracy'] for r in rows],color=color,marker=marker,ms=3,lw=.8,label=label)
        axes[2].fill_between(x,[100*min(s['accuracy'] for s in r['source_seed_means']) for r in rows],[100*max(s['accuracy'] for s in r['source_seed_means']) for r in rows],color=color,alpha=.12,lw=0)
    axes[2].set(ylabel='Source accuracy (%)');axes[2].legend(frameon=False,fontsize=7,loc='lower right')
    for ax,title in zip(axes,['(a) Natural reconstruction','(b) All 8,192 atoms','(c) Reselected source groups']):
        ax.set(xticks=x,xticklabels=[f'{v:.1f}' for v in x],xlabel='Training tokens (million)');ax.set_title(title,loc='left',fontsize=8.2,pad=8);ax.tick_params(labelsize=7)
    fig.text(.075,.02,'Bands: five-seed ranges; gray paths: ten dependent seed pairs. Fixed validation and task inputs.',fontsize=7)
    save(fig,'training32_learning')
    # Show the interaction rather than averaging away opposing factor/consumer changes.
    rows=[(f,c,r) for f in ['number','time'] for c in ['contrast','complete_removal'] for r in ['temporal','quoted']]
    lookup={(r['step'],r['factor'],r['consumer'],r['role']):r for r in data['transfer_changes']}
    values=np.array([[100*lookup[s,*key]['relative_change'] for s in STEPS] for key in rows])
    bound=max(abs(values.min()),abs(values.max()),1)
    cmap=LinearSegmentedColormap.from_list('fixed_teacher_change',['#286956','#faf9f5','#785481'])
    fig=plt.figure(figsize=(7,3.6));ax=fig.add_axes([.30,.10,.51,.78])
    im=ax.imshow(values,cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
    for (i,j),v in np.ndenumerate(values):
        ax.text(j,i,f'{v:+.1f}%',ha='center',va='center',fontsize=8,color='white' if abs(v)>.62*bound else '#262626')
    ax.set(xticks=range(4),xticklabels=['8M','16M','24M','32M'],yticks=range(8),
           yticklabels=[f"{f.title()} / {'contrast' if c=='contrast' else 'removal'} / {'T' if r=='temporal' else 'Q'}" for f,c,r in rows])
    ax.tick_params(length=0,labelsize=8);ax.xaxis.tick_top()
    for sep in [.5,1.5,2.5,3.5,4.5,5.5,6.5]:ax.axhline(sep,color='white',lw=.5)
    cb=fig.colorbar(im,cax=fig.add_axes([.84,.20,.025,.56]));cb.set_label('KL change from 4M target (%)',fontsize=8);cb.ax.tick_params(labelsize=7)
    fig.text(.30,.96,'Shared OLS; same source teachers, contexts and selection budget',fontsize=9,va='top')
    fig.text(.30,.025,'Negative: improved fidelity. T: temporal cue; Q: quoted title.',fontsize=7.5)
    save(fig,'training32_fixed_teacher')
