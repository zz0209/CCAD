"""Source-backed multisite/native-participation evidence for the one manuscript."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

TASKS=['agr_gender','filler_gap_obj','npi_any_obj-relc','garden_npz_v-trans']
LABELS=['Gender','Filler--gap','NPI licensing','NP/Z']
OPS=['whole','part0','part1','half_dose','mixed_dose']
METHODS=[('native_participation','Shared native'),('native_exclusive','Exclusive native'),
 ('global_hungarian_native','Global geometry native'),('task_activation_assignment_native','Task correlation native'),
 ('random_native','Random native'),('wrong_task_participation','Wrong-task native'),
 ('native_direct_target_supervision','Target-label native'),
 ('compact_response_fit','Compact readout + response'),('compact_ridge','Compact ridge'),
 ('dense_response_fit','Dense-selected + response'),('dense_ridge','Dense-selected ridge'),
 ('full_response_fit','Full-code + response'),('full_ridge','Full-code ridge'),
 ('raw_response_fit','Raw + response'),('raw_ridge','Raw ridge')]


def export(root,out,read):
    import numpy as np
    bases=[Path('runs')/f'FINAL5_R15_participation_five_s{s}_v2_20260908' for s in range(1,6)]
    if not all((root/b/'metrics.summary.json').exists() for b in bases):
        return None
    sourcebase=Path('runs/FINAL5_R15_multisite_source_v2_20260908')
    coverage=read(sourcebase/'metrics.summary.json')
    assert coverage['status']=='PASS' and read(sourcebase/'contract_validation.json')['ok']
    def verified_raw(base,summary):
        path=root/base/'metrics.raw.jsonl'
        payload=path.read_bytes()
        assert hashlib.sha256(payload).hexdigest()==summary['metrics_raw_sha256']
        rows=[json.loads(line) for line in payload.splitlines()]
        assert len(rows)==summary['rows']
        return rows
    verified_raw(sourcebase,coverage)
    # Recompute selection from original cells. Never choose a position on held.
    source_coverage=[]
    for task in TASKS:
        for seed in range(6):
            rows=[r for r in coverage['cells'] if r['task']==task and r['seed']==seed]
            single=max([r for r in rows if r['split']=='fit' and r['mode'].startswith('region_') and r['mode']!='region_ends_nonfinal'],
                       key=lambda r:r['donor_oriented_change_mean'])
            selected=next(r for r in rows if r['split']=='held_component_development' and r['mode']==single['mode'])
            multi=next(r for r in rows if r['split']=='held_component_development' and r['mode']=='suffix_nonfinal')
            source_coverage.append(dict(task=task,seed=seed,fit_selected_single=single['mode'],single=selected,multi=multi))
    cells=[];raw=[];fits=[];examples=[];sources=[];panels=[];matrix_example=None
    for s,base in enumerate(bases,1):
        summary=read(base/'metrics.summary.json')
        assert summary['status']=='PASS' and read(base/'contract_validation.json')['ok']
        cfg=read(base/'config.resolved.json')
        assert cfg['source_seed']==s and cfg['target_seed']==s%5+1
        actual=verified_raw(base,summary)
        panel=read(base/'panel.json')['rows'];panels.append(panel)
        for r in summary['cells']:
            if r['split']=='held_component_development':
                cells.append(dict(r,source_seed=s,actual_target_seed=s%5+1))
        raw.extend(dict(r,source_seed=s,actual_target_seed=s%5+1) for r in actual if r['split']=='held_component_development')
        for task in TASKS:
            definition=read(base/f'{task}_source_definition.json')
            meta={kind:read(base/f'{task}_{kind}_fit.json') for kind in
                  ['native_participation','native_exclusive','native_target_supervision','compact_readout','dense_readout','full_readout','raw_readout']}
            readers=read(base/f'{task}_reader_initialization.json')
            fits.append(dict(source_seed=s,target_seed=s%5+1,task=task,definition=definition,methods=meta,readers=readers))
            if s==1:
                p=next(r for r in panel if r['task']==task and r['split']=='held_component_development')
                examples.append(dict(task=task,panel=p,donor=panel[p['donor_id']],
                    observations=[r for r in actual if r['row_id']==p['row_id']],
                    rule='First hash-ordered retained component and side; no outcome screening'))
        if s==1:
            g=np.load(root/base/'npi_any_obj-relc_participation_export.npz')
            matrix_example={k:g[k].tolist() for k in g.files}
            sources.append((base/'npi_any_obj-relc_participation_export.npz').as_posix())
        sources += [(base/name).as_posix() for name in ['metrics.summary.json','metrics.raw.jsonl','config.resolved.json','code_hashes.json','panel.json']]
    assert all(p==panels[0] for p in panels), 'Same prompt components required for paired seed comparisons'
    lookup={(r['source_seed'],r['task'],r['operation'],r['method']):r for r in cells}
    comparisons=[]
    # Component bootstrap keeps all SAE seeds and reciprocal directions together.
    # This is conditional on the five trained SAEs and the observed task templates.
    rng=np.random.default_rng(150908)
    for task in TASKS:
        for operations,opname in [(['whole'],'whole'),(['part0','part1'],'parts'),(['half_dose','mixed_dose'],'untrained_doses')]:
            for method in ['native_exclusive','global_hungarian_native','task_activation_assignment_native','compact_response_fit','dense_response_fit','full_response_fit','raw_response_fit']:
                native=[lookup[s,task,op,'native_participation']['kl_to_reference_mean'] for s in range(1,6) for op in operations]
                other=[lookup[s,task,op,method]['kl_to_reference_mean'] for s in range(1,6) for op in operations]
                byseed=[]
                for s in range(1,6):
                    a=fmean(lookup[s,task,op,'native_participation']['kl_to_reference_mean'] for op in operations)
                    b=fmean(lookup[s,task,op,method]['kl_to_reference_mean'] for op in operations)
                    byseed.append(dict(source=s,target=s%5+1,native=a,baseline=b,reduction_fraction=1-a/b))
                leave=[]
                for excluded in range(1,6):
                    keep=[v for v in byseed if excluded not in [v['source'],v['target']]]
                    assert len(keep)==3
                    leave.append(dict(excluded_seed=excluded,reduction_fraction=1-fmean(x['native'] for x in keep)/fmean(x['baseline'] for x in keep)))
                group=defaultdict(lambda:defaultdict(list))
                for r in raw:
                    if r['task']==task and r['operation'] in operations and r['method'] in ['native_participation',method]:
                        group[r['component']][r['method']].append(r['kl_to_reference'])
                values=np.array([[fmean(v['native_participation']),fmean(v[method])] for _,v in sorted(group.items())])
                assert values.shape==(7,2)
                resampled=values[rng.integers(0,7,size=(5000,7))].mean(axis=1)
                boot=1-resampled[:,0]/resampled[:,1]
                comparisons.append(dict(task=task,operation_family=opname,baseline=method,native_kl=fmean(native),baseline_kl=fmean(other),
                    reduction_fraction=1-fmean(native)/fmean(other),directions=byseed,leave_incident_seed_out=leave,
                    observed_component_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    positive_directions=sum(x['native']<x['baseline'] for x in byseed)))
    data=dict(source_coverage=source_coverage,cells=cells,comparisons=comparisons,fits=fits,examples=examples,matrix_example=matrix_example,
        input_summary_paths=sources+[str(sourcebase/'metrics.summary.json'),str(sourcebase/'metrics.raw.jsonl')],
        scope='Exposed CausalGym train components. Five dependent cyclic SAE pairs; one shared prompt panel; source-only IG32. Interleaved rank parts are experimental controls, not named independent causal variables. No SemanticOT/DBM/DAS result for these source groups. Raw/full/compact readers retain all response and ridge variants.',
        uncertainty='Seven exact-prompt components resampled jointly across every seed and reciprocal direction; conditional on these templates and five SAEs, not an independent-seed confidence interval.')
    (out/'data/native_participation.json').write_text(json.dumps(data,indent=2)+'\n')
    with (out/'data/native_participation_cells.csv').open('w',newline='') as f:
        keys=sorted(set().union(*(r.keys() for r in cells)));writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(cells)
    lines=[]
    for method,label in METHODS:
        vals=[fmean(lookup[s,task,'whole',method]['kl_to_reference_mean'] for s in range(1,6)) for task in TASKS]
        lines.append(label+' & '+' & '.join(f'{v:.6f}' for v in vals)+r' \\')
    (out/'tables/participation_whole.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for task,label in zip(TASKS,LABELS):
        for method,ml in [METHODS[0],METHODS[1],METHODS[2],METHODS[3],METHODS[7],METHODS[11],METHODS[13]]:
            vals=[fmean(lookup[s,task,op,method]['kl_to_reference_mean'] for s in range(1,6)) for op in OPS[1:]]
            lines.append(label+' & '+ml+' & '+' & '.join(f'{v:.6f}' for v in vals)+r' \\')
    (out/'tables/participation_components.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for task,label in zip(TASKS,LABELS):
        for method,ml in [('raw_donor','Raw multisite'),('full_source_sae','Complete source SAE'),('source_selected','Source32'),
                          ('source_signed_16','Signed source16'),('source_signed_64','Signed source64'),('source_absolute_32','Absolute source32'),
                          ('native_participation','Shared target32'),('native_exclusive','Exclusive target32')]:
            a=fmean(lookup[s,task,'whole',method]['donor_oriented_change_mean'] for s in range(1,6))
            b=100*fmean(lookup[s,task,'whole',method]['donor_label_correct_mean'] for s in range(1,6))
            lines.append(label+' & '+ml+f' & {a:.4f} & {b:.2f}'+r' \\')
    (out/'tables/participation_source.tex').write_text('\n'.join(lines)+'\n')
    # Exact example table gives observed effects for all four fixed contexts.
    def tex(s):
        for a,b in [('&',r'\&'),('_',r'\_'),('%',r'\%'),('#',r'\#')]:s=str(s).replace(a,b)
        return s
    lines=[]
    for e,label in zip(examples,LABELS):
        p=e['panel'];d=e['donor'];obs={(r['operation'],r['method']):r for r in e['observations']}
        spans=[tex(s) for s in p['spans'][1:]]
        spans[p['changed_region']-1]=r'\emph{'+spans[p['changed_region']-1]+'}'
        context=label+': '+''.join(spans)+r'\newline '+tex(p['spans'][p['changed_region']].strip())+r' $\to$ '+tex(d['spans'][d['changed_region']].strip())
        vals=[obs['whole',m]['donor_oriented_change'] for m in ['source_selected','native_participation','native_exclusive']]
        vals += [obs['part0','native_participation']['kl_to_reference'],obs['part1','native_participation']['kl_to_reference']]
        lines.append(context+' & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\[5pt]')
    (out/'tables/participation_examples.tex').write_text('\n'.join(lines)+'\n')
    return data


def plot(data,save):
    if not data:return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    green,purple,ink,gray='#286956','#785481','#262626','#81837d'
    labels=['Gender','Filler–gap','NPI licensing','NP/Z']
    fig,axs=plt.subplots(1,2,figsize=(7,2.85),gridspec_kw={'width_ratios':[1,1.03]})
    fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.83,wspace=.39)
    for i,task in enumerate(TASKS):
        rows=[r for r in data['source_coverage'] if r['task']==task and r['seed']>0]
        for r in rows:
            y=i+(r['seed']-3)*.047
            vals=[r[k]['donor_oriented_change_mean'] for k in ['single','multi']]
            axs[0].plot(vals,[y,y],color=gray,lw=.6,alpha=.7)
            axs[0].plot(vals[0],y,'o',color=gray,ms=3,mfc='white')
            axs[0].plot(vals[1],y,'o',color=green,ms=3)
        raw=next(r for r in data['source_coverage'] if r['task']==task and r['seed']==0)
        axs[0].plot(raw['multi']['donor_oriented_change_mean'],i,'|',color=ink,ms=12,mew=1.2)
        for j,(method,color,marker) in enumerate([('source_signed_16',gray,'o'),('source_selected',green,'s'),('source_signed_64',purple,'^'),('full_source_sae',ink,'|')]):
            vals=[r['donor_oriented_change_mean'] for r in data['cells'] if r['task']==task and r['operation']=='whole' and r['method']==method]
            y=i+(j-1.5)*.15
            axs[1].plot([min(vals),max(vals)],[y,y],color=color,lw=.85)
            axs[1].plot(fmean(vals),y,marker=marker,color=color,ms=4)
    for ax in axs:
        ax.set(yticks=range(4),ylim=(3.47,-.47),xlim=(0,6.5),xlabel='Donor-oriented effect (nat)')
        ax.tick_params(axis='y',length=0)
    axs[0].set_yticklabels(labels);axs[1].set_yticklabels([])
    axs[0].set_title('(a) Position coverage',loc='left',pad=22)
    axs[1].set_title('(b) One shared source group',loc='left',pad=22)
    axs[0].legend(handles=[Line2D([],[],marker='o',mfc='white',color=gray,lw=0,label='Best single'),
         Line2D([],[],marker='o',color=green,lw=0,label='Multisite'),Line2D([],[],marker='|',color=ink,lw=0,label='Raw multi')],
         loc='lower left',bbox_to_anchor=(-.12,1.005),ncol=3,fontsize=7.2,frameon=False,columnspacing=.7,handletextpad=.2)
    axs[1].legend(handles=[Line2D([],[],marker=m,color=c,lw=0,label=l) for m,c,l in [('o',gray,'16'),('s',green,'32'),('^',purple,'64'),('|',ink,'All')]],
         loc='lower left',bbox_to_anchor=(-.05,1.005),ncol=4,fontsize=7.2,frameon=False,columnspacing=.8,handletextpad=.2)
    save(fig,'multisite_source_coverage')

    # Log-KL dot intervals preserve task scale and show all dependent directions.
    fig,axs=plt.subplots(1,3,figsize=(7,3.0),sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.21,top=.79,wspace=.25)
    styles=[('native_participation',green,'o','Shared native'),('native_exclusive',purple,'s','Exclusive native'),
            ('global_hungarian_native',gray,'^','Geometry native'),('compact_response_fit',ink,'x','Compact readout')]
    for ax,ops,title in zip(axs,[['whole'],['part0','part1'],['half_dose','mixed_dose']],['(a) Whole','(b) Individual parts','(c) Untrained doses']):
        for i,task in enumerate(TASKS):
            for j,(method,color,marker,_) in enumerate(styles):
                vals=[fmean(r['kl_to_reference_mean'] for r in data['cells'] if r['source_seed']==s and r['task']==task and r['operation'] in ops and r['method']==method) for s in range(1,6)]
                y=i+(j-1.5)*.15
                ax.plot([min(vals),max(vals)],[y,y],color=color,lw=.75)
                ax.plot(fmean(vals),y,marker=marker,color=color,ms=3.5)
        ax.set(xscale='log',xlim=(1e-4,.8),xticks=[.0001,.001,.01,.1],ylim=(3.5,-.5),yticks=range(4),xlabel='Source-to-target KL (nat)')
        ax.set_title(title,loc='left',pad=7)
        ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=7)
        for y in [.5,1.5,2.5]:ax.axhline(y,color='#ecece7',lw=.5,zorder=0)
    axs[0].set_yticklabels(labels)
    fig.legend(handles=[Line2D([],[],marker=m,color=c,lw=.8,label=l,ms=4) for _,c,m,l in styles],
               loc='upper center',bbox_to_anchor=(.52,.995),ncol=4,frameon=False,fontsize=7.3,columnspacing=1.1)
    save(fig,'native_participation_fidelity')

    fig=plt.figure(figsize=(7,2.55))
    grid=fig.add_gridspec(1,2,width_ratios=[.7,1.8],left=.09,right=.98,bottom=.25,top=.80,wspace=.53)
    ax=fig.add_subplot(grid[0]);g=np.array(data['matrix_example']['target_participation']);idx=np.flatnonzero(np.any(g!=0,axis=1))
    # At most32 selected target rows at true paper width, not an 8192-cell atlas.
    cmap=LinearSegmentedColormap.from_list('participation',['#ffffff',green])
    ax.imshow(g[idx].T,aspect='auto',vmin=0,vmax=1,cmap=cmap,interpolation='none')
    ax.set(yticks=[0,1],yticklabels=['Part 0','Part 1'],xticks=[0,len(idx)-1],xticklabels=['1',str(len(idx))],xlabel='Selected target members')
    ax.set_title('(a) NPI: one target group',loc='left',pad=13)
    ax.tick_params(axis='both',length=0)
    ax.text(.5,-.34,'White: 0    Dark green: 1',transform=ax.transAxes,ha='center',fontsize=7)
    bx=fig.add_subplot(grid[1]);families=['whole','parts','untrained_doses']
    for i,task in enumerate(TASKS):
        for j,family in enumerate(families):
            r=next(r for r in data['comparisons'] if r['task']==task and r['operation_family']==family and r['baseline']=='native_exclusive')
            y=i+(j-1)*.18;v=100*r['reduction_fraction'];lo,hi=np.array(r['observed_component_bootstrap95'])*100
            bx.plot([lo,hi],[y,y],color=[gray,green,purple][j],lw=.85)
            bx.plot(v,y,marker=['o','s','^'][j],color=[gray,green,purple][j],ms=3.5)
    bx.axvline(0,color=ink,lw=.65);bx.set(yticks=range(4),yticklabels=labels,ylim=(3.4,-.4),xlabel='KL reduction vs exclusive native (%)')
    bx.tick_params(axis='y',length=0);bx.set_title('(b) Sharing across source controls',loc='left',pad=13)
    bx.legend(handles=[Line2D([],[],marker=m,color=c,lw=0,label=l) for m,c,l in [('o',gray,'Whole'),('s',green,'Parts'),('^',purple,'Doses')]],loc='upper left',bbox_to_anchor=(0,-.30),frameon=False,ncol=3,fontsize=7.3)
    save(fig,'native_participation_sharing')
