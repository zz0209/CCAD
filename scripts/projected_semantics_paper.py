"""Export measured projected-source/readout/native-writing evidence."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import numpy as np
from summarize_ravel_development import summarize

PREFIX='FINAL5_R17_l7_'
SINGLES=['100','010','001']
ATTRS=['Country','Continent','Language']
LABELS={'frozen_source':'Frozen source','direct_shared_space':'Shared-space readout',
 'atom_pw_mcc':'PW-MCC readout','anchor_family_r128':'Family correction (128)',
 'anchor_state_r128':'State correction (128)','anchor_full':'Full correction',
 'unanchored_full':'Full ridge','unanchored_family_r128':'Family rank 128',
 'target_reencode':'Re-encode','adaptive_native64':'Native 64',
 'adaptive_native256':'Native 256','adaptive_native512':'Native 512',
 'random_base_native256':'Random + base 256','wrong_attribute_shared_space':'Wrong attribute',
 'raw_supervised_das':'Raw supervised DAS','source_reverse_order':'Reverse order'}
LABELS.update({'unanchored_full_ridge':'Full ridge','anchored_full_ridge':'Full correction',
 'anchored_state_rank32':'State correction (32)','anchored_state_rank128':'State correction (128)',
 'anchored_family_rank32':'Family correction (32)','anchored_family_rank128':'Family correction (128)',
 'unanchored_family_rank32':'Family rank 32','unanchored_family_rank128':'Family rank 128'})


def family(r):
    if r['operation'] in SINGLES:return 'single'
    return 'joint' if all(v in [0,1] for v in r['control']) else 'fractional'


def aggregate(rows):
    result=summarize(rows)
    result['source_kl']=fmean(r['kl_to_reference'] for r in rows) if rows else None
    # The source's first fractional forward defaults to natural donor as reference.
    # It is a producer, so no self-KL is inferred from that value.
    if all(r['method']=='frozen_source' for r in rows):result['source_kl']=0.
    return result


def export(root,out,read):
    inputs=[];runs=[];source=[];transfer=[];native=[];native_diag=[];examples=[];raw_transfer=[]
    def get(name):
        base=Path('runs')/name;s=read(base/'metrics.summary.json')
        assert s['status']=='PASS' and read(base/'contract_validation.json')['ok']
        payload=(root/base/'metrics.raw.jsonl').read_bytes()
        assert hashlib.sha256(payload).hexdigest()==s['metrics_raw_sha256']
        rr=[json.loads(x) for x in payload.splitlines()];assert len(rr)==s['rows']
        paths=[(base/x).as_posix() for x in ['metrics.summary.json','metrics.raw.jsonl','status.json','config.resolved.json','code_hashes.json','panel.json']]
        inputs.extend(paths);runs.append(dict(run_id=name,summary={k:v for k,v in s.items() if k!='cells'},status=read(base/'status.json')))
        return base,rr
    for name,setting in [('FINAL5_R16_l7_16m_semantic512_v3_20260908','48 cities / native gates and raw'),
        (PREFIX+'16m_decoded_das_v1_20260908','48 cities / decoded DAS'),
        (PREFIX+'16m_diverse_das_v1_20260908','176 cities / DAS')]:
        base,rr=get(name)
        for method in dict.fromkeys(r['method'] for r in rr if r['kind']=='semantic'):
            ss=[r for r in rr if r['kind']=='semantic' and r['method']==method and r['operation'] in SINGLES]
            if ss:source.append(dict(run_id=name,setting=setting,method=method,**aggregate(ss)))
    base,rr=get(PREFIX+'projected_transfer_v1_20260908');raw_transfer=[r for r in rr if r['kind']=='semantic']
    grouped=defaultdict(list)
    for r in raw_transfer:grouped[(r['method'],r['target_seed'],family(r))].append(r)
    for (m,t,f),ss in grouped.items():transfer.append(dict(method=m,target_seed=t,family=f,**aggregate(ss)))
    for suffix in ['s2','s345']:
        base,rr=get(PREFIX+f'projected_native_{suffix}_v1_20260908')
        if suffix=='s2':fixed=min((r for r in rr if r['method']=='frozen_source'),key=lambda x:(x['component'],x['row_id']))
        grouped=defaultdict(list)
        for r in rr:
            if r['kind']=='semantic':grouped[(r['method'],r['target_seed'],family(r))].append(r)
        for (m,t,f),ss in grouped.items():
            if suffix=='s345' and m=='frozen_source':continue
            native.append(dict(method=m,target_seed=t,family=f,**aggregate(ss)))
        diag=read(base/'native_writer_diagnostics.json');inputs.append((base/'native_writer_diagnostics.json').as_posix())
        for m in dict.fromkeys(r['method'] for r in diag['rows']):
            for t in sorted({r['target_seed'] for r in diag['rows']}):
                ss=[r for r in diag['rows'] if r['method']==m and r['target_seed']==t]
                errors=[v for r in ss for v in r.get('squared_error',r.get('squared_writer_error',[]))]
                energy=[v for r in ss for v in r['desired_energy']]
                native_diag.append(dict(method=m,target_seed=t,mean_changed_members=fmean(v for r in ss for v in r['changed_members']),
                    relative_squared_writer_error=sum(errors)/sum(energy),minimum_final_state=min(r['minimum_final_state'] for r in ss),
                    max_relative_projected_gradient=max([v for r in ss for v in r.get('relative_projected_gradient',[])] or [0]),
                    converged_rows=sum(r.get('converged_rows',0) for r in ss),
                    selection_seconds=sum(r.get('selection_seconds') or 0 for r in ss),solve_seconds=sum(r.get('solve_seconds') or 0 for r in ss)))
        if suffix=='s2':
            panel=read(base/'panel.json')['rows']
            for r in rr:
                if r['kind']!='semantic' or r['component']!=fixed['component'] or r['entity']!=fixed['entity'] or r['operation'] not in SINGLES+['111']:continue
                p=panel[r['row_id']]
                # Keep the first stored template of each task, without outcome selection.
                first=min(x['row_id'] for x in rr if x['kind']=='semantic' and x['component']==fixed['component'] and x['entity']==fixed['entity'] and x['task']==r['task'])
                if r['row_id']==first:examples.append(dict(**r,text=p['text'],expected_ids=p['expected_ids'],donor_expected_ids=p['donor_expected_ids']))
    pooled=[];comparisons=[]
    for section,records in [('readout',transfer),('native',native)]:
        for m in dict.fromkeys(r['method'] for r in records if r['target_seed'] in [2,3,4,5]):
            for f in ['single','joint','fractional']:
                ss=[r for r in records if r['method']==m and r['family']==f and r['target_seed'] in [2,3,4,5]]
                if not ss:continue
                pooled.append(dict(section=section,method=m,family=f,target_seeds=[r['target_seed'] for r in ss],
                    source_kl=fmean(r['source_kl'] for r in ss),
                    **{k:fmean(r[k]['mean'] for r in ss) for k in ['cause','iso','disentangle'] if all(k in r for r in ss)}))
        if section=='native':
            for m in ['target_reencode','adaptive_native64','adaptive_native256','adaptive_native512','random_base_native256']:
                ss=[r for r in records if r['method']==m and r['family']=='single']
                other=[r for r in records if r['method']=='target_reencode' and r['family']=='single']
                if m=='target_reencode':continue
                vals=[]
                for pair in sorted({v['component'] for r in ss for v in r['per_pair']}):
                    a=[v['disentangle'] for r in ss for v in r['per_pair'] if v['component']==pair]
                    b=[v['disentangle'] for r in other for v in r['per_pair'] if v['component']==pair]
                    vals.append(fmean(a)-fmean(b))
                rng=np.random.default_rng(170908);v=np.asarray(vals);samples=rng.integers(0,len(v),(10000,len(v)))
                comparisons.append(dict(method=m,comparator='target_reencode',mean_score_difference=float(v.mean()),
                    conditional_city_pair_bootstrap95=np.quantile(v[samples].mean(1),[.025,.975]).tolist(),
                    leave_one_target_out=[dict(omitted=t,difference=fmean(r['disentangle']['mean'] for r in ss if r['target_seed']!=t)-fmean(r['disentangle']['mean'] for r in other if r['target_seed']!=t)) for t in [2,3,4,5]],
                    scope='Jointly resample eight disjoint city pairs, retaining all target directions per pair. Conditional on source1 and these dictionaries; source1 deletion removes all observations.'))
    composition=read(Path('artifacts/final_five_research_20260908/r17_projected_semantics/ORDERED_COMPOSITION.json'))
    inputs.append('artifacts/final_five_research_20260908/r17_projected_semantics/ORDERED_COMPOSITION.json')
    order=[]
    for c in ['110','101','011','111']:
        ss=[r for r in raw_transfer if r['method']=='source_reverse_order' and r['operation']==c]
        order.append(dict(operation=c,**aggregate(ss)))
    data=dict(source=source,transfer=transfer,native=native,native_diagnostics=native_diag,pooled=pooled,
        native_comparisons=comparisons,ordered_geometry=composition['summary'],order=order,examples=examples,
        input_summary_paths=list(dict.fromkeys(inputs)),run_inventory=runs,
        scope='All city outcomes are exposed development; four targets share source1 and eight city pairs. Fractions use KL to the same frozen source, categorical labels only descriptive. Dynamic native supports are per-input/control, not globally fixed semantic memberships. Ordinary anchored corrections and sparse synthesis are not algorithmic novelty.')
    (out/'data/projected_semantics.json').write_text(json.dumps(data,indent=2)+'\n')
    flat=[]
    for section in ['transfer','native']:
        for r in data[section]:flat.append(dict(section=section,method=r['method'],target_seed=r['target_seed'],family=r['family'],source_kl=r['source_kl'],**{k:r[k]['mean'] for k in ['cause','iso','disentangle'] if k in r}))
    with (out/'data/projected_semantics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['section','method','target_seed','family','cause','iso','disentangle','source_kl']);w.writeheader();w.writerows(flat)
    def table(name,lines):(out/'tables'/name).write_text('\n'.join(lines)+'\n')
    lines=[]
    for r in pooled:
        if r['family']!='single':continue
        operation='readout' if r['method']=='direct_shared_space' else r['section']
        lines.append(LABELS.get(r['method'],r['method']).replace('_',r'\_')+' & '+operation+' & '+' & '.join(f'{100*r[k]:.2f}' for k in ['cause','iso','disentangle'])+f" & {r['source_kl']:.4f}"+r' \\')
    table('projected_transfer.tex',lines)
    lines=[]
    for r in source:
        label={'native_shared':'Shared native','native_exclusive':'Exclusive native','sae_mdbm':'SAE DBM','sae_mdas':'Decoded DAS','mdas':'Raw DAS','mdbm':'Raw DBM'}[r['method']]
        lines.append(('176' if '176' in r['setting'] else '48')+' & '+label+' & '+' & '.join(f"{100*r[k]['mean']:.2f}" for k in ['cause','iso','disentangle'])+r' \\')
    table('projected_sources.tex',lines)
    lines=[]
    for m in ['target_reencode','adaptive_native64','adaptive_native256','adaptive_native512','random_base_native256']:
        rr=[r for r in native_diag if r['method']==m]
        kkt='--' if m=='target_reencode' else f"{max(r['max_relative_projected_gradient'] for r in rr):.2g}"
        lines.append(LABELS[m]+f" & {fmean(r['mean_changed_members'] for r in rr):.1f} & {fmean(r['relative_squared_writer_error'] for r in rr):.4f} & {kkt}"+r' \\')
    table('projected_native_diagnostics.tex',lines)
    lines=[]
    for m in ['frozen_source','direct_shared_space','target_reencode','adaptive_native256','adaptive_native512','random_base_native256']:
        for c in SINGLES+['111']:
            pred=[]
            for attr in ATTRS:
                e=next(r for r in examples if r['method']==m and r['operation']==c and r['task']==attr)
                word=e['predicted_token_text'].strip().replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
                pred.append(word+r' $'+(r'\checkmark' if e['first_token_correct'] else r'\times')+'$')
            lines.append((LABELS[m] if c=='100' else '')+' & '+dict(zip(SINGLES+['111'],ATTRS+['All three']))[c]+' & '+' & '.join(pred)+r' \\')
        lines.append(r'\addlinespace[3pt]')
    table('projected_fixed_case.tex',lines)
    return data


def plot(data,save):
    if not data:return
    import matplotlib.pyplot as plt
    green,purple,ink,gray='#286956','#785481','#262626','#888880'
    fig,axes=plt.subplots(1,2,figsize=(7,3.1));fig.subplots_adjust(left=.19,right=.98,bottom=.22,top=.87,wspace=.78)
    selected=[r for r in data['source'] if r['method'] in ['native_shared','sae_mdas','mdas']]
    selected=[r for r in selected if not(r['method']=='native_shared' and '176' in r['setting'])]
    for i,r in enumerate(selected):
        v=r['disentangle'];lo,hi=np.asarray(v['conditional_pair_bootstrap95'])*100
        color=purple if r['method']=='mdas' else green
        axes[0].plot([lo,hi],[i,i],color=color,lw=.8);axes[0].plot(100*v['mean'],i,'o',ms=4,color=color)
    labels=[('176' if '176'in r['setting'] else '48')+' cities / '+{'native_shared':'native gate','sae_mdas':'decoded DAS','mdas':'raw DAS'}[r['method']] for r in selected]
    axes[0].set(yticks=range(len(selected)),yticklabels=labels,ylim=(len(selected)-.5,-.5),xlim=(20,90),xticks=[25,50,75],xlabel='Mean Cause / Iso (%)')
    axes[0].set_title('(a) Source operation and fit cities',loc='left',pad=10)
    methods=['direct_shared_space','target_reencode','adaptive_native64','adaptive_native256','adaptive_native512','random_base_native256']
    for i,m in enumerate(methods):
        rr=[r for r in data['native'] if r['method']==m and r['family']=='single'];vals=[100*r['disentangle']['mean'] for r in rr]
        axes[1].plot([min(vals),max(vals)],[i,i],color=gray,lw=.7)
        for r,v in zip(rr,vals):axes[1].plot(v,i+(r['target_seed']-3.5)*.055,marker=['o','s','^','D'][r['target_seed']-2],ms=3.7,color=green if m.startswith('adaptive') else ink,mfc='white' if m=='direct_shared_space' else None)
    axes[1].set(yticks=range(len(methods)),yticklabels=[LABELS[m] for m in methods],ylim=(len(methods)-.5,-.5),xlim=(30,70),xticks=[30,50,70],xlabel='Mean Cause / Iso (%)')
    axes[1].set_title('(b) Actual writes in four target SAEs',loc='left',pad=10)
    for ax in axes:ax.tick_params(axis='y',length=0,labelsize=8);ax.grid(axis='x',lw=.4,color='#e8e8e3')
    fig.text(.19,.035,'Left: pair intervals on eight exposed city pairs. Right: individual targets; all share source seed 1.',fontsize=7)
    save(fig,'projected_source_native')
    methods=['direct_shared_space','atom_pw_mcc','target_reencode','adaptive_native64','adaptive_native256','adaptive_native512','random_base_native256']
    fig,axes=plt.subplots(1,2,figsize=(7,3.1),gridspec_kw={'width_ratios':[1,1]});fig.subplots_adjust(left=.19,right=.99,bottom=.2,top=.85,wspace=.75)
    for ax,f,title in zip(axes,['single','fractional'],['(a) Single-attribute operations','(b) Untrained fractional controls']):
        mat=[]
        for m in methods:
            src=data['transfer'] if m=='atom_pw_mcc' else data['native']
            mat.append([next(r['source_kl'] for r in src if r['method']==m and r['target_seed']==t and r['family']==f) for t in [2,3,4,5]])
        mat=np.asarray(mat);ax.imshow(np.log10(np.maximum(mat,.03)),cmap='Greys',vmin=np.log10(.03),vmax=np.log10(2),aspect='auto')
        for i in range(len(methods)):
            for j in range(4):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>.32 else ink)
        ax.set(xticks=range(4),xticklabels=['1 to 2','1 to 3','1 to 4','1 to 5'],yticks=range(len(methods)),yticklabels=[LABELS[m] for m in methods]);ax.tick_params(length=0,labelsize=7.5);ax.set_title(title,loc='left',pad=10)
    fig.text(.19,.035,'Cell values: KL to the identical frozen source control (nats); darker means larger error.',fontsize=7)
    save(fig,'projected_native_fidelity')
