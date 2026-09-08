"""Source-backed semantic material and independently named control evidence."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from summarize_ravel_development import summarize

ATTRS=['Country','Continent','Language']
RUNS=[
 ('FINAL5_R16_ravel_coverage_l3_v1_20260908','Early 4M, k64'),
 ('FINAL5_R16_ravel_coverage_k128_v1_20260908','Early 4M, k128'),
 ('FINAL5_R16_l7_4m_source_coverage_five_v1_20260908','Middle 4M, k128'),
 ('FINAL5_R16_l7_4m_source_coverage_gpu_five_v2_20260908','Middle 4M, k128 (GPU)'),
 ('FINAL5_R16_l7_16m_source_coverage_five_v1_20260908','Middle 16M, k128'),
 ('FINAL5_R16_ravel_semantic_source_v1_20260908','Early: 480 updates'),
 ('FINAL5_R16_ravel_raw_semantic_l7_v1_20260908','Middle: 480 updates'),
 ('FINAL5_R16_l7_4m_semantic512_v2_20260908','Middle 4M: 384 updates'),
 ('FINAL5_R16_l7_16m_semantic512_v3_20260908','Middle 16M: 1536 updates')]
LABELS={'native_shared':'Shared native','native_exclusive':'Exclusive native',
        'sae_mdbm':'SAE DBM','mdbm':'Raw DBM','mdas':'Raw DAS'}


def tex(value):
    value=str(value)
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:value=value.replace(a,b)
    return value


def export(root,out,read):
    import numpy as np
    coverage=[];semantic=[];fits=[];examples=[];differences=[];inputs=[];inventory=[];unedited=None
    for name,label in RUNS:
        base=Path('runs')/name
        if not (root/base/'semantic_summary.json').is_file():continue
        summary=read(base/'metrics.summary.json')
        assert summary['status']=='PASS' and read(base/'contract_validation.json')['ok']
        cfg=read(base/'config.resolved.json');panel=read(base/'panel.json')['rows']
        path=root/base/'metrics.raw.jsonl';payload=path.read_bytes()
        assert hashlib.sha256(payload).hexdigest()==summary['metrics_raw_sha256']
        raw=[json.loads(line) for line in payload.splitlines()]
        assert len(raw)==summary['rows']
        inputs += [(base/f).as_posix() for f in ['metrics.summary.json','metrics.raw.jsonl','config.resolved.json','panel.json','code_hashes.json']]
        correction_path=base/'METADATA_CORRECTION.json'
        correction=read(correction_path) if (root/correction_path).is_file() else None
        if correction:inputs.append(correction_path.as_posix())
        inventory.append(dict(run_id=name,label=label,layer=cfg['layer'],config=cfg,summary=summary,metadata_correction=correction))
        if 'coverage' in name:
            cells=read(base/'semantic_summary.json')['cells']
            coverage += [dict(run_id=name,material=label,layer=cfg['layer'],**r) for r in cells if r['mode'] in ['none','region_1']]
            if unedited is None:
                rows=[]
                for r in raw:
                    if r['method']!='unedited' or r['split']!='held_component_development':continue
                    p=panel[r['row_id']]
                    for a in range(3):
                        cause=ATTRS.index(r['task'])==a
                        accepted=p['donor_expected_ids'] if cause else p['expected_ids']
                        rows.append(dict(component=r['component'],endpoint='Cause' if cause else 'Iso',
                            first_token_correct=r['predicted_token_id'] in accepted))
                unedited=dict(run_id=name,**summarize(rows))
            dpath=base/'difference_reconstruction.json'
            if (root/dpath).is_file():
                records=read(dpath)['rows'];inputs.append(dpath.as_posix())
                for seed in cfg['seeds']:
                    for split in ['fit','calibration','held_component_development']:
                        rr=[r for r in records if r['seed']==seed and r['split']==split]
                        if rr:differences.append(dict(run_id=name,material=label,seed=seed,split=split,
                            difference_explained=1-sum(r['squared_difference_error'] for r in rr)/sum(r['raw_energy'] for r in rr)))
            continue
        for kind in cfg['methods']:
            meta=read(base/f'{kind}_fit.json');inputs.append((base/f'{kind}_fit.json').as_posix())
            cost=meta.get('active_weights',meta.get('binary_weights',3*cfg.get('das_rank',0)))
            fits.append(dict(run_id=name,setting=label,method=kind,meta=meta))
            for family,counts in [('single',[1]),('pair',[2]),('triple',[3])]:
                rr=[r for r in raw if r['method']==kind and sum(r['control']) in counts]
                semantic.append(dict(run_id=name,setting=label,layer=cfg['layer'],method=kind,method_label=LABELS[kind],
                    family=family,actual_cost=cost,cost_kind='sum of subspace ranks' if kind=='mdas' else 'positive weights',
                    **summarize(rr),per_attribute={a:summarize([r for r in rr if r['task']==a]) for a in ATTRS}))
            # First retained held city pair, first direction, first template of
            # each attribute. Selection is by stored order, never model outcome.
            first=next(p for p in panel if p['split']=='held_component_development')
            for attr in ATTRS:
                p=next(p for p in panel if p['component']==first['component'] and p['entity']==first['entity'] and p['task']==attr)
                for control in ['100','010','001','111']:
                    row=next(r for r in raw if r['method']==kind and r['row_id']==p['row_id'] and r['operation']==control)
                    examples.append(dict(run_id=name,setting=label,method=kind,task=attr,control=control,
                        text=p['text'],entity=p['entity'],donor_entity=p['donor_entity'],
                        base_expected=p['expected_ids'],donor_expected=p['donor_expected_ids'],
                        base_label=p['label'],donor_label=p['donor_label'],observation=row))
    if not inventory:return None
    training=[];atom_matching=[];split_diagnostic=None
    trainingbase=Path('runs/FINAL5_R16_l7_w16k_k128_train16m_five_v1_20260908')
    if (root/trainingbase/'metrics.summary.json').is_file():
        summary=read(trainingbase/'metrics.summary.json')
        if summary['status']=='PASS':
            payload=(root/trainingbase/'metrics.raw.jsonl').read_bytes()
            assert hashlib.sha256(payload).hexdigest()==summary['metrics_raw_sha256']
            training=[json.loads(line) for line in payload.splitlines()]
            inputs += [(trainingbase/f).as_posix() for f in ['metrics.summary.json','metrics.raw.jsonl','config.resolved.json','training_trace.json','checkpoints.json']]
    atombase=Path('runs/FINAL5_R16_l7_atom_4m16m_five_v1_20260908')
    if (root/atombase/'metrics.summary.json').is_file():
        summary=read(atombase/'metrics.summary.json')
        if summary['status']=='PASS':
            assert read(atombase/'contract_validation.json')['ok']
            payload=(root/atombase/'metrics.raw.jsonl').read_bytes()
            assert hashlib.sha256(payload).hexdigest()==summary['metrics_raw_sha256']
            atom_matching=[json.loads(line) for line in payload.splitlines()]
            inputs += [(atombase/f).as_posix() for f in ['metrics.summary.json','metrics.raw.jsonl','config.resolved.json']]
    splitbase=Path('runs/FINAL5_R16_l7_16m_frozen_source_splits_v2_20260908')
    if (root/splitbase/'metrics.summary.json').is_file():
        ss=read(splitbase/'metrics.summary.json')
        if ss['status']=='PASS':
            assert read(splitbase/'contract_validation.json')['ok']
            payload=(root/splitbase/'metrics.raw.jsonl').read_bytes()
            assert hashlib.sha256(payload).hexdigest()==ss['metrics_raw_sha256']
            records=[json.loads(line) for line in payload.splitlines()];cells=[]
            for method in dict.fromkeys(r['method'] for r in records):
                for split in ['fit','calibration','held_component_development']:
                    rr=[r for r in records if r['method']==method and r['split']==split]
                    cells.append(dict(method=method,split=split,**summarize(rr)))
            split_diagnostic=dict(run_id=splitbase.name,cells=cells,summary=ss,witness=read(splitbase/'replay_witness.json'))
            inputs += [(splitbase/f).as_posix() for f in ['metrics.summary.json','metrics.raw.jsonl','config.resolved.json','replay_witness.json','panel.json']]
    union_analyses=[]
    for material in ['4M','16M']:
        path=Path('artifacts/final_five_research_20260908/r16_semantic_operations')/f'UNION_INTERACTION_{material}.json'
        if (root/path).is_file():
            analysis=read(path)
            assert analysis['source_parent_status']=='PASS'
            union_analyses.append(dict(material=material,**analysis))
            inputs.extend([path.as_posix(),path.with_suffix('.npz').as_posix()])
    material_check_path=Path('artifacts/final_five_research_20260908/r16_semantic_operations/MATCHED_DEVICE_MATERIAL.json')
    material_check=read(material_check_path) if (root/material_check_path).is_file() else None
    if material_check:inputs.append(material_check_path.as_posix())
    data=dict(coverage=coverage,semantic=semantic,fits=fits,examples=examples,differences=differences,training=training,atom_matching=atom_matching,unedited=unedited,split_diagnostic=split_diagnostic,union_analyses=union_analyses,matched_device_material=material_check,
        runs=inventory,input_summary_paths=inputs,
        uncertainty='Eight disjoint held city pairs, clustered across reciprocal directions, templates and attributes. Conditional on the observed source SAE; five-SAE material ranges are not independent-seed confidence intervals.',
        scope='Exposed RAVEL training cities and templates. Alias-aware first-token classification, not full generation or complete original RAVEL. Every material/fitting change retains its config. Natural quality, full-source capture, selective source semantics and cross-seed correspondence are distinct.')
    (out/'data/ravel_semantics.json').write_text(json.dumps(data,indent=2)+'\n')
    for filename,records in [('ravel_material',coverage),('ravel_difference',differences)]:
        with (out/f'data/{filename}.csv').open('w',newline='') as f:
            if records:
                writer=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in records))));writer.writeheader();writer.writerows(records)
    lines=[]
    for name,label in RUNS:
        rr=[r for r in coverage if r['run_id']==name and r['method']=='full_sae' and r['split']=='fit']
        if rr:
            vals=[100*fmean(r['first_token_correct'] for r in rr if r['task']==a) for a in ATTRS]
            seeds=len({r['seed'] for r in rr});lines.append(tex(label)+f' & {seeds} & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    (out/'tables/ravel_material.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for r in semantic:
        if r['family']!='single':continue
        metric=r['disentangle'];lo,hi=metric['conditional_pair_bootstrap95']
        lines.append(tex(r['setting'])+' & '+tex(r['method_label'])+f" & {r['actual_cost']} & {100*r['cause']['mean']:.2f} & {100*r['iso']['mean']:.2f} & {100*metric['mean']:.2f} & [{100*lo:.1f}, {100*hi:.1f}]"+r' \\')
    (out/'tables/ravel_selectivity.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for step in sorted({r['step'] for r in training}):
        rr=[r for r in training if r['step']==step];qq=[r['quality'] for r in rr]
        match=[r['mean_absolute_cosine'] for r in atom_matching if r['step']==step]
        values=[f'{step*1024/1e6:.3f}',f"{100*fmean(q['fve'] for q in qq):.3f}",
            f"{100*fmean(q['ce_recovered'] for q in qq):.3f}",f"{fmean(q['actual_nonzero_l0'] for q in qq):.2f}",
            f"{min(q['alive_features'] for q in qq)}--{max(q['alive_features'] for q in qq)}",
            f"{max(r['decoder_norm_max_error'] for r in rr):.2g}",f'{fmean(match):.4f}' if match else '--']
        lines.append(' & '.join(values)+r' \\')
    (out/'tables/ravel_training.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    if split_diagnostic:
        cells=split_diagnostic['cells']
        for method in dict.fromkeys(r['method'] for r in cells):
            fit=next(r for r in cells if r['method']==method and r['split']=='fit')
            values=[fit[k]['mean'] for k in ['cause','iso','disentangle']]
            values.extend(next(r for r in cells if r['method']==method and r['split']==s)['disentangle']['mean'] for s in ['calibration','held_component_development'])
            lines.append(tex(LABELS[method])+' & '+' & '.join(f'{100*v:.2f}' for v in values)+r' \\')
    (out/'tables/ravel_fit_generalization.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for analysis in union_analyses:
        for kind in analysis['methods']:
            values=[next(r['relative_rms_interaction'] for r in analysis['summary'] if r['method']==kind['method'] and r['control']==c) for c in ['110','101','011','111']]
            lines.append(analysis['material']+' & '+tex(LABELS[kind['method']])+' & '+' & '.join('--' if v is None else f'{100*v:.1f}' for v in values)+r' \\')
    (out/'tables/ravel_union_interaction.tex').write_text('\n'.join(lines)+'\n')
    if examples:
        current=next(name for name,_ in reversed(RUNS) if any(e['run_id']==name for e in examples))
        fixed=[e for e in examples if e['run_id']==current];first=fixed[0];lines=[]
        for kind in dict.fromkeys(e['method'] for e in fixed):
            for c,label in [('100','Country'),('010','Continent'),('001','Language'),('111','All three')]:
                predictions=[]
                for attr in ATTRS:
                    e=next(e for e in fixed if e['method']==kind and e['control']==c and e['task']==attr)
                    o=e['observation'];mark=r'\checkmark' if o['first_token_correct'] else r'\times'
                    predictions.append(tex(o['predicted_token_text'].strip() or '(empty)')+r' $'+mark+'$')
                lines.append((tex(LABELS[kind]) if c=='100' else '')+' & '+label+' & '+' & '.join(predictions)+r' \\')
            lines.append(r'\addlinespace[3pt]')
        (out/'tables/ravel_fixed_case.tex').write_text('\n'.join(lines)+'\n')
        facts=[];prompts=[]
        for attr in ATTRS:
            e=next(e for e in fixed if e['task']==attr)
            facts.append(attr+': '+tex(e['base_label'].strip())+r'$\to$'+tex(e['donor_label'].strip()))
            prompts.append(r'\emph{'+tex(e['text'])+'}')
        caption='The fixed base is '+tex(first['entity'])+' and the donor is '+tex(first['donor_entity'])+'. Representative accepted first tokens are '+', '.join(facts)+'. The exact prompts are '+ '; '.join(prompts)+'.'
        (out/'tables/ravel_case_context.tex').write_text(caption+'\n')
    return data


def plot(data,save):
    if not data:return
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    green,purple,ink,gray='#286956','#785481','#262626','#81837d'
    materials=[]
    for _,label in RUNS:
        if any(r['material']==label and r['method']=='full_sae' for r in data['coverage']):materials.append(label)
    fig,axes=plt.subplots(1,3,figsize=(7,2.45),sharex=True,sharey=True)
    fig.subplots_adjust(left=.23,right=.985,bottom=.23,top=.85,wspace=.16)
    for ax,attr in zip(axes,ATTRS):
        for i,label in enumerate(materials):
            rows=[r for r in data['coverage'] if r['material']==label and r['task']==attr and r['method']=='full_sae' and r['split']=='fit']
            vals=np.array([r['first_token_correct'] for r in rows])*100
            color=green if 'Middle' in label else gray
            ax.plot([vals.min(),vals.max()],[i,i],color=color,lw=.8)
            for r,v in zip(rows,vals):ax.plot(v,i+(r['seed']-3)*.04 if len(rows)>1 else i,'o',color=color,ms=3,mfc='white' if 'k64' in label else color)
            raw=[r for r in data['coverage'] if r['material']==label and r['task']==attr and r['method']=='raw_donor' and r['split']=='fit']
            if raw:ax.plot(100*raw[0]['first_token_correct'],i,'|',color=ink,ms=10,mew=1)
        ax.set(xlim=(0,100),xticks=[0,25,50,75,100],yticks=range(len(materials)),ylim=(len(materials)-.5,-.5),xlabel='Donor-label success (%)')
        ax.tick_params(axis='y',length=0);ax.set_title(attr,loc='left',pad=10)
        ax.grid(axis='x',color='#eeeeea',lw=.5,zorder=0)
    axes[0].set_yticklabels(materials)
    fig.text(.23,.03,'Circles: complete SAE (individual seeds). Vertical marks: raw replacement at the same layer.',fontsize=7)
    save(fig,'ravel_material_coverage')
    singles=[r for r in data['semantic'] if r['family']=='single']
    fig,axes=plt.subplots(1,3,figsize=(7,max(2.6,.25*len(singles)+.9)),sharey=True)
    fig.subplots_adjust(left=.39,right=.985,bottom=.16,top=.89,wspace=.23)
    labels=[]
    for r in singles:
        setting=r['setting'].replace(': ',' / ').replace(' updates',' steps')
        labels.append(setting+' / '+r['method_label'])
    for ax,metric,title,color in zip(axes,['cause','iso','disentangle'],['Cause (%)','Iso (%)','Mean (%)'],[green,purple,ink]):
        if data.get('unedited'):ax.axvline(100*data['unedited'][metric]['mean'],color='#a8aaa4',lw=.75,ls='--')
        for j,r in enumerate(singles):
            stat=r[metric];mean=100*stat['mean'];lo,hi=np.array(stat['conditional_pair_bootstrap95'])*100
            ax.plot([lo,hi],[j,j],color=color,lw=.85)
            ax.plot(mean,j,'o',ms=3.6,color=color,mfc='white' if r['method']=='native_exclusive' else color)
            if j and singles[j-1]['run_id']!=r['run_id']:
                ax.axhline(j-.5,color='#dfdfda',lw=.55)
        ax.set(xlim=(0,100),xticks=[0,50,100],yticks=range(len(singles)),ylim=(len(singles)-.5,-.5))
        ax.set_title(title,loc='left',pad=7,fontsize=8.2)
        ax.grid(axis='x',color='#eeeeea',lw=.5)
        ax.tick_params(axis='y',length=0,labelsize=6.5)
    axes[0].set_yticklabels(labels)
    fig.text(.39,.035,'Intervals: eight held city pairs, conditional on each source. Dashed: no edit.',fontsize=6.6)
    save(fig,'ravel_cause_isolation')
    if data.get('training'):
        fig,axes=plt.subplots(1,3,figsize=(7,2.55))
        fig.subplots_adjust(left=.09,right=.98,bottom=.24,top=.86,wspace=.42)
        for ax,key,title,color in zip(axes,['fve','ce_recovered','alive_fraction'],['FVE (%)','CE recovered (%)','Alive on validation (%)'],[green,purple,ink]):
            for seed in sorted({r['seed'] for r in data['training']}):
                rr=sorted((r for r in data['training'] if r['seed']==seed),key=lambda r:r['step'])
                x=[r['step']*1024/1e6 for r in rr];y=[100*r['quality'][key] for r in rr]
                ax.plot(x,y,color=color,lw=.7,alpha=.7,marker='o',ms=2.5)
            ax.set_xscale('log',base=2);ax.set_xticks([.25,1,4,16],['0.25','1','4','16'])
            ax.set_xlabel('Training tokens (M)');ax.set_title(title,loc='left',pad=9,fontsize=8.2)
        save(fig,'ravel_training_quality')
