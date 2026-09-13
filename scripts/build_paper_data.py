"""Regenerate manuscript tables and plot data from retained experiment summaries.

This performs no SAE fitting, inference, filtering by outcomes, or new confirmation.
Run from any directory: python scripts/build_paper_data.py --output paper
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('artifacts/seven_round_rebuild_20260906')
PARTS = ['temporal/familiar_cue','temporal/new_cue','quoted/familiar_cue','quoted/new_cue']
NAMES = {
    'fcc_group':'Compact FCC', 'raw_native_units':'Raw ridge',
    'full_code_ridge':'Full-code ridge', 'dense_select':'Dense selection',
    'rrr_rank4':'Rank-4 ridge', 'rrr_rank1':'Rank-1 ridge',
    'one_to_one':'One-to-one assignment', 'conditional_ot':'Conditional OT',
    'das_style_raw_rank1':'DAS-style rank-1', 'dsca_rank1':'dSCA-style rank-1',
    'direct_target_native':'Direct target native', 'same_members_native':'FCC members, native',
    'global_factor_mean':'Global factor mean', 'cue_factor_mean':'Cue factor mean',
    'source_teacher':'Source operation', 'no_op':'No operation',
    'same_members_behavior_gain':'FCC native, behavioral gain',
    'same_members_native_gain':'FCC native, geometric gain',
    'same_members_projected':'FCC members, projected native',
    'direct_target_behavior_gain':'Direct native, behavioral gain',
    'direct_target_native_gain':'Direct native, geometric gain',
    'single_atom':'Best single code predictor', 'random_refit':'Random members + ridge',
    'raw_feature_rms':'Raw ridge, per-coordinate RMS', 'full_native':'Full target native',
    'wrong_factor':'Swapped native factors',
}
MAIN = ['fcc_group','raw_native_units','full_code_ridge','dense_select','rrr_rank4',
        'one_to_one','conditional_ot','das_style_raw_rank1','direct_target_native',
        'same_members_native','global_factor_mean','cue_factor_mean']

def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    fields = list(rows[0])
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'paper')
    args=parser.parse_args();root=args.project_root.resolve();out=args.output.resolve()
    data_dir=out/'data';table_dir=out/'tables'
    data_dir.mkdir(parents=True,exist_ok=True);table_dir.mkdir(parents=True,exist_ok=True)
    sources=[]
    def read(rel):
        rel=Path(rel)
        p=root/rel;b=p.read_bytes();sources.append(dict(path=rel.as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        return json.loads(b)
    context=read(BASE/'r4_l15/R4_L15_CONTEXT_SUMMARY.json')
    mechanism=read(BASE/'r5_mechanism/R5_MECHANISM_SUMMARY.json')
    learning=read(BASE/'r4_l15/FUNCTIONAL_LEARNING_SUMMARY.json')
    training=read(BASE/'r4_l15/TRAINING_SUMMARY.json')
    example=read(BASE/'r5_mechanism/figure_member_example_manifest.json')
    panel=read(BASE/'r4_l15/PANEL_PREDECLARATION.json')
    donor_path=BASE/'r7_science_package/R7_DONOR_SUMMARY.json'
    donor=read(donor_path) if (root/donor_path).exists() else None
    if donor:
        donor_keys=['partition','factor','method','n','kl','accuracy','teacher_agreement',
                    'abs_number_shift','abs_time_shift','signed_number_shift','signed_time_shift','delta_norm']
        write_csv(data_dir/'donor_results.csv',[{k:r[k] for k in donor_keys} for r in donor['rows']])
        donor_lookup={(r['partition'],r['factor'],r['method']):r for r in donor['rows']}
        comparisons={(r['partition'],r['factor'],r['comparator']):r for r in donor['comparisons']}
        lines=[]
        for role in ['temporal','quoted']:
            for factor in ['number','time','joint']:
                values=[donor_lookup[role,factor,m]['kl'] for m in ['fcc_group','fcc_wrong_donor','fcc_wrong_donor_norm_matched']]
                gaps=[r['excess_kl'] for r in comparisons[role,factor,'fcc_wrong_donor_norm_matched']['leave_one_seed_out']]
                lines.append(f"{role.title()} & {factor.title()} & "+' & '.join(f'{v:.4f}' for v in values)+f' & {min(gaps):.4f}--{max(gaps):.4f}'+r' \\')
        (table_dir/'donor_specificity.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        lines=[]
        for factor in ['number','time','joint']:
            for name,label in [('fcc_group','FCC'),('fcc_wrong_donor','Wrong donor'),('fcc_wrong_donor_norm_matched','Wrong donor, same norm')]:
                values=[donor_lookup[part,factor,name]['kl'] for part in PARTS]
                lines.append(factor.title()+' & '+label+' & '+' & '.join(f'{v:.5f}' for v in values)+r' \\')
        (table_dir/'donor_by_cue.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lookup={(x['partition'],x['factor'],x['method']):x for x in context['rows']}
    keys=['partition','factor','method','n','kl','noop_kl','accuracy','teacher_agreement',
          'abs_time_shift','abs_number_shift','signed_time_shift','signed_number_shift','delta_norm']
    write_csv(data_dir/'context_results.csv',[{k:x.get(k,'') for k in keys} for x in context['rows']])
    seed_rows=[]
    for x in context['rows']:
        for s in x.get('source_seed_means',[]):
            seed_rows.append(dict(partition=x['partition'],factor=x['factor'],method=x['method'],
                                  source_seed=s['source_seed'],**{k:s.get(k,'') for k in keys[3:]}))
    write_csv(data_dir/'context_source_seeds.csv',seed_rows)
    def table(methods,factor,metric,fmt):
        lines=[]
        for method in methods:
            if all((part,factor,method) in lookup for part in PARTS):
                label=NAMES[method];vals=[fmt(lookup[part,factor,method][metric]) for part in PARTS]
                if method=='fcc_group':label=r'\textbf{'+label+'}'
                lines.append(label+' & '+' & '.join(vals)+r' \\')
        return '\n'.join(lines)+'\n'
    (table_dir/'joint_main.tex').write_text(table(MAIN,'joint','kl',lambda v:f'{v:.4f}'),encoding='utf-8')
    for factor in ['number','time','joint']:
        order=list(NAMES)
        (table_dir/f'{factor}_all_kl.tex').write_text(table(order,factor,'kl',lambda v:f'{v:.5f}'),encoding='utf-8')
        (table_dir/f'{factor}_all_accuracy.tex').write_text(table(order,factor,'accuracy',lambda v:f'{v*100:.2f}'),encoding='utf-8')
        combined=[]
        for method in order:
            if not all((part,factor,method) in lookup for part in PARTS):continue
            label=NAMES[method]
            if method=='fcc_group':label=r'\textbf{'+label+'}'
            values=[f"{lookup[part,factor,method]['kl']:.5f}" for part in PARTS]
            values += [f"{100*lookup[part,factor,method]['accuracy']:.2f}" for part in PARTS]
            combined.append(label+' & '+' & '.join(values)+r' \\')
        (table_dir/f'{factor}_complete.tex').write_text('\n'.join(combined)+'\n',encoding='utf-8')
    mech={(x['partition'],x['factor'],x['method']):x for x in mechanism['rows']}
    member_table=[]
    for role in ['temporal','quoted']:
        for factor in ['number','time']:
            a=mech[role,factor,'without_member'];b=mech[role,factor,'without_member_norm_control']
            assert a['reference_kind']==b['reference_kind']=='full_fcc'
            member_table.append(f"{role.title()} & {factor.title()} & {a['kl']:.4f} & {b['kl']:.4f} & {a['abs_number_difference_from_fcc']:.3f} & {a['abs_time_difference_from_fcc']:.3f}"+r' \\')
    (table_dir/'member_effects.tex').write_text('\n'.join(member_table)+'\n',encoding='utf-8')
    quality=[]
    for x in training['rows']:
        q=x['quality'];quality.append(dict(seed=x['seed'],step=x['step'],tokens=x['packed_train_tokens'],
            fve=q['fve'],ce_recovered=q['ce_recovered'],l0=q['actual_nonzero_l0'],
            validation_active=q['alive_features'],norm_error=x['decoder_norm_max_error']))
    write_csv(data_dir/'training_quality.csv',quality)
    qlines=[f"{x['seed']} & {x['step']} & {100*x['fve']:.2f} & {100*x['ce_recovered']:.2f} & {x['l0']:.2f} & {x['validation_active']} & {x['norm_error']:.2g}"+r' \\' for x in quality]
    (table_dir/'training_quality.tex').write_text('\n'.join(qlines)+'\n',encoding='utf-8')
    def esc(s):
        text=''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(s))
        chunks=text.split('"')
        return ''.join((('``' if i%2 else "''") if i else '')+part for i,part in enumerate(chunks))
    # Every fixed case, including source and transfer failures, remains visible.
    case_path=root/BASE/'r5_mechanism/fixed_cases.csv'
    with case_path.open(encoding='utf-8',newline='') as f:cases=list(csv.DictReader(f))
    case_lines=[];context_lines=[];seen=set()
    for i,row in enumerate(cases):
        cells=[]
        for key in ['row_id','role','syntax','factor','expected','source','fcc','equal_norm_swap']:
            value=esc({'pp':'PP','object_relative':'OR'}.get(row[key],row[key])) if key=='syntax' else esc(row[key])
            if key in ['source','fcc','equal_norm_swap'] and row[key]!=row['expected']:value=r'\textbf{'+value+'}'
            cells.append(value)
        case_lines.append(' & '.join(cells)+r' \\')
        if (i+1)%3==0 and i+1<len(cases):case_lines.append(r'\addlinespace[2pt]')
        if row['row_id'] not in seen:
            seen.add(row['row_id']);context_lines.append(esc(row['row_id'])+' & '+esc(row['recipient'])+r' \\')
    assert len(cases)==48 and len(seen)==16
    (table_dir/'fixed_case_outputs.tex').write_text('\n'.join(case_lines)+'\n',encoding='utf-8')
    (table_dir/'fixed_case_contexts.tex').write_text('\n'.join(context_lines)+'\n',encoding='utf-8')
    for role in ['temporal','quoted']:
        lines=[];role_cases=[r for r in cases if r['role']==role]
        identifiers=list(dict.fromkeys(r['row_id'] for r in role_cases))
        assert len(identifiers)==8 and len(role_cases)==24
        for row_id in identifiers:
            rows=[r for r in role_cases if r['row_id']==row_id]
            title='Row '+row_id+' ('+{'pp':'PP','object_relative':'OR'}[rows[0]['syntax']]+'): '+rows[0]['recipient']
            lines.append(r'\multicolumn{5}{p{0.965\textwidth}}{'+esc(title)+r'} \\[2pt]')
            for row in rows:
                cells=[]
                for key in ['factor','expected','source','fcc','equal_norm_swap']:
                    value=esc(row[key])
                    if key in ['source','fcc','equal_norm_swap'] and row[key]!=row['expected']:value=r'\textbf{'+value+'}'
                    cells.append(value)
                lines.append(' & '.join(cells)+r' \\')
            if row_id!=identifiers[-1]:lines.append(r'\addlinespace[5pt]')
        (table_dir/f'fixed_cases_{role}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    members=[x for x in example['displayed_values'] if 'member' in x]
    lines=[]
    for x in members:
        lines.append(f"{x['member']} & {100*x['natural_active_fraction']:.2f} & {x['temporal_code_change']:.2f} & {x['quoted_code_change']:.2f} & {x['temporal_removal_time_shift_loss']:.3f} & {x['quoted_removal_time_shift_loss']:.3f}"+r' \\')
    (table_dir/'display_members.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    natural_lines=[]
    for x in members:
        natural_lines.append(f"{x['member']} & "+esc(x['natural_excerpt'])+f" & {100*x['natural_active_fraction']:.2f} & {x['native_predictive_cosine']:.3f}"+r' \\')
    (table_dir/'natural_members.tex').write_text('\n'.join(natural_lines)+'\n',encoding='utf-8')
    # Preserve complete earlier summary tables as exportable evidence, with a
    # short directly comparable subset for the typeset development narrative.
    early2=read(BASE/'R2_RESULT_SUMMARY.json')
    early3=read(BASE/'r3_composition/R3_RESULT_SUMMARY.json')
    early3_behavior=read(BASE/'r3_composition/R3_BEHAVIOR_SUMMARY.json')
    early4=read(BASE/'r4_frozen/R4_FROZEN_SUMMARY.json')
    (data_dir/'earlier_results.json').write_text(json.dumps(dict(single_factor=early2,early_composition=early3,early_behavior=early3_behavior,early_confirmation=early4),indent=2)+'\n',encoding='utf-8')
    e3={(x['detail'][1],x['method']):x for x in early3['macro'] if x['factor']=='joint' and x['detail'][0]=='partition'}
    e3.update({(x['detail'][1],x['method']):x for x in early3_behavior['rows'] if x['factor']=='joint' and x['detail'][0]=='partition'})
    e4={(x['partition'],x['method']):x for x in early4['rows'] if x['factor']=='joint'}
    e_lines=[]
    for method in ['fcc_group','raw_native_units','full_code_ridge','dense_select','rrr_rank4','rrr_rank1','conditional_ot','one_to_one','das_style_raw_rank1','direct_target_native','same_members_native']:
        if ('known_cue',method) in e3 and ('familiar_syntax',method) in e4:
            challenge=[e4[p,method] for p in ['object_relative','subject_late']]
            challenge_kl=sum(r['n']*r['kl'] for r in challenge)/sum(r['n'] for r in challenge)
            vals=[e3['known_cue',method]['mean_kl'],e3['new_cue',method]['mean_kl'],e4['familiar_syntax',method]['kl'],challenge_kl]
            e_lines.append(NAMES[method]+' & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\')
    (table_dir/'early_composition.tex').write_text('\n'.join(e_lines)+'\n',encoding='utf-8')
    fidelity=[]
    for method in ['source_teacher','fcc_group','raw_native_units','full_code_ridge','no_op']:
        for part in PARTS:
            row=lookup[part,'joint',method];seeds=[s['kl'] for s in row['source_seed_means']]
            fidelity.append(f"{NAMES[method]} & {part.replace('temporal/','T: ').replace('quoted/','Q: ').replace('familiar_cue','familiar').replace('new_cue','new')} & {100*row['accuracy']:.2f} & {100*row['teacher_agreement']:.2f} & {min(seeds):.4f}--{max(seeds):.4f}"+r' \\')
    (table_dir/'fidelity_and_accuracy.tex').write_text('\n'.join(fidelity)+'\n',encoding='utf-8')
    fixed=[]
    for row_id in [0,256]:
        matches=[x for x in mechanism['fixed_examples'] if x['row_id']==row_id and x['factor']=='time' and x.get('target_seed')==2 and x['source_seed']==1 and x.get('member',-1)==-1 and x['method'] in ['source_teacher','fcc_group','role_swap_norm_matched']]
        assert len(matches)==3
        row=panel['rows'][row_id]
        fixed.append(dict(row_id=row_id,role=row['cue_role'],text=row['text'].replace('<|endoftext|>',''),
                          donor=panel['rows'][panel['pairs'][row_id]['time']]['text'].replace('<|endoftext|>',''),outputs=matches))
    plot=dict(context_rows=context['rows'],diagnostic_rows=donor['rows'] if donor else [],geometry=mechanism['geometry_ratios'],mechanism_rows=mechanism['rows'],
              memberships=[x for x in mechanism['memberships'] if x['source_seed']==1 and x['target_seed']==2],
              fixed_examples=fixed,display_members=[x for x in example['displayed_values'] if 'member' in x],
              functional_learning=learning['rows'],training_quality=quality)
    pair_base=Path('artifacts/extension_five_20260907/r9_blueprint')
    pair_complete=None
    if (root/pair_base/'R9_PAIR_COMPLETE_SUMMARY.json').is_file():
        pair_complete=read(pair_base/'R9_PAIR_COMPLETE_SUMMARY.json')
        pair_anchor=read(pair_base/'R9_PAIR_ANCHOR_SUMMARY.json')
        extra=[x for x in pair_anchor['rows'] if x['method'].startswith('anchor_')]
        pair_rows=pair_complete['rows']+extra
        plot['pair_completion_rows']=pair_rows
        pair_names={'legacy_contrast':'Original contrast map','legacy_plus_mean':'+ global mean',
            'paired_complete':'Complete-state ridge','paired_balanced':'Energy-balanced ridge',
            'full_code_complete':'Full-code complete ridge','raw_complete':'Raw complete ridge',
            'same_members_native':'Same-member native','anchor_compact':'Paired compact completion',
            'anchor_full_code':'Paired full-code completion','anchor_raw':'Paired raw completion'}
        pair_lookup={(r['factor'],r['consumer'],r['role'],r['method']):r for r in pair_rows}
        for consumer in ['contrast','complete_removal']:
            lines=[]
            for method,label in pair_names.items():
                vals=[pair_lookup[f,consumer,r,method]['kl'] for f in ['number','time'] for r in ['temporal','quoted']]
                if method=='anchor_compact':lines.append(r'\midrule')
                lines.append(label+' & '+' & '.join(f'{v:.5f}' for v in vals)+r' \\')
            (table_dir/f'pair_{consumer}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (data_dir/'pair_completion.json').write_text(json.dumps(dict(complete=pair_complete,anchor=pair_anchor),indent=2)+'\n',encoding='utf-8')
    from complete_support_paper import export as export_complete_support
    complete_support=export_complete_support(root,out,read)
    if complete_support:
        plot['complete_support']=complete_support
    from component_paper import export as export_components
    components=export_components(root,out,read)
    if components:plot['component_reuse']=components
    from state_projection_paper import export as export_states
    states=export_states(root,out,read)
    if states:plot['source_state_projection']=states
    from toy_paper import export as export_toys
    toys=export_toys(root,out,read)
    if toys:plot['learned_superposition']=toys
    from native_transfer_paper import export as export_native
    native=export_native(root,out,read)
    if native:plot['external_native_transfer']=native
    from training_curve_paper import export as export_training32
    training32=export_training32(root,out,read)
    if training32:plot['training32_curve']=training32
    from native_participation_paper import export as export_participation
    participation=export_participation(root,out,read)
    if participation:plot['native_participation']=participation
    from ravel_semantics_paper import export as export_ravel
    ravel=export_ravel(root,out,read)
    if ravel:plot['ravel_semantics']=ravel
    from projected_semantics_paper import export as export_projected
    projected=export_projected(root,out,read)
    if projected:plot['projected_semantics']=projected
    from semantic_confirmation_paper import export as export_confirmation
    confirmation=export_confirmation(root,out,read)
    if confirmation:plot['semantic_confirmation']=confirmation
    from selector_paper import export as export_selector
    selection=export_selector(root,out,read)
    if selection:plot['group_selection']=selection
    from axis_paper import export as export_axis
    axis=export_axis(root,out,read)
    if axis:plot['axis_transfer']=axis
    if (out/'reform_runs.json').exists():
        from reform_paper import export as export_reform
        reform=export_reform(root,out)
        sources.extend(dict(path=v['path'],sha256=v['sha256'],bytes=(root/v['path']).stat().st_size) for v in reform['inputs'])
    (data_dir/'figure_data.json').write_text(json.dumps(plot,indent=2)+'\n',encoding='utf-8')
    for relation in plot['memberships']:
        matrix_rows=[dict(target_member=member,**{f'source_{source}':value for source,value in zip(relation['source_members'],row)})
                     for member,row in zip(relation['members'],relation['source_lift'])]
        write_csv(data_dir/f"signed_relation_s1_t2_{relation['factor']}.csv",matrix_rows)
    for rel in [BASE/'r5_mechanism/fixed_cases.csv',BASE/'r5_mechanism/member_effects.csv',BASE/'r5_mechanism/mechanism_directions.csv']:
        p=root/rel;b=p.read_bytes();(data_dir/p.name).write_bytes(b)
        sources.append(dict(path=rel.as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    outputs=[p for folder in [data_dir,table_dir] for p in folder.iterdir() if p.is_file() and p.name!='DATA_MANIFEST.json']
    manifest=dict(sources=sources,output_files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(outputs)],
       scope='Deterministic extraction of retained summaries; all observed strata and methods retained. No fitting, outcome-based selection, inference, or independent confirmation. Member KL reference is intact FCC; context KL reference is source operation.',
       original_raw_sha256=dict(context=context['source_raw_sha256'],mechanism=mechanism['raw_sha256'],training=training['raw_sha256'],learning=learning['raw_sha256'],
                                donor=donor['run_summary']['metrics_raw_sha256'] if donor else None))
    (data_dir/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    claims=[
        dict(id='operation_and_native_equivalence',paper='Sections 2, A.1-A.3',result='Explicit signed predictor and finite-panel equivalence condition; no semantic uniqueness claim',
             evidence=['paper/sections/appendix_theory.tex','artifacts/seven_round_rebuild_20260906/R5_METHODS_AND_PROOFS.md','src/ccad/factor_correspondence.py']),
        dict(id='rrr_and_composition',paper='Sections 2.3, A.4-A.6',result='Exact penalized rank fit and position-aware composition; standard algebra',
             evidence=['artifacts/seven_round_rebuild_20260906/R3_METHODS_AND_ALGEBRA.md','artifacts/seven_round_rebuild_20260906/R4_METHODS_AND_ALGEBRA.md','src/ccad/factor_correspondence.py']),
        dict(id='controlled_material',paper='Language-model experimental design; Appendix C.1; fig:learning and tab:quality',result='Five controlled late-hook SAEs and fixed-checkpoint quality/function; no convergence certificate',
             evidence=['artifacts/seven_round_rebuild_20260906/r4_l15/TRAINING_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r4_l15/FUNCTIONAL_LEARNING_SUMMARY.json','configs/seven_r4_l15_train_k64_five_v1.json','configs/seven_r4_l15_checkpoint_function_v1.json']),
        dict(id='role_confirmation',paper='Context-sensitive operation reuse; fig:example and fig:roles; Appendix C.2-C.3',result='New-role source operation reuse; raw/full more accurate; source mistakes retained',
             evidence=['artifacts/seven_round_rebuild_20260906/r4_l15/PANEL_PREDECLARATION.json','artifacts/seven_round_rebuild_20260906/r4_l15/frozen/FREEZE.json','artifacts/seven_round_rebuild_20260906/r4_l15/R4_L15_CONTEXT_SUMMARY.json','configs/seven_r4_l15_confirmation_v1.json','scripts/run_frozen_composition.py']),
        dict(id='early_positive_and_counterexample',paper='The early-hook counterexample; Appendix C.4; tab:early',result='Early development positive, weaker frozen confirmation, and prefix-rank limitation all preserved',
             evidence=['artifacts/seven_round_rebuild_20260906/R2_RESULT_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r3_composition/R3_RESULT_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r4_frozen/R4_FROZEN_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r4_frozen/R4_PREFIX_DIAGNOSTIC.json']),
        dict(id='member_mechanism',paper='Inspecting the components; fig:members and fig:member_cases; Appendix C.5',result='Individual attenuation, norm-matched swaps and complete member removal; follow-up on exposed panel',
             evidence=['artifacts/seven_round_rebuild_20260906/r5_mechanism/R5_MECHANISM_SUMMARY.json','configs/seven_r5_member_mechanism_v1.json','scripts/run_member_mechanism.py','scripts/summarize_member_mechanism.py']),
        dict(id='fixed_examples_and_signed_relations',paper='fig:example and fig:member_cases; signed-relation results; Appendix C.5-C.7',result='All 16 fixed contexts/48 operations, complete signed matrices and heterogeneous natural examples',
             evidence=['artifacts/seven_round_rebuild_20260906/r5_mechanism/fixed_cases.csv','artifacts/seven_round_rebuild_20260906/r5_mechanism/figure_member_example_manifest.json','configs/seven_r5_member_natural_contexts_v1.json','scripts/run_member_natural_contexts.py']),
        dict(id='swapped_native_control_correction',paper='Appendix B.4; complete comparisons in C.2',result='Historical wrong_factor swaps native group operations; at the same position its joint edit equals direct native. It is not an independent joint-specificity test.',
             evidence=['scripts/run_frozen_composition.py','scripts/run_composition_correspondence.py','paper/sections/appendix_methods.tex']),
    ]
    if donor:
        claims.extend([
            dict(id='frozen_map_donor_specificity',paper='fig:roles; wrong-donor diagnostic; Appendix A.8 and C.8',
                 result='Wrong-factor code inputs increase source KL, including after matching total physical edit norm. Follow-up on the exposed panel, not new confirmation or unique semantic identification.',
                 evidence=[donor_path.as_posix(),'configs/seven_r7_donor_specificity_v1.json','scripts/run_donor_specificity.py','scripts/summarize_donor_specificity.py','paper/sections/appendix_theory.tex']),
            dict(id='executable_predictive_operation',paper='Appendix D.2',
                 result='Forty portable signed maps reproduce frozen physical operations; output is a source-aligned residual update, not native feature deletion.',
                 evidence=['src/ccad/predictive_operation.py','scripts/apply_predictive_operation.py','runs/SEVEN_R7_donor_specificity_v1_20260907/operations/INDEX.json','tests/test_predictive_operation.py'])])
    if pair_complete:
        claims.append(dict(id='contrast_and_pair_common_completion',paper='Completing the source contribution; Appendix A.9 and C.9',
            result='Contrasts omit pair-common contribution. Complete ridge improves source-group removal but can worsen contrasts. Pair-conditioned completion preserves frozen contrasts and improves removal on exposed development data; needs a donor and does not solve unpaired/native/semantic correspondence.',
            evidence=[(pair_base/'R9_PAIR_COMPLETE_SUMMARY.json').as_posix(),(pair_base/'R9_PAIR_ANCHOR_SUMMARY.json').as_posix(),
                      'src/ccad/pair_complete_correspondence.py','scripts/run_pair_complete_correspondence.py','scripts/summarize_pair_complete.py',
                      'configs/ext_r9_pair_complete_v2.json','configs/ext_r9_pair_anchor_v1.json','paper/sections/appendix_theory.tex',
                      'scripts/apply_paired_completion.py','scripts/export_paired_completion.py',(pair_base/'operations/INDEX.json').as_posix()]))
    if complete_support:
        claims.append(dict(id='complete_support_and_controlled_target_training',paper='From a contrast to a complete contribution; Selecting for the complete contribution; Appendix A.10 and C.10',
            result='At the original4M target checkpoint, classical shared OLS improves both consumers over same-budget dense selection on exposed development. Continued targets improve number but worsen time correspondence under fixed4M teachers; reconstruction and source function also improve, so they do not establish monotone transfer quality.',
            evidence=complete_support['input_summary_paths']+['src/ccad/complete_support.py','scripts/run_complete_support.py','scripts/run_continued_code_cache.py','scripts/summarize_continued_material.py',
                'configs/ext_r10_complete_support_v1.json','configs/ext_r10_target8m_support_v1.json','configs/ext_r10_l15_continue8m_five_v1.json','paper/sections/appendix_theory.tex']))
    if components:
        claims.append(dict(id='frozen_source_component_reuse',paper='Section on reusing specified source members; component theory and results appendices',
            result='A frozen allocation map exposes source-member removal and contrast. A fresh authored panel evaluates six partial masks and two additional doses; failed component-metric development and stronger full/raw controls remain. Operational source identity is not target-native or human semantic identity.',
            evidence=components['input_summary_paths']+['src/ccad/component_correspondence.py','src/ccad/component_operation.py','scripts/run_component_correspondence.py','scripts/apply_component_operation.py','scripts/summarize_component_correspondence.py','configs/ext_r11_component_confirmation_v1.json','paper/sections/appendix_theory.tex']))
    if states:
        claims.append(dict(id='admissible_source_states_and_finite_loss',paper='Source-state subsection and theory/method/results appendices',
            result='Classical nonnegative projection constrains absolute source allocations before signed operations; its state-metric guarantee does not imply contrast or finite-KL improvement. Frozen fresh-input comparison retains identical constraints for all strong controls, and finite-output development failures remain.',
            evidence=states['input_summary_paths']+['src/ccad/source_state_projection.py','src/ccad/finite_output_fit.py','scripts/run_source_state_correspondence.py','scripts/run_finite_output_correspondence.py','configs/ext_r12_state_confirmation_v1.json','paper/sections/appendix_theory.tex']))
    if toys:
        claims.append(dict(id='learned_superposition_fidelity_and_truth',paper='Learned-toy main section; toy theory, methods, result and reproduction appendices',
            result='Two new trained toy seeds and five controlled SAEs per material confirm fixed-support finite ReLU fitting while preserving stronger full-code controls. A controlled L1 change improves the specified source grouping against latent-factor truth while lowering FVE; this does not identify optimal semantic information or replace the real LM evidence.',
            evidence=toys['input_summary_paths']+['src/ccad/toy_superposition.py','src/ccad/rectified_operation.py','scripts/run_r13_learned_superposition.py','scripts/apply_rectified_operation.py','scripts/export_r13_operations.py','configs/insert_r13_frozen_low_l1_v1.json','configs/insert_r13_frozen_high_l1_v1.json','paper/sections/toy_theory.tex']))
    if native:
        claims.append(dict(id='target_native_external_operations',paper='Native writing method and external-task results; native-writing appendices',
            result='Separate reading and native writing supports, nonnegative target-state increments, and equally supervised finite writers on four CausalGym train tasks at nonfinal layer-3 sites. One dictionary direction and seven evaluation prompt components per task are exploratory; source weakness, full/raw controls and untrained operation outcomes remain.',
            evidence=native['input_summary_paths']+['src/ccad/native_operation.py','scripts/run_causalgym_native_transfer.py',
                'scripts/native_transfer_paper.py','configs/final5_r14_causalgym_native_cpu_v2.json','configs/final5_r14_causalgym_source_ig16_v3.json',
                'configs/final5_r14_causalgym_finite_writer_v4.json','paper/sections/native_theory.tex','paper/sections/native_methods.tex']))
        if native.get('semantic'):
            claims[-1]['evidence'] += ['src/ccad/semantic_context_matching.py','scripts/prepare_semantic_context_matches.py',
                'configs/final5_r14_semantic_contexts_v2.json','configs/final5_r14_causalgym_semantic_controls_v5.json']
    if training32:
        claims.append(dict(id='controlled_training_to32m',paper='Training trajectory and fixed-teacher transfer; Appendix training continuation',
            result='Five controlled dictionaries continued from8M to32M with fresh natural tokens and a declared new LR phase. Quality, all-atom geometry, reselected source function and fixed-teacher target transfer are distinct endpoints; direction and checkpoint dependence remain explicit.',
            evidence=training32['input_summary_paths']+['scripts/training_curve_paper.py','scripts/run_training_checkpoint_curve.py',
                'scripts/run_checkpoint_atom_matching.py','scripts/run_composition_checkpoint_function.py',
                'configs/final5_r14_l15_continue32m_five_v1.json','configs/final5_r14_continued_source_function_v1.json']))
    if participation:
        claims.append(dict(id='multisite_native_participation',paper='External source coverage and native participation; native appendices',
            result='Five controlled cyclic SAE pairs on four exposed CausalGym train tasks. Source-only shared-member selection, feasible target participation and independently optimized exclusive rows test source whole, parts and untrained doses. Strong readouts and all weak-source outcomes remain; interleaved source parts do not establish independent semantic variables or a general matcher advantage.',
            evidence=participation['input_summary_paths']+['src/ccad/native_participation.py','scripts/run_causalgym_multisite.py',
                'scripts/run_causalgym_native_participation.py','scripts/native_participation_paper.py',
                'configs/final5_r15_multisite_source_v2.json','paper/sections/native_theory.tex']+
                [f'configs/final5_r15_participation_five_s{s}_v2.json' for s in range(1,6)]))
    if ravel:
        claims.append(dict(id='independent_semantic_controls_and_material',paper='RAVEL semantic controls and union-family appendix',
            result='Source-only named attribute controls distinguish complete-SAE material capture from selective Cause/Iso performance. Exposed city-pair development, declared fitting budgets and local RAVEL adapters remain separate from cross-seed fidelity and independent confirmation. The finite union-cube characterization is a hook-space statement.',
            evidence=ravel['input_summary_paths']+['scripts/ravel_semantics_paper.py','scripts/run_ravel_semantic_source.py',
                'scripts/run_ravel_source_coverage.py','src/ccad/semantic_participation.py','src/ccad/union_family.py',
                'src/ccad/ravel_controls.py','paper/sections/semantic_theory.tex']))
    if projected:
        claims.append(dict(id='projected_semantic_source_and_native_use',paper='Projected semantic source, ordered-family theory and native-use comparison',
            result='Decoded difference DAS separates source operation restrictions from retained information. Four target seeds reuse one fixed source; dynamic nonnegative native writes compare to readout, PW-MCC, reencoding and random members. Natural learned corrections do not establish an advantage. All city results are development and sparse synthesis is a classical component.',
            evidence=projected['input_summary_paths']+['scripts/projected_semantics_paper.py','scripts/evaluate_projected_native_writer.py','scripts/evaluate_projected_family_transfer.py',
                'scripts/fit_projected_family_correspondence.py','src/ccad/projected_correspondence.py','src/ccad/native_operation.py','paper/sections/semantic_theory.tex']))
    if confirmation:
        claims.append(dict(id='frozen_named_operation_family_and_native_writes',paper='Named operation families, frozen city confirmation and semantic appendices',
            result='Source calibration selects among matched independent/common projection families and prior singleton fits; the selected procedure is replicated across five controlled source dictionaries. A separate new-city panel tests complete source behavior, raw/decoded/PW-MCC readers, actual feasible native writes and exact reencoding-count budgets. Source validity, fidelity, computation and dependence remain separate; the full original RAVEL benchmark and stable semantic groups are not claimed.',
            evidence=confirmation['input_summary_paths']+['scripts/semantic_confirmation_paper.py','scripts/summarize_semantic_sources.py','scripts/summarize_semantic_confirmation.py',
                'scripts/run_semantic_family_source.py','scripts/fit_semantic_cycle_atoms.py','scripts/freeze_semantic_confirmation.py','scripts/evaluate_semantic_confirmation.py',
                'src/ccad/semantic_family.py','src/ccad/semantic_readout.py','scripts/apply_semantic_family.py','paper/sections/semantic_theory.tex']))
    if selection:
        claims.append(dict(id='source_conditioned_candidate_selection',paper='Selection methods and original CausalGym development',result='Two SAE objectives on GPT2Medium, five controlled seeds each. Fixed native candidate groups are selected before held-dev candidate execution. Descriptive choice/ranking/calibration and direct-trial controls retain source failures; development is not official-test confirmation.',evidence=selection['input_summary_paths']+['scripts/run_causalgym_group_selector.py','scripts/summarize_group_selection.py','src/ccad/native_group_selection.py','src/ccad/causalgym_interface.py','paper/sections/selection_theory.tex']))
    if axis:
        claims.append(dict(id='source_axis_native_transfer_and_selection',paper='Original-task source-axis transfer, complete native comparisons and selection decisions',result=axis['scope'],evidence=axis['input_summary_paths']+['scripts/axis_paper.py','scripts/summarize_group_selection.py','scripts/summarize_selection_dependence.py','src/ccad/axis_transfer_selection.py','src/ccad/selection_budget.py','src/ccad/finite_native_group.py','paper/sections/selection_theory.tex']))
        if axis.get('confirmation_dependence'):
            claims.append(dict(id='complete_frozen_original_task_confirmation',paper='Main fixed-method and dependence tables; complete original-test appendix',result='All435 query/objective/edge cells across three fixed model/SAE conditions. Compiled native writing improves over dynamic native writing with positive observed-node/task/frame sensitivity intervals in all three. Matched unrestricted accuracy remains higher in the point estimates. Source screening beats halving at the primary allowance in two conditions with positive intervals, but does not beat fixed compiled deployment. This is conditional confirmation, not general SAE/model/corpus population inference.',evidence=axis['input_summary_paths']+['scripts/summarize_frozen_axis.py','scripts/summarize_lexical_dependencies.py','paper/sections/main_text.tex','paper/sections/axis_results.tex','paper/sections/appendix_evidence.tex']))
        if axis.get('objective_comparisons'):
            claims.append(dict(id='compiled_fitting_objective_tradeoff',paper='Output fitting and its relation to hidden realization error',result='All six exposed development queries have larger error against the same requested hidden readout after output fitting, but lower source-output KL and higher IIA. Source axis, target reader and write supports are identical; source-output supervision differs. This does not isolate anisotropy from reading-error compensation or establish independent confirmation.',evidence=axis['input_summary_paths']+['scripts/summarize_compiled_objective.py','paper/sections/axis_results.tex','paper/sections/selection_theory.tex']))
        if axis.get('interim_confirmation'):
            claims.append(dict(id='interim_frozen_original_test_edge',paper='Interim confirmation on the first independently trained GPT2 edge',result=axis['interim_confirmation']['scope']+' This completed edge retains all58 query/objective cells. It does not replace the unfinished full-suite confirmation or establish a general selection advantage.',evidence=axis['input_summary_paths']+['scripts/axis_paper.py','paper/sections/axis_results.tex','paper/sections/appendix_evidence.tex']))
        if axis.get('energy_controls'):
            claims.append(dict(id='development_direction_and_magnitude_exchange',paper='Output fitting mechanism in the main results and Appendix output geometry',result='Across all six exposed development queries, exchanging the saved geometric and fitted norms shows that magnitude explains the IIA improvement. At fitted norm, fitted orientation lowers IIA in three cells and ties in three while lowering source-output KL in all six. Same reader and fixed supports; no new fitting or official-test use. This is explanatory development evidence, not an independently confirmed norm-only method or semantic correspondence.',evidence=axis['input_summary_paths']+['scripts/run_compiled_energy_control.py','scripts/summarize_compiled_energy.py','paper/sections/axis_results.tex','paper/sections/selection_theory.tex']))
        if axis.get('selection_headroom'):
            claims.append(dict(id='observed_finite_menu_selection_headroom',paper='Candidate quality and finite-menu selection headroom',result='Post-outcome query-level maxima on the same frozen finite candidate menu separate observed candidate limitations from shortlist and refinement regret. These maxima use test outcomes: they are neither deployable policies nor population or per-example adaptive-policy upper bounds.',evidence=axis['input_summary_paths']+['scripts/summarize_selection_headroom.py','paper/sections/axis_results.tex','paper/sections/appendix_evidence.tex']))
    if (out/'reform_runs.json').exists():
        claims.append(dict(id='amplitude_attribution_and_native_coarsening',paper='Output-fitting attribution and native-group development',
            result='Six-cell independently trained amplitude control and24 native source-only anchors. Strong two-scale IIA matches full weight fitting on development; full fit improves KL. Joint native groups improve on the same-source singleton comparison, but rank-one compression retains similar behavior. All conditions remain development.',
            evidence=[v['path'] for v in reform['inputs']]+['scripts/reform_paper.py','src/ccad/native_coarsening.py','paper/sections/native_group_development.tex']))
    for claim in claims:
        verified=[]
        for rel in claim['evidence']:
            p=root/rel
            if not p.is_file():raise FileNotFoundError(f"Evidence entry {claim['id']}: {p}")
            verified.append(dict(path=rel,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        claim['evidence']=verified
    current_ids=['operation_and_native_equivalence','contrast_and_pair_common_completion','learned_superposition_fidelity_and_truth','frozen_named_operation_family_and_native_writes','source_axis_native_transfer_and_selection','complete_frozen_original_task_confirmation','compiled_fitting_objective_tradeoff','development_direction_and_magnitude_exchange','observed_finite_menu_selection_headroom']
    if (out/'reform_runs.json').exists():current_ids.append('amplitude_attribution_and_native_coarsening')
    (out/'EVIDENCE_INDEX.json').write_text(json.dumps(dict(current_manuscript_claim_ids=[c['id'] for c in claims if c['id'] in current_ids],claims=claims,data_manifest='data/DATA_MANIFEST.json',figure_manifest='figures/FIGURE_MANIFEST.json',
        raw_hashes=manifest['original_raw_sha256'],scope='Current manuscript evidence locator. Hashes establish file identity, not scientific validity; source and member KL references differ.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(context_rows=len(context['rows']),source_seed_rows=len(seed_rows),member_directions=20,fixed_cases=16,tables=len(list(table_dir.glob('*.tex'))),output=str(out))))

if __name__=='__main__':
    main()
