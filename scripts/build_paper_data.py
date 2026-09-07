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
        p=root/rel;b=p.read_bytes();sources.append(dict(path=rel.as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
        return json.loads(b)
    context=read(BASE/'r4_l15/R4_L15_CONTEXT_SUMMARY.json')
    mechanism=read(BASE/'r5_mechanism/R5_MECHANISM_SUMMARY.json')
    learning=read(BASE/'r4_l15/FUNCTIONAL_LEARNING_SUMMARY.json')
    training=read(BASE/'r4_l15/TRAINING_SUMMARY.json')
    example=read(BASE/'r5_mechanism/figure_member_example_manifest.json')
    panel=read(BASE/'r4_l15/PANEL_PREDECLARATION.json')
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
    members=[x for x in example['displayed_values'] if 'member' in x]
    lines=[]
    for x in members:
        lines.append(f"{x['member']} & {100*x['natural_active_fraction']:.2f} & {x['temporal_code_change']:.2f} & {x['quoted_code_change']:.2f} & {x['temporal_removal_time_shift_loss']:.3f} & {x['quoted_removal_time_shift_loss']:.3f}"+r' \\')
    (table_dir/'display_members.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
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
    plot=dict(context_rows=context['rows'],geometry=mechanism['geometry_ratios'],mechanism_rows=mechanism['rows'],
              memberships=[x for x in mechanism['memberships'] if x['source_seed']==1 and x['target_seed']==2],
              fixed_examples=fixed,display_members=[x for x in example['displayed_values'] if 'member' in x],
              functional_learning=learning['rows'],training_quality=quality)
    (data_dir/'figure_data.json').write_text(json.dumps(plot,indent=2)+'\n',encoding='utf-8')
    for rel in [BASE/'r5_mechanism/fixed_cases.csv',BASE/'r5_mechanism/member_effects.csv',BASE/'r5_mechanism/mechanism_directions.csv']:
        p=root/rel;b=p.read_bytes();(data_dir/p.name).write_bytes(b)
        sources.append(dict(path=rel.as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    outputs=[p for folder in [data_dir,table_dir] for p in folder.iterdir() if p.is_file() and p.name!='DATA_MANIFEST.json']
    manifest=dict(sources=sources,output_files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(outputs)],
       scope='Deterministic extraction of retained summaries; all observed strata and methods retained. No fitting, outcome-based selection, inference, or independent confirmation. Member KL reference is intact FCC; context KL reference is source operation.',
       original_raw_sha256=dict(context=context['source_raw_sha256'],mechanism=mechanism['raw_sha256'],training=training['raw_sha256'],learning=learning['raw_sha256']))
    (data_dir/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    claims=[
        dict(id='operation_and_native_equivalence',paper='Sections 2, A.1-A.3',result='Explicit signed predictor and finite-panel equivalence condition; no semantic uniqueness claim',
             evidence=['paper/sections/appendix_theory.tex','artifacts/seven_round_rebuild_20260906/R5_METHODS_AND_PROOFS.md','src/ccad/factor_correspondence.py']),
        dict(id='rrr_and_composition',paper='Sections 2.3, A.4-A.6',result='Exact penalized rank fit and position-aware composition; standard algebra',
             evidence=['artifacts/seven_round_rebuild_20260906/R3_METHODS_AND_ALGEBRA.md','artifacts/seven_round_rebuild_20260906/R4_METHODS_AND_ALGEBRA.md','src/ccad/factor_correspondence.py']),
        dict(id='controlled_material',paper='Section 3.1; Figure 4; Tables 2-3',result='Five controlled late-hook SAEs and fixed-checkpoint quality/function; no convergence certificate',
             evidence=['artifacts/seven_round_rebuild_20260906/r4_l15/TRAINING_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r4_l15/FUNCTIONAL_LEARNING_SUMMARY.json','configs/seven_r4_l15_train_k64_five_v1.json','configs/seven_r4_l15_checkpoint_function_v1.json']),
        dict(id='role_confirmation',paper='Section 4; Figures 1-2; Tables 1,4-7',result='New-role source operation reuse; raw/full more accurate; source mistakes retained',
             evidence=['artifacts/seven_round_rebuild_20260906/r4_l15/PANEL_PREDECLARATION.json','artifacts/seven_round_rebuild_20260906/r4_l15/frozen/FREEZE.json','artifacts/seven_round_rebuild_20260906/r4_l15/R4_L15_CONTEXT_SUMMARY.json','configs/seven_r4_l15_confirmation_v1.json','scripts/run_frozen_composition.py']),
        dict(id='early_positive_and_counterexample',paper='Section 4.3; Appendix C.4; Table 8',result='Early development positive, weaker frozen confirmation, and prefix-rank limitation all preserved',
             evidence=['artifacts/seven_round_rebuild_20260906/R2_RESULT_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r3_composition/R3_RESULT_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r4_frozen/R4_FROZEN_SUMMARY.json','artifacts/seven_round_rebuild_20260906/r4_frozen/R4_PREFIX_DIAGNOSTIC.json']),
        dict(id='member_mechanism',paper='Section 5; Figure 3; Tables 9-10',result='Individual attenuation, norm-matched swaps and complete member removal; follow-up on exposed panel',
             evidence=['artifacts/seven_round_rebuild_20260906/r5_mechanism/R5_MECHANISM_SUMMARY.json','configs/seven_r5_member_mechanism_v1.json','scripts/run_member_mechanism.py','scripts/summarize_member_mechanism.py']),
        dict(id='fixed_examples_and_signed_relations',paper='Figures 1,5; Tables 10-12',result='All 16 fixed contexts/48 operations, complete signed matrices and heterogeneous natural examples',
             evidence=['artifacts/seven_round_rebuild_20260906/r5_mechanism/fixed_cases.csv','artifacts/seven_round_rebuild_20260906/r5_mechanism/figure_member_example_manifest.json','configs/seven_r5_member_natural_contexts_v1.json','scripts/run_member_natural_contexts.py']),
        dict(id='swapped_native_control_correction',paper='Appendix B.4; Tables 4-6',result='Historical wrong_factor swaps native group operations; at the same position its joint edit equals direct native. It is not an independent joint-specificity test.',
             evidence=['scripts/run_frozen_composition.py','scripts/run_composition_correspondence.py','paper/sections/appendix_methods.tex']),
    ]
    for claim in claims:
        verified=[]
        for rel in claim['evidence']:
            p=root/rel
            if not p.is_file():raise FileNotFoundError(f"Evidence entry {claim['id']}: {p}")
            verified.append(dict(path=rel,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        claim['evidence']=verified
    (out/'EVIDENCE_INDEX.json').write_text(json.dumps(dict(claims=claims,data_manifest='data/DATA_MANIFEST.json',figure_manifest='figures/FIGURE_MANIFEST.json',
        raw_hashes=manifest['original_raw_sha256'],scope='Current manuscript evidence locator. Hashes establish file identity, not scientific validity; source and member KL references differ.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(context_rows=len(context['rows']),source_seed_rows=len(seed_rows),member_directions=20,fixed_cases=16,tables=len(list(table_dir.glob('*.tex'))),output=str(out))))

if __name__=='__main__':
    main()
