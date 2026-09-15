"""Specify equal-question-budget source coverage and disjoint carry tests."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_arithmetic_carry_confirmation import select_canonical

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    old=json.loads((ART/'R50_FROZEN_TRIPLE_PANEL.json').read_text())
    excluded={tuple(r[k] for k in ['a','b','c']) for r in old['rows']}
    evaluation,eval_questions,meta=select_canonical(9511501,excluded)
    fit,fit_questions,_=select_canonical(9511502,excluded|eval_questions)
    assert not excluded & (eval_questions|fit_questions) and not eval_questions & fit_questions
    templates=old['rows'];forms=json.loads((ROOT/'configs/reform_r50_frozen_carry_three_operands_v1.json').read_text())['templates']
    rows=[]
    for q in sorted(fit_questions):
        total,carry=meta[q]
        rows.append(dict(template=0,a=q[0],b=q[1],c=q[2],total=total,carry=carry,tens=total//10,
            unit=total%10,split='fit',prompt=forms[0].format(a=q[0],b=q[1],c=q[2])))
    old_two=json.loads((ART/'R50_SOURCE_CE_TWO_OPERAND_PANEL_V2.json').read_text())
    offset=len(rows);rows.extend(old_two['rows']);pairs=[];canonical=[]
    for p in old_two['pairs']:
        pairs.append(dict(p,component='two_'+p['component'],recipient=offset+p['recipient'],donor=offset+p['donor']))
    for p in old_two['canonical_pairs']:canonical.append(dict(p,component='two_'+p['component'],arity=2))
    lookup={}
    for t,form in enumerate(forms):
        for q in sorted(eval_questions):
            total,carry=meta[q];lookup[t,q]=len(rows)
            rows.append(dict(template=t,a=q[0],b=q[1],c=q[2],total=total,carry=carry,tens=total//10,
                unit=total%10,split='confirmation',prompt=form.format(a=q[0],b=q[1],c=q[2])))
    for p in evaluation:
        p=dict(p,component='three_'+p['component'],arity=3);canonical.append(p)
        for t in range(2):
            i,j=[lookup[t,tuple(p[k])] for k in ['recipient_question','donor_question']]
            for a,b in [(i,j),(j,i)]:
                pairs.append(dict(component=p['component'],condition=p['condition'],stratum=p['stratum'],
                    carry_group=p['carry_group'],template=t,recipient=a,donor=b,
                    expected=rows[a]['total']+10*(rows[b]['carry']-rows[a]['carry'])))
    source_cache='runs/REFORM_R32_qwen_counterfactual_source_v1_20260914'
    source_rows=json.loads((ROOT/source_cache/'panel.json').read_text())['rows']
    fit_indices=[i for i,r in enumerate(source_rows) if r['split']=='fit']
    old_indices=sorted(fit_indices,key=lambda i:hashlib.sha256(f'9511503:{source_rows[i]["a"]}:{source_rows[i]["b"]}'.encode()).hexdigest())[:277]
    assert len(rows)==640 and len(pairs)==512 and len(fit_questions)==128 and len(old_indices)==277
    assert all(rows[p[k]]['split']!='fit' for p in pairs for k in ['recipient','donor'])
    prior=[];new_prompts={r['prompt'] for r in rows if 'c' in r}
    for p in sorted((ROOT/'runs').glob('REFORM_R*/panel.json')):
        observed={r['prompt'] for r in json.loads(p.read_text()).get('rows',[]) if 'prompt' in r}
        if observed:
            assert not new_prompts&observed,p
            prior.append(p.relative_to(ROOT).as_posix())
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    panelpath=ART/'R51_MIXED_CARRY_PANEL.json';freezepath=ART/'R51_MIXED_CARRY_INPUTS.json'
    configpath=ROOT/'configs/reform_r51_mixed_carry_source_v1.json'
    assert not any(p.exists() for p in [panelpath,freezepath,configpath])
    scope='Fixed source-coverage comparison:405source questions (277two+128three),256backward batches,64members. Newthree-operand confirmation is disjoint from fit andR50; oldtwo-operand evaluation is exposed development. No target fitting.'
    panel=dict(written_at_utc=stamp,rows=rows,pairs=pairs,canonical_pairs=canonical,scope=scope,
        prior_prompt_panels=prior,fit_canonical_pairs=fit,old_fit_indices=old_indices,
        primary='Newthree-operand complete rule success andsame-carry preservation; mixedCEbinary minus fixedtwo-operand CEbinary. Weighted gates andraw controls secondary. All predetermined rows retained.',
        selection_seeds=dict(evaluation=9511501,fit=9511502,original_subset=9511503))
    cfg=json.loads((ROOT/'configs/reform_r50_function_ce_carry_source_v1.json').read_text())
    cfg.update(round_id='R51',run_parent='R51',run_id='REFORM_R51_mixed_carry_source_v1_20260915',written_at_utc=stamp,
        scope=scope,purpose='Learn a functional source component across computation changes at the same question/update/member budgets.',
        evidence_level='frozen_source_coverage_comparison',dataset_revision='CCAD R51 mixedcarry metadata9511501/2/3',
        budget='600driver seconds,640baseline prompts (128fit,256exposedtwo,256newthree),source1only,256backward batches.',
        statistics_unit='Newthree48change and16preserve canonicalpairs; oldtwo32percondition; reciprocal orientations andtwoforms kept together. Report arities separately.')
    cfg.pop('evaluation_split',None)
    cfg['source_function_refit']['mixed_source']=dict(original_fit_indices=old_indices)
    cfg['counterfactual_fit']['balance_arity_carry']=True
    spec=cfg['frozen_rule_evaluation'];spec.update(panel=panelpath.relative_to(ROOT).as_posix(),freeze=freezepath.relative_to(ROOT).as_posix())
    parent='runs/REFORM_R50_function_ce_carry_source_v1_20260915'
    spec['memberships']={'1':dict(run=parent,methods=['noop','raw_carry_direction','raw_mixed_direction',
        'field_fitted_binary_64','field_fitted_weighted_64','function_ce_binary_64','function_ce_weighted_64',
        'mixed_ce_binary_64','mixed_ce_weighted_64'])}
    panelpath.write_text(json.dumps(panel,indent=2)+'\n');configpath.write_text(json.dumps(cfg,indent=2)+'\n')
    paths=[panelpath,configpath,Path(__file__)]+[ROOT/'scripts'/name for name in
        ['prepare_arithmetic_carry_confirmation.py','arithmetic_carry_source_fit.py','arithmetic_counterfactual_fit.py','arithmetic_frozen_rules.py','run_arithmetic_digit_components.py']]
    paths += [ROOT/'src/ccad/semantic_participation.py']
    paths += [ROOT/p/name for p in [source_cache,parent] for name in ['config.resolved.json','status.json','code_hashes.json']]
    paths += [ROOT/source_cache/name for name in ['panel.json','source_seed1.npz','states.npz']]
    paths += [ROOT/parent/'rule_members_seed1.npz']
    freezepath.write_text(json.dumps(dict(written_at_utc=stamp,inputs=[dict(path=p.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],primary=panel['primary'],
        selection='Metadata andfinal256-update methods specified before newthree outputs. Sourceonly; equal405question and256backward budgets; balancedarity/carry fitting strata.'),indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),pairs=len(pairs),fit_questions=405,new_triples=len(fit_questions|eval_questions),written_at_utc=stamp)))


if __name__=='__main__':main()
