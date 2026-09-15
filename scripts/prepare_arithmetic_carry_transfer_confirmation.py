"""Freeze a source-only-selected operation and its four-target confirmation."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from prepare_arithmetic_carry_confirmation import select_canonical,ROOT,ART


def main():
    excluded=set(); prior=[]
    for p in sorted((ROOT/'runs').glob('REFORM_R*/panel.json')):
        rows=json.loads(p.read_text()).get('rows',[])
        excluded.update(tuple(r[k] for k in ['a','b','c']) for r in rows if 'c' in r)
        prior.append(p.relative_to(ROOT).as_posix())
    canonical,questions,meta=select_canonical(9511505,excluded,operand_stop=50)
    source='runs/REFORM_R51_code_range_source_v1_20260915'
    cfg=json.loads((ROOT/source/'config.resolved.json').read_text())
    rows=[];lookup={}
    for t,template in enumerate(cfg['templates']):
        for q in sorted(questions):
            total,carry=meta[q];lookup[t,q]=len(rows)
            rows.append(dict(a=q[0],b=q[1],c=q[2],template=t,total=total,tens=total//10,
                unit=total%10,carry=carry,split='confirmation',prompt=template.format(a=q[0],b=q[1],c=q[2])))
    pairs=[]
    for p in canonical:
        for t in range(2):
            i,j=[lookup[t,tuple(p[k])] for k in ['recipient_question','donor_question']]
            for a,b in [(i,j),(j,i)]:
                pairs.append(dict(component=p['component'],condition=p['condition'],stratum=p['stratum'],
                    carry_group=p['carry_group'],template=t,recipient=a,donor=b,
                    expected=rows[a]['total']+10*(rows[b]['carry']-rows[a]['carry'])))
    assert len(rows)==len(pairs)==256 and not questions&excluded
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    panelpath=ART/'R51_TRANSFER_PANEL.json';freeze=ART/'R51_TRANSFER_INPUTS.json'
    configpath=ROOT/'configs/reform_r51_carry_transfer_confirmation_v1.json'
    assert not freeze.exists()
    if panelpath.exists():
        # Resume metadata preparation only; a completed freeze is immutable.
        previous=json.loads(panelpath.read_text())
        assert previous['rows']==rows and previous['pairs']==pairs
    scope='Fresh threeoperand questions withdistinct operands10..49 andtotalbelow100, disjoint from all prior triples andsource fitting. The original10..39pool has onlyfour unused0-2canonical matches; the wider operandrange was chosen before anynewoutputs. Mixedsource coverage includes the same arity/carry classes; this tests newquestions andoperandrange, not an unseen computation. Source operation selected using sourceonly development before fourtarget fits ornewoutputs.'
    panel=dict(rows=rows,pairs=pairs,canonical_pairs=canonical,written_at_utc=stamp,
        selection_seed=9511505,excluded_triples=len(excluded),prior_panels=prior,scope=scope)
    panelpath.write_text(json.dumps(panel,indent=2)+'\n')
    for k in ['source_function_refit','counterfactual_fit','evaluation_split']:cfg.pop(k,None)
    cfg.update(run_id='REFORM_R51_carry_transfer_confirmation_v1_20260915',scope=scope,
        purpose='Confirm the improved source rule and translate it into four independent SAE dictionaries.',
        evidence_level='frozen_new_question_confirmation',seeds=[1,2,3,4,5],written_at_utc=stamp,
        budget_seconds=900,budget='900driver seconds.256new prompts; source6methods plusfourtargets4methods. Two256-step field fits pertarget with256source-only pairs,512candidates,64union members,zero targetoutput gradients.')
    cfg['carry_relation_fit']=dict(source_run=source,source_method='code_scalar_64',seed=9511504,
        pairs=256,pool=512,steps=256,lr=.05)
    spec=cfg['frozen_rule_evaluation'];spec.update(panel=panelpath.relative_to(ROOT).as_posix(),freeze=freeze.relative_to(ROOT).as_posix())
    spec['memberships']={'1':dict(run=source,methods=['noop','raw_carry_direction','raw_mixed_direction','function_ce_binary_64','arity_scalar_64','code_scalar_64'])}
    for seed in range(2,6):spec['memberships'][str(seed)]=dict(run=source,methods=['raw_mixed_direction','assignment_64','direct_code_64','translated_code_64'])
    configpath.write_text(json.dumps(cfg,indent=2)+'\n')
    files=[panelpath,configpath,Path(__file__),ROOT/'scripts/prepare_arithmetic_carry_confirmation.py']
    files += [ROOT/'scripts'/name for name in ['arithmetic_carry_relation_fit.py','arithmetic_counterfactual_fit.py','arithmetic_frozen_rules.py','run_arithmetic_digit_components.py']]
    files += [ROOT/source/name for name in ['status.json','config.resolved.json','rule_members_seed1.npz','SOURCE_FIT_PANEL.json','states.npz','evaluation_seed1.npz']]
    old='runs/REFORM_R32_qwen_counterfactual_source_v1_20260914'
    files += [ROOT/old/name for name in ['status.json','config.resolved.json','panel.json','states.npz','source_seed1.npz']]
    freeze.write_text(json.dumps(dict(written_at_utc=stamp,inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
        primary='Source code_scalar minusarity_scalar complete rule success andpreservation; target translated_code minusassignment anddirect_code withsource-relative agreement also reported. Pair bootstrap stratifiedby carrygroup, allorientations/forms anddependent targetcohort together. No outcome-based exclusions orcheckpoint changes.',scope=scope),indent=2)+'\n')
    print(json.dumps(dict(written_at_utc=stamp,excluded_triples=len(excluded),rows=len(rows),pairs=len(pairs))))


if __name__=='__main__':main()
