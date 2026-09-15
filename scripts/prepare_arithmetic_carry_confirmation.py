"""Freeze metadata-only three-operand questions and existing member artifacts."""
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'


def select_canonical(seed=9501501, excluded=(), operand_stop=40):
    key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest()
    triples=[x for x in itertools.combinations(range(10,operand_stop),3) if sum(x)<100]
    meta={x:(sum(x),sum(a%10 for a in x)//10) for x in triples}
    used=set(excluded);canonical=[]
    # Allocate the rare carry0/carry2 matches before the abundant adjacent classes.
    strata=[('carry_0_2',0,2,16),('carry_1_2',1,2,16),('carry_0_1',0,1,16),
            ('preserve_c0',0,0,6),('preserve_c1',1,1,5),('preserve_c2',2,2,5)]
    for label,c1,c2,n in strata:
        candidates=[];pool=[]
        if c1!=c2:
            for total in sorted({meta[x][0] for x in triples}):
                pool.extend(itertools.product([x for x in triples if meta[x]==(total,c1)],
                                              [x for x in triples if meta[x]==(total,c2)]))
        else:
            for unit in range(10):
                pool.extend(itertools.combinations([x for x in triples if meta[x][1]==c1 and meta[x][0]%10==unit],2))
        for a,b in pool:
            sa,ca=meta[a];sb,cb=meta[b]
            if set(a)&set(b):continue
            eligible=(sa==sb and {ca,cb}=={c1,c2} if c1!=c2
                      else ca==cb==c1 and sa%10==sb%10 and sa!=sb)
            if eligible and all(10<=s+10*(d-c)<100 for s,c,d in [(sa,ca,cb),(sb,cb,ca)]):
                candidates.append((a,b))
        count=0
        for a,b in sorted(candidates,key=key):
            if a in used or b in used:continue
            used.update([a,b])
            canonical.append(dict(component=f'pair{len(canonical):03}',
                stratum=label if c1!=c2 else 'preserve',carry_group=label,
                condition='same_answer_opposite_carry' if c1!=c2 else 'same_carry_different_answer',
                recipient_question=a,donor_question=b))
            count+=1
            if count==n:break
        assert count==n,(label,count)
    return canonical,used-set(excluded),meta


def main():
    seed=9501501
    canonical,used,meta=select_canonical(seed)
    templates=['12+23=35\n34+12=46\n{a}+{b}+{c}=',
        'Q: What is 12 plus 23? A: 35\nQ: What is 34 plus 12? A: 46\nQ: What is {a} plus {b} plus {c}? A:']
    rows=[];lookup={}
    for t,form in enumerate(templates):
        for q in sorted(used):
            total,carry=meta[q];lookup[t,q]=len(rows)
            rows.append(dict(template=t,a=q[0],b=q[1],c=q[2],total=total,carry=carry,
                tens=total//10,unit=total%10,split='confirmation',
                prompt=form.format(a=q[0],b=q[1],c=q[2])))
    pairs=[]
    for p in canonical:
        for t in range(2):
            i,j=[lookup[t,tuple(p[k])] for k in ['recipient_question','donor_question']]
            for a,b in [(i,j),(j,i)]:
                r,d=rows[a],rows[b]
                pairs.append(dict(component=p['component'],condition=p['condition'],stratum=p['stratum'],
                    carry_group=p['carry_group'],template=t,recipient=a,donor=b,
                    expected=r['total']+10*(d['carry']-r['carry'])))
    assert len(rows)==len(pairs)==256
    prior=[]
    for path in sorted((ROOT/'runs').glob('REFORM_R*/panel.json')):
        old=json.loads(path.read_text())
        prompts={r['prompt'] for r in old.get('rows',[]) if 'prompt' in r}
        if prompts:
            assert not prompts&{r['prompt'] for r in rows}
            prior.append(path.relative_to(ROOT).as_posix())
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    panelpath=ART/'R50_FROZEN_TRIPLE_PANEL.json';freezepath=ART/'R50_FROZEN_RULE_INPUTS.json'
    configpath=ROOT/'configs/reform_r50_frozen_carry_three_operands_v1.json'
    assert not any(p.exists() for p in [panelpath,freezepath,configpath])
    panel=dict(written_at_utc=stamp,rows=rows,pairs=pairs,canonical_pairs=canonical,selection_seed=seed,
        prior_prompt_panels=prior,
        scope='Frozen new three-operand questions; source/target memberships learned only on two operands. All primary rows retained irrespective of baseline competence.',
        primary='48canonical carry pairs(16each0-1,1-2,0-2) and16same-carry pairs; reciprocal orientations andtwoforms; pair-cluster bootstrap stratified by carry-pair class.')
    panelpath.write_text(json.dumps(panel,indent=2)+'\n')
    source='runs/REFORM_R49_carry_participation_source_v1_20260915'
    target='runs/REFORM_R49_carry_source_relation_four_targets_v1_20260915'
    cfg=json.loads((ROOT/'configs/reform_r49_carry_source_relation_four_targets_v1.json').read_text())
    for k in ['source_cache_run','functional_rule_test','evaluation_operand_pairs']:cfg.pop(k,None)
    cfg.update(round_id='R50',run_id='REFORM_R50_frozen_carry_three_operands_v1_20260915',run_parent='R50',
        written_at_utc=stamp,purpose='Predict three-operand carry swaps using frozen source and target memberships learned on two-operand additions.',
        scope=panel['scope'],templates=templates,max_length=96,budget_seconds=600,
        budget='At most600driver seconds using existing model andfiveSAEs;256new baseline prompts andfrozen interventions,zero fittingupdates.',
        dataset_revision='CCAD R50 frozen three-operand metadata panel9501501',statistics_unit=panel['primary'],
        evidence_level='controlled_frozen_confirmation',tens_sites=[0,1])
    memberships={'1':dict(run=source,methods=['noop','whole_state','raw_carry_direction',
        'conditional_carry_64','field_fitted_binary_64','field_fitted_weighted_64'])}
    for s in range(2,6):
        memberships[str(s)]=dict(run=target,methods=['noop','raw_carry_direction','assignment_64',
            'field_fitted_binary_64','transferred_fitted_binary_64','transferred_fitted_weighted_64'])
    cfg['frozen_rule_evaluation']=dict(panel=panelpath.relative_to(ROOT).as_posix(),
        freeze=freezepath.relative_to(ROOT).as_posix(),memberships=memberships)
    configpath.write_text(json.dumps(cfg,indent=2)+'\n')
    paths=[panelpath,configpath,Path(__file__),ROOT/'scripts/arithmetic_frozen_rules.py',
           ROOT/'scripts/run_arithmetic_digit_components.py']
    for run in [source,target]:
        paths += [ROOT/run/x for x in ['config.resolved.json','status.json','code_hashes.json']]
    for s in range(1,6):paths.append(ROOT/(source if s==1 else target)/f'rule_members_seed{s}.npz')
    freeze=dict(written_at_utc=stamp,inputs=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        selection='Allmembers andrawdirection frozen fromR49 before anythree-operand baseline orintervention outcome. Sourcebinary isprimary; weighted isprespecifiedcontrol. No newtuning.',
        primary_contrasts=['sourcebinary carry-rule/preservation','targetbinary carry-rule/preservation',
                          'targetbinary-assignment','targetbinary-direct','targetbinary-raw'],statistics=panel['primary'])
    freezepath.write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(dict(written_at_utc=stamp,rows=len(rows),pairs=len(pairs),prior_panels=len(prior))))


if __name__=='__main__':main()
