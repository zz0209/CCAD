"""Freeze state-writer comparisons on questions excluded from all prior panels."""
import hashlib
import json
from datetime import datetime,timezone
from prepare_arithmetic_carry_confirmation import select_canonical,ROOT,ART


def main():
    excluded=set();prior=[]
    for p in sorted((ROOT/'runs').glob('REFORM_R*/panel.json')):
        rows=json.loads(p.read_text()).get('rows',[])
        excluded.update(tuple(r[k] for k in ['a','b','c']) for r in rows if 'c' in r)
        prior.append(p.relative_to(ROOT).as_posix())
    used=set(excluded);canonical=[];questions=set();meta={}
    for seed in [9541501,9541502]:
        local,new,details=select_canonical(seed,used,operand_stop=80,allow_repetition=True)
        for item in local:item['component']=f"pair{len(canonical):03}";canonical.append(item)
        used.update(new);questions.update(new);meta.update(details)
    source='runs/REFORM_R54_state_write_source_v1_20260915'
    assert json.loads((ROOT/source/'status.json').read_text())['status']=='PASS'
    cfg=json.loads((ROOT/source/'config.resolved.json').read_text());rows=[];lookup={}
    for t,template in enumerate(cfg['templates']):
        for q in sorted(questions):
            total,carry=meta[q];lookup[t,q]=len(rows)
            rows.append(dict(a=q[0],b=q[1],c=q[2],template=t,total=total,tens=total//10,unit=total%10,
                             carry=carry,split='confirmation',prompt=template.format(a=q[0],b=q[1],c=q[2])))
    pairs=[]
    for item in canonical:
        for t in range(2):
            i,j=[lookup[t,tuple(item[k])] for k in ['recipient_question','donor_question']]
            for a,b in [(i,j),(j,i)]:
                pairs.append(dict(component=item['component'],condition=item['condition'],stratum=item['stratum'],
                    carry_group=item['carry_group'],template=t,recipient=a,donor=b,
                    expected=rows[a]['total']+10*(rows[b]['carry']-rows[a]['carry'])))
    assert len(rows)==len(pairs)==512 and not questions&excluded
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    panel=ART/'R54_STATE_CONFIRMATION_PANEL.json';freeze=ART/'R54_STATE_CONFIRMATION_INPUTS.json'
    config=ROOT/'configs/reform_r54_state_write_confirmation_v1.json'
    assert not any(p.exists() for p in [panel,freeze,config])
    scope='Frozen source1 state-writer comparison on512 new prompts/requests;256 different triples excluded from every prior run panel. Two-digit operands including repetition, totalbelow100 (enumeration10..79); new questions and expanded operand/multiplicity range, not unseen language phenomenon. Distinct-operand metadata-only greedy allocations failed at10..49(1/16) and10..79(4/16) in a carry0-2 stratum, before any new outputs. Repeated operands admitted before freezing; source parameters/comparators unchanged.'
    panel.write_text(json.dumps(dict(rows=rows,pairs=pairs,canonical_pairs=canonical,written_at_utc=stamp,
        excluded_triples=len(excluded),prior_panels=prior,selection_seeds=[9541501,9541502],scope=scope),indent=2)+'\n')
    for key in ['source_function_refit','counterfactual_fit','evaluation_split']:cfg.pop(key,None)
    cfg.update(run_id='REFORM_R54_state_write_confirmation_v1_20260915',scope=scope,state_write_evaluation=True,
        purpose='Confirm whether state-dependent write direction preserves its benefit over matched constant/scalar controls.',
        budget_seconds=600,budget='600 driver seconds;512 fresh requests,nine frozen source controls,no fitting.',
        evidence_level='frozen_new_question_confirmation',written_at_utc=stamp,operand_range=[10,80])
    cfg['frozen_rule_evaluation']=dict(panel=panel.relative_to(ROOT).as_posix(),freeze=freeze.relative_to(ROOT).as_posix(),
        memberships={'1':dict(run=source,methods=['noop','readwrite_code_64','code_read_raw_write']+
                              [f'state_{m}_{s}' for s in ['code','raw'] for m in ['constant','scalar','direction']])})
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    files=[panel,config,ROOT/'scripts/prepare_arithmetic_carry_confirmation.py',__file__,ROOT/'scripts/arithmetic_state_write.py',
           ROOT/'scripts/arithmetic_frozen_rules.py',ROOT/'scripts/run_arithmetic_digit_components.py']
    files += [ROOT/source/p for p in ['status.json','config.resolved.json','rule_members_seed1.npz']]
    from pathlib import Path
    freeze.write_text(json.dumps(dict(written_at_utc=stamp,inputs=[dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in files],
        primary='state_direction_code minus state_constant_code and state_scalar_code on complete carry change and same-carry preservation; all control results retained, no new fit/selection or outcome exclusion.',
        statistics='96 changing canonical pairs:32 each0-1,1-2,0-2;32preservation pairs. Pair bootstrap retains two forms and reciprocal directions, within carry stratum.',scope=scope),indent=2)+'\n')
    print(json.dumps(dict(written_at_utc=stamp,excluded_triples=len(excluded),new_triples=len(questions),requests=len(pairs))))


if __name__=='__main__':main()
