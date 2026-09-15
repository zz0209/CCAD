"""Collect the frozen response-reuse experiment, preserving its fixed seed cohort."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from analyze_binding_member_selection import analyze

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    runs=[ROOT/f'runs/REFORM_R57_binding_member_confirm_t{t}_v1_20260915' for t in (2,3,4,5)]
    panels=[];identities=[];cost=[];records=[]
    original=np.load(ROOT/'runs/REFORM_R57_binding_member_selection_v1_20260915/response_bank.npz')
    for run in runs:
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        panels.append(json.loads((run/'panel.json').read_text()))
        records.append([json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()])
        bank=np.load(run/'response_bank.npz')
        for k in original.files:
            if not k.startswith('target_path'):assert np.array_equal(original[k],bank[k]),(run,k)
        calls=json.loads((run/'selection_cost.json').read_text())['calls']
        assert calls['source']==calls['raw']==0 and calls['target']==256
        cost.append(dict(run=run.name,calls=calls,progress=json.loads((run/'progress.json').read_text())))
        identities.append(dict(run=run.relative_to(ROOT).as_posix(),files={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ['config.resolved.json','panel.json','code_hashes.json','metrics.raw.jsonl','response_bank.npz','status.json']}))
    assert all(p==panels[0] for p in panels)
    def identity(w):return (tuple(w['names']),tuple(tuple(p) for p in w['places']))
    fresh={identity(w) for w in panels[0]['worlds'] if w['split']=='development'}
    oldpaths=['REFORM_R57_binding_member_selection_v1_20260915','REFORM_R56_binding_roles_confirm_s1_v1_20260915']
    prior={identity(w) for p in oldpaths for w in json.loads((ROOT/'runs'/p/'panel.json').read_text())['worlds']}
    assert not fresh&prior
    source_rows=[{(r['component'],r['template'],r['order'],r['query'],r['method'],r['members'],r['operation']):r['answer_id'] for r in rr if r['kind']=='member_selection' and r['method'].startswith('source_')} for rr in records]
    # Only the source interventions themselves must be target-independent.
    source_controls=[{k:v for k,v in d.items() if k[4] in ('source_full','source_selected')} for d in source_rows]
    assert all(d==source_controls[0] for d in source_controls)
    groups={label:analyze(runs,forms) for label,forms in [('all_forms',None),('fit_forms',[0,1]),('heldout_forms',[2,3])]}
    per_target={run.name:analyze(run) for run in runs}
    case=None;eligible=0
    for target,rr in zip((2,3,4,5),records):
        lookup={(r['component'],r['template'],r['order'],r['query'],r['method'],r['members'],r['operation']):r for r in rr if r['kind']=='member_selection'}
        for row in panels[0]['rows']:
            if row['split']!='development' or row['query']!=0:continue
            key=(row['component'],row['template'],row['order'])
            get=lambda method,k,op,q:lookup[(*key,q,method,k,op)]
            if all(get('source_path',16,op,q)['correct'] for op in ('entity','both') for q in (0,1)) and not all(get('geometry',16,op,q)['correct'] for op in ('entity','both') for q in (0,1)):
                eligible+=1
                if case is None:case=dict(target_seed=target,row=row,outputs={m:{op:[get(m,16,op,q)['answer_id'] for q in (0,1)] for op in ('entity','both')} for m in ('geometry','source_cached','source_path','raw_path','target_path')})
    if case:
        from tokenizers import Tokenizer
        cfg=json.loads((runs[0]/'config.resolved.json').read_text())
        tok=Tokenizer.from_file(str(Path(cfg['model_local_dir'])/'tokenizer.json'))
        case['decoded_outputs']={m:{op:[tok.decode([i]).strip() for i in ids] for op,ids in ops.items()} for m,ops in case['outputs'].items()}
    result=dict(written_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=groups,per_target=per_target,runs=identities,cost=cost,
        checks=dict(all_runs_pass=True,fresh_worlds_disjoint=True,source_banks_bytewise_equal=True,source_controls_equal=True),
        example=case,eligible_examples=eligible,example_selection='First target/world/form/order where source_path16 preserves both rules and geometry16 fails; descriptive only.',
        scope='Fixed source1, four target seeds2–5. Complete requires both cities for entity exchange and E+A restoration. Attribute-only is not this endpoint. Two SAE locations inside a shared layer-stacked context program.')
    (ART/'r57_member_confirmation_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for group,analysis in groups.items():
        print(group)
        for key in ['target_full/k64','geometry/k16','source_cached/k16','source_path/k16','source_pooled/k16','raw_path/k16','target_path/k16']:
            print(key,analysis['methods'][key]['complete'])
        print('contrasts',{k:v['complete'] for k,v in analysis['contrasts'].items() if k.startswith('source_path/k16')})


if __name__=='__main__':main()
