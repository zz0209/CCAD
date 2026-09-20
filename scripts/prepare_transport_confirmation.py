"""Freeze the two label-free fixed-support probes before new evaluations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/transport_exploration_20260919'

def save(p,v):
    if p.exists():assert json.loads(p.read_text(encoding='utf8'))==v,p
    else:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    hc=json.loads((ROOT/'configs/transport01_refined_human_v1.json').read_text())
    original=json.loads((ROOT/'configs/reform_r58_shift_source_development_v2.json').read_text())
    panel=json.loads((Path(hc['frozen_source_run'])/'panel.json').read_text())
    excluded={r['document_sha256'] for r in panel['rows']}
    buckets={(y,g):[] for y in [0,1] for g in [0,1]}
    data=pq.read_table(original['dev_data'],filters=[('profession','in',[13,21])],columns=['hard_text','profession','gender'])
    for r in data.to_pylist():
        h=hashlib.sha256(r['hard_text'].encode()).hexdigest()
        if h in excluded:continue
        excluded.add(h);y=int(r['profession']==13)
        buckets[y,r['gender']].append(dict(text=r['hard_text'],label=y,gender=r['gender'],document_sha256=h,split='fresh',original_split='dev'))
    # The rare male-nurse dev cell was exhausted by the original source panel.
    # Its unused original-train records supply fresh texts before any new outputs.
    scarce={k for k,v in buckets.items() if len(v)<32}
    if scarce:
        more=pq.read_table(original['train_data'],filters=[('profession','in',[13,21])],columns=['hard_text','profession','gender'])
        for r in more.to_pylist():
            y=int(r['profession']==13);key=(y,r['gender'])
            if key not in scarce:continue
            h=hashlib.sha256(r['hard_text'].encode()).hexdigest()
            if h in excluded:continue
            excluded.add(h)
            buckets[key].append(dict(text=r['hard_text'],label=y,gender=r['gender'],document_sha256=h,split='fresh',original_split='train'))
    rows=[]
    for k,v in buckets.items():
        assert len(v)>=32
        rows+=sorted(v,key=lambda r:hashlib.sha256(('transport01/'+r['document_sha256']).encode()).hexdigest())[:32]
    tok=AutoTokenizer.from_pretrained(hc['model_local_dir'],local_files_only=True)
    for i,r in enumerate(rows):r.update(row_id=i,tokens=tok.encode(r['text'],truncation=True,max_length=2048))
    hp=OUT/'FRESH_HUMAN_PANEL.json';hh=save(hp,dict(rows=rows,source='96 unused original-dev and32 unused original-train documents; all original source-panel rows excluded. Rare male-nurse dev cell exhausted before this round.',scope='Three published resid4 members, original source head; no new annotations'))
    gc=json.loads((ROOT/'configs/transport01_initial_open_grammar_v1.json').read_text())
    oldpaths=['artifacts/independent_reuse_20260916/IR01_fit_panel.json','artifacts/independent_reuse_20260916/IR01_evaluation_panel.json','artifacts/science_upgrade_20260919/ROUND03_INFINITIVE_PANEL.json','artifacts/science_upgrade_20260919/ROUND04_INFINITIVE_PANEL.json']
    oldrows=[r for p in oldpaths for r in json.loads((ROOT/p).read_text())['rows']]
    oldv={r['pair'].split(':')[0] for r in oldrows};oldn={r['pair'].split(':')[1] for r in oldrows}
    verbs=['convinced','trained','commanded','pressured','motivated','assigned']
    nouns=['inspectors','dancers','librarians','gardeners']
    assert not set(verbs)&oldv,(verbs,oldv)
    assert not set(nouns)&oldn,(nouns,oldn)
    grows=[]
    for v in verbs:
        for n in nouns:
            for f,t in enumerate([f'The {n} had been {v}',f'The committee {v} the {n}',f'Yesterday the {n} were {v}',f'At the meeting, the supervisor {v} the {n}']):
                grows.append(dict(pair=f'{v}:{n}',verb=v,noun=n,form=f,role='predicate' if f%2==0 else 'object',text=t))
    queries={f'members_{b:04b}':[(b>>i)&1 for i in range(4)] for b in range(1,16)}
    gp=OUT/'FRESH_GRAMMAR_PANEL.json';gh=save(gp,dict(rows=grows,queries=queries,answer=' to',scope='Six unused verbs and four unused nouns; four existing sentence forms; all15 nonempty member subsets'))
    configs=[]
    for s in range(1,6):
        td='D:/CCAD_Storage/training_curves/'+('REFORM_R58_shift_dictionaries_curve_v1_20260915' if s==1 else 'REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915')+'/step_8192'
        h={**hc,'run_id':f'TRANSPORT01_frozen_human_t{s}_v1_20260920','target_seed':s,'seeds':[s],'target_directory':td,'evaluation_panel':str(hp),'evaluation_panel_sha256':hh,'audit_opened':True,'candidate_family_frozen':True,'evidence_level':'frozen_new_contexts','purpose':'Confirm label-free common-support refinement on new human documents','budget_seconds':240}
        g={**gc,'run_id':f'TRANSPORT01_frozen_grammar_t{s}_v1_20260920','target_seed':s,'seeds':[s],'target_directory':td,'panel':str(gp),'request_panel_sha256':gh,'audit_opened':True,'candidate_family_frozen':True,'evidence_level':'frozen_new_contexts','purpose':'Confirm label-free common-support refinement on new grammatical vocabulary and all member subsets','budget_seconds':240,'baselines':['native_tangent_relation_8','transport_open_initial','transport_refined_open','raw_reconstruction']}
        g['relation_run']=str(Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE04_infinitive_seed1_material_v1_20260919')) if s==1 else 'runs/IR01_infinitive_natural_dev_v1_20260916' if s==2 else f'runs/IR01_infinitive_confirm_t{s}_v1_20260916'
        for label,c in [('human',h),('grammar',g)]:
            p=ROOT/f'configs/transport01_frozen_{label}_t{s}_v1.json';save(p,c);configs.append(str(p))
    freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),primary='Mean per-query source-response nRMSE. Projected coefficient refinement versus identical untrained open support. Report both explanations separately.',secondary='Active-only initialization, reconstruction readout, hidden-field error on human probe, source effect and feasibility.',recipe='Initial encoder map; unchanged target dictionary; same common support of6 human or8 grammar members;64 projected gradient iterations using source fields only; no function-response training.',statistics='Five paired target seeds. Human stratified label/gender document bootstrap; grammar crossed verb/noun bootstrap. Exhaustive source-subset families fixed; sources fixed. Target dictionaries have prior history.',budget='Ten evaluation drivers expected below300seconds total; within standalone3600second allocation.',configurations=configs,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [hp,gp]+[Path(x) for x in configs]},code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/prepare_transport_confirmation.py','scripts/evaluate_transport_heldout.py','scripts/train_infinitive_program.py','src/ccad/intervention_transport.py']})
    save(OUT/'REFINEMENT_FREEZE.json',freeze)
    print(json.dumps(dict(human=len(rows),grammar=len(grows),targets=5,freeze=freeze['written_at_utc'])))

if __name__=='__main__':main()
