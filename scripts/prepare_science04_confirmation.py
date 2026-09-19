"""Freeze fresh contexts, requests and matched recipes without model outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
BULK=Path('D:/CCAD_Storage/runs/science_upgrade_20260919')

def save(path,value):
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8'))==value,path
    else: path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    base=json.loads((ROOT/'configs/science03_shift_execution_reform_v1.json').read_text())
    source=json.loads(Path(base['source_manifest']).read_text())
    previous=json.loads((ROOT/'artifacts/independent_reuse_20260916/IR04_NEW_TASKS_PANEL.json').read_text())
    original=json.loads((ROOT/'configs/reform_r58_shift_source_development_v2.json').read_text())
    old_source=json.loads((Path(base['frozen_source_run'])/'panel.json').read_text())
    excluded={r['document_sha256'] for p in [previous,old_source] for r in p['rows']}
    table=pq.read_table(original['dev_data'],filters=[('profession','in',[5,25,12,24])],columns=['hard_text','profession','gender'])
    buckets={(p,g):[] for p in [5,25,12,24] for g in [0,1]}
    for row in table.to_pylist():
        h=hashlib.sha256(row['hard_text'].encode()).hexdigest()
        if h in excluded:continue
        excluded.add(h)
        buckets[row['profession'],row['gender']].append(dict(text=row['hard_text'],profession=row['profession'],gender=row['gender'],document_sha256=h,split='dev'))
    rows=[]
    for key,values in buckets.items():
        assert len(values)>=32,(key,len(values))
        rows+=sorted(values,key=lambda x:hashlib.sha256(('science04/'+x['document_sha256']).encode()).hexdigest())[:32]
    tok=AutoTokenizer.from_pretrained(base['model_local_dir'],local_files_only=True)
    tokens=tok([r['text'] for r in rows],truncation=True,max_length=2048,add_special_tokens=True)['input_ids']
    for i,(r,t) in enumerate(zip(rows,tokens)):r.update(row_id=i,tokens=t)
    hp=OUT/'ROUND04_HUMAN_PANEL.json'
    save(hp,dict(rows=rows,original_split='dev',evidence='Unused contexts selected before frozen method execution; later readouts fixed from IR04.',counts_before_selection={str(k):len(v) for k,v in buckets.items()},exclusions=['IR04_NEW_TASKS_PANEL.json train and test','Original human explanation panel all rows'],dataset_revision=base['dataset_revision']))
    rng=np.random.default_rng(2026091904)
    groups=['pronouns','names','associated_words']
    dose={};family={}
    endpoints=['pronouns','names','pronouns+names','associated_words','pronouns+associated_words','names+associated_words','full']
    for q in endpoints:family[q]='endpoints'
    for kind in ['interior','boundary']:
        for i in range(12):
            v=rng.uniform(.05,.95,3)
            if kind=='boundary':v[i%3]=(i//3)%2
            q=f'{kind}_{i:02d}';dose[q]=dict(zip(groups,v.tolist()));family[q]=kind
    members={}
    for i in range(8):
        q=f'member_subset_{i:02d}'
        members[q]={s:rng.integers(0,2,len(ids)).tolist() for s,ids in source['members'].items()}
        family[q]='member_subsets'
    queries=endpoints+list(dose)+list(members)
    rp=OUT/'ROUND04_REQUESTS.json'
    rh=save(rp,dict(queries=queries,dose_queries=dose,member_queries=members,families=family,seed=2026091904))
    iprev=json.loads((OUT/'ROUND03_INFINITIVE_PANEL.json').read_text())
    fit=json.loads((ROOT/'artifacts/independent_reuse_20260916/IR01_fit_panel.json').read_text())
    old=json.loads((ROOT/'artifacts/independent_reuse_20260916/IR01_evaluation_panel.json').read_text())
    verbs=['permitted','obliged','requested','authorized','compelled','coached']
    nouns=['technicians','volunteers','delegates','pilots']
    oldverbs={r['pair'].split(':')[0] for p in [fit,old,iprev] for r in p['rows']}
    oldnouns={r['pair'].split(':')[1] for p in [fit,old,iprev] for r in p['rows']}
    assert not set(verbs)&oldverbs,(verbs,oldverbs)
    assert not set(nouns)&oldnouns,(nouns,oldnouns)
    ir=[]
    for v in verbs:
        for n in nouns:
            forms=[f'The {n} had been {v}',f'The committee {v} the {n}',f'Yesterday the {n} were {v}',f'At the meeting, the supervisor {v} the {n}']
            for f,t in enumerate(forms):ir.append(dict(pair=f'{v}:{n}',verb=v,noun=n,form=f,role='predicate' if f%2==0 else 'object',text=t))
    iq={'predicate':[1,1,1,0],'object':[0,0,0,1],'full':[1,1,1,1]};iff={k:'endpoints' for k in iq}
    for kind in ['interior','boundary']:
        for i in range(12):
            v=rng.uniform(.05,.95,2)
            if kind=='boundary':v[i%2]=(i//2)%2
            q=f'{kind}_{i:02d}';iq[q]=[float(v[0])]*3+[float(v[1])];iff[q]=kind
    for bits in range(1,16):
        v=[(bits>>i)&1 for i in range(4)]
        if v in [iq[k] for k in ['predicate','object','full']]:continue
        q=f'member_subset_{bits:02d}';iq[q]=v;iff[q]='member_subsets'
    ip=OUT/'ROUND04_INFINITIVE_PANEL.json'
    ih=save(ip,dict(rows=ir,queries=iq,families=iff,answer=' to',scope='Six new verbs, four new nouns, four forms, 96 contexts; fixed source four members; fresh request coordinates and all 12 nonsemantic binary subsets.'))
    config_paths=[]
    for seed in [2,3,4,5]:
        c=base.copy();rid=f'SCIENCE04_shift_t{seed}_v1_20260919'
        c.update(run_id=rid,run_parent='SCIENCE_UPGRADE_04',target_seed=seed,seeds=[seed],audit_opened=True,evidence_level='frozen_new_context_request_confirmation',purpose='Confirm trained input-dependent programs on fresh biographies, requests and later readouts.',target_directory='D:/CCAD_Storage/training_curves/REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915/step_8192',relation_run=f'runs/REFORM_R59_shift_confirm_seed{seed}_v1_20260915',bulk_output_dir=f'D:/CCAD_Storage/training_curves/{rid}',queries=queries,dose_queries=dose,member_queries=members,request_panel=str(rp),request_panel_sha256=rh,evaluation_panel=str(hp),evaluation_split='dev',evaluation_evidence='frozen_fresh_contexts',evaluation_description='Fresh original-dev documents excluded from prior source and later-consumer panels',variants=['head_mixed','tangent_gain','tangent_mixed'],evaluate_baselines=['input_initial','raw_reconstruction'],eval_token_budget=4096,budget_seconds=1200,budget='Three matched512-update arms and39requests/256fresh biographies; <=1200driver seconds per target, four targets; prior two-arm615seconds.',scope='Four additional controlled targets for the frozen new method. Original source/use training unchanged, new original-dev texts and requests, same four previously frozen later readouts. Eight finer member masks never trained. No selection on confirmation.',statistics_unit='Target initialization, paired profession/gender documents and request family; four fixed later readouts and fixed source explanation.')
        path=ROOT/f'configs/science04_shift_t{seed}_v1.json';save(path,c);config_paths.append(str(path))
    ibase=json.loads((ROOT/'configs/science03_infinitive_execution_reform_v1.json').read_text())
    prep=json.loads((ROOT/'configs/ir01_infinitive_natural_dev_v1.json').read_text())
    rid='SCIENCE04_infinitive_seed1_material_v1_20260919'
    prep.update(run_id=rid,run_parent='SCIENCE_UPGRADE_04',purpose='Fit existing fixed comparator for target1 using the unchanged natural-state recipe; exposed development panel only.',target_seed=1,seeds=[1],target_directory=base['target_directory'],methods=['none','source','native'],run_storage_root=str(BULK))
    pp=ROOT/'configs/science04_infinitive_seed1_material_v1.json';save(pp,prep)
    for seed in [1,3,4,5]:
        c=ibase.copy();rid2=f'SCIENCE04_infinitive_t{seed}_v1_20260919'
        c.update(run_id=rid2,run_parent='SCIENCE_UPGRADE_04',target_seed=seed,seeds=[seed],target_directory=base['target_directory'] if seed==1 else ibase['target_directory'],relation_run=str(BULK/rid) if seed==1 else f'runs/IR01_infinitive_confirm_t{seed}_v1_20260916',panel=str(ip),request_panel_sha256=ih,variants=['mixed','tangent_gain','tangent_mixed'],baselines=['native','native_tangent_relation_8','raw_reconstruction'],gain_before_selection=True,audit_opened=True,evidence_level='frozen_new_context_request_confirmation',purpose='Confirm input-dependent program training on independent infinitive contexts and target seeds.',budget_seconds=300,budget='Three512-update arms and39requests/96fresh contexts,<=300driver seconds per target.',scope='Fixed public four-member source; six new verbs/four new nouns/four forms; four additional targets1,3,4,5; calibration before shared selection; no confirmation selection.',dataset_revision='science04-new-infinitive-contexts-v1',statistics_unit='Target seed, crossed verb/noun and paired request family; fixed public source.')
        path=ROOT/f'configs/science04_infinitive_t{seed}_v1.json';save(path,c);config_paths.append(str(path))
    freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),primary='Mean per-query response nRMSE on24 fresh interior/boundary coordinates. Compare tangent_mixed with same-budget fixed mixed training and gain. Report both explanations separately.',secondary='Deletion endpoints, unfitted fine member masks, source role effects, later response agreement and classification utility, natural reconstruction.',statistics='2000 paired target-seed/document/request draws for human; target-seed crossed verb/noun and query draws for infinitive. Shared texts/requests paired across seeds. Fixed source explanations and four later heads.',recipe='512updates, lr dictionary1e-4/gain or fixed coefficients.005,0.9source response+0.1state, naturalMSEweight1, no later labels/heads in fit. Half semantic endpoints, half uniform continuous requests. Same members-per-source allowance. Source gain before support scoring in both settings.',evidence='All model comparisons frozen before new outputs. Existing target dictionaries have old-method history; new contexts and requests plus recipe-frozen seed replication.',budget='4800driver seconds initial for confirmation; seed1 fixed material preparation separately<=600, measured throughput may justify documented extension.',configurations=config_paths,material_configuration=str(pp),hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [hp,rp,ip]+[Path(f) for f in config_paths]+[pp]},code_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['scripts/train_shift_response_space.py','scripts/train_infinitive_program.py','scripts/prepare_science04_confirmation.py']})
    save(OUT/'ROUND04_FREEZE.json',freeze)
    print(json.dumps(dict(human_contexts=len(rows),infinitive_contexts=len(ir),human_queries=len(queries),infinitive_queries=len(iq),targets=8,freeze=freeze['written_at_utc'])))

if __name__=='__main__':main()
