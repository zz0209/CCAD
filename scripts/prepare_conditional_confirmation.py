from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260920_round02'


def save(path, value):
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')


def main():
    base = json.loads((ROOT/'configs/conditional_profile_development_t2_r2_20260920.json').read_text())
    previous_path = ROOT/'artifacts/final_science_20260920/PROFILE_CONFIRMATION_HUMAN.json'
    previous = json.loads(previous_path.read_text())
    paths = [previous_path] + [Path(r['path']) for r in previous['exclusions']]
    excluded = set()
    provenance = []
    for path in dict.fromkeys(paths):
        data = json.loads(path.read_text())
        excluded.update(r['document_sha256'] for r in data['rows'])
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    original = json.loads((ROOT/'configs/reform_r58_shift_source_development_v2.json').read_text())
    table = pq.read_table(original['dev_data'], filters=[('profession','in',[5,25,12,24])], columns=['hard_text','profession','gender'])
    cells = {(p,g):[] for p in [5,25,12,24] for g in [0,1]}
    for row in table.to_pylist():
        digest = hashlib.sha256(row['hard_text'].encode()).hexdigest()
        if digest in excluded: continue
        excluded.add(digest)
        cells[row['profession'],row['gender']].append(dict(text=row['hard_text'],document_sha256=digest,
            profession=row['profession'],gender=row['gender'],split='confirm'))
    rows = []
    for cell, values in cells.items():
        if len(values)<16: raise ValueError(f'Insufficient unseen documents in {cell}')
        rows.extend(sorted(values,key=lambda r:hashlib.sha256(('conditional02/'+r['document_sha256']).encode()).hexdigest())[:16])
    tokenizer = AutoTokenizer.from_pretrained(base['model_local_dir'],local_files_only=True)
    for i,row in enumerate(rows): row.update(row_id=i,tokens=tokenizer.encode(row['text'],truncation=True,max_length=2048))
    panel = OUT/'CONDITIONAL_CONFIRMATION_HUMAN.json'
    save(panel,dict(rows=rows,original_split='dev',exclusions=provenance,counts_before_selection={str(k):len(v) for k,v in cells.items()}))
    rng = np.random.default_rng(2026092032)
    queries = {k:v for k,v in base['human_queries'].items() if not k.startswith('participation_')}
    families = {k:'center' if k=='center' else 'endpoints' for k in queries}
    groups = ['pronouns','names','associated_words']
    for j in range(8):
        name=f'participation_{j:02d}'
        weights=rng.uniform(.1,.9,3)
        if j>=4: weights[j%3]=j%2
        queries[name]=dict(zip(groups,weights.tolist()))
        families[name]='participation'
    source=json.loads(Path(base['source_manifest']).read_text())
    masks={}
    for j in range(4):
        name=f'member_subset_{j:02d}'
        masks[name]={s:rng.integers(0,2,len(ids)).tolist() for s,ids in source['members'].items()}
        families[name]='members'
    requests=OUT/'CONDITIONAL_CONFIRMATION_REQUESTS.json'
    save(requests,dict(human_queries=queries,human_member_queries=masks,human_families=families,random_seed=2026092032))
    configs=[]
    for seed in range(1,6):
        c={**base,'run_id':f'CONDITIONAL_CONFIRM_T{seed}_20260920','target_seed':seed,'seeds':[seed],
           'human_panel':str(panel),'human_queries':queries,'human_member_queries':masks,'request_panel':str(requests),
           'audit_opened':True,'candidate_family_frozen':True,'evidence_level':'frozen_new_document_and_request_confirmation',
           'purpose':'Test input-conditioned source sensitivity across five unchanged target dictionaries',
           'scope':'One fixed published55-member eleven-site source explanation;128 unused biographies; eight new participation requests; four fixed later heads excluded from profiling',
           'statistics_unit':'Paired target seeds, profession/gender-stratified documents and new participation requests; fixed source explanation and later heads',
           'budget_seconds':1000,'budget':'Five drivers of at most1000seconds; original2p support and128 solver steps, with shared frozen source profiles'}
        if seed==1:
            c['target_directory']='D:/CCAD_Storage/training_curves/REFORM_R58_shift_dictionaries_curve_v1_20260915/step_8192'
            c['task_adapted_reference']=None
        else:
            c['task_adapted_reference']['human']={s:f'D:/CCAD_Storage/training_curves/SCIENCE04_shift_t{seed}_v1_20260919/tangent_mixed/{s}_seed{seed}.pt' for s in ['embed','mlp_0','resid_0']}
            for path in c['task_adapted_reference']['human'].values():
                if not Path(path).is_file(): raise FileNotFoundError(path)
        path=ROOT/f'configs/conditional_confirm_t{seed}_20260920.json'
        save(path,c); configs.append(path)
    code=['scripts/train_intervention_changes.py','src/ccad/intervention_transport.py','scripts/prepare_conditional_confirmation.py',
          'scripts/analyze_conditional_confirmation.py','scripts/analyze_profile_confirmation.py']
    files=[panel,requests,Path(base['source_metric_cache']),Path(base['conditional_profile_cache'])]+configs+[ROOT/p for p in code]
    freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),round_id='FINAL_SCIENCE_02',
        primary='Mean nRMSE across eight new participation requests and four fixed later heads on their respective profession cohorts; conditional versus global source profile',
        secondary='Original head, seven semantic endpoints, center, four new member subsets; existing program training on matched seeds2to5',
        recipe='Unchanged source profile acquired from32 original fit biographies, four full-deletion fractions; embedding token-conditioned Gram with4-observation global prior and.05 identity shrinkage; other sites retain global Gram; same2p support and128 FISTAsteps',
        statistics='2000 paired target-seed, profession/gender-stratified document and request draws; shared document weights for heads on each cohort; semantic endpoints fixed; new participation and member requests resampled',
        evidence='All five targets have earlier project history; target2 development is exposed; these documents, participation coordinates and member subsets are unused before this freeze',
        configuration_paths=list(map(str,configs)),
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files])
    save(OUT/'CONDITIONAL_CONFIRMATION_FREEZE.json',freeze)
    print(json.dumps(dict(frozen_at=freeze['written_at_utc'],documents=len(rows),maximum_tokens=max(len(r['tokens']) for r in rows),configs=list(map(str,configs)))))


if __name__=='__main__': main()
