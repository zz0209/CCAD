from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from prepare_shared_column_panel import records


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round03'


def save(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    c=json.loads((ROOT/'configs/rg03_trajectory_development.json').read_text())
    prior={p.resolve() for p in (ROOT/'artifacts').rglob('*.json') if 'panel' in p.name.lower()}
    for path in (ROOT/'configs').glob('*.json'):
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value,dict):continue
        for key in ['evaluation_panel','human_panel','panel','fit_panel']:
            name=value.get(key)
            if isinstance(name,str) and Path(name).is_file():prior.add(Path(name).resolve())
        if isinstance(value.get('frozen_source_run'),str):
            name=Path(value['frozen_source_run'])/'panel.json'
            if name.is_file():prior.add(name.resolve())
    excluded=set()
    for path in sorted(prior):
        for row in records(json.loads(path.read_text(encoding='utf-8-sig'))):
            if 'document_sha256' in row:excluded.add(row['document_sha256'])
            if isinstance(row.get('text'),str):excluded.add(hashlib.sha256(row['text'].encode()).hexdigest())
    original=json.loads((ROOT/'configs/reform_r58_shift_source_development_v2.json').read_text())
    table=pq.read_table(original['dev_data'],filters=[('profession','in',[5,25,12,24])],columns=['hard_text','profession','gender'])
    cells={(p,g):[] for p in [5,25,12,24] for g in [0,1]};seen=set(excluded)
    for row in table.to_pylist():
        text=row['hard_text'];digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest);cells[row['profession'],row['gender']].append(dict(text=text,document_sha256=digest,
            profession=row['profession'],gender=row['gender'],split='dev'))
    rows=[]
    for cell,available in cells.items():
        if len(available)<8:raise ValueError(cell)
        rows+=sorted(available,key=lambda r:hashlib.sha256(('rg03trajectory/'+r['document_sha256']).encode()).hexdigest())[:8]
    tok=AutoTokenizer.from_pretrained(c['model_local_dir'],local_files_only=True)
    tokens=tok([r['text'] for r in rows],truncation=True,max_length=128,add_special_tokens=True)['input_ids']
    for i,(row,ids) in enumerate(zip(rows,tokens)):row.update(row_id=i,tokens=ids)
    panel=OUT/'TRAJECTORY_CONFIRMATION_PANEL.json'
    save(panel,dict(rows=rows,exclusion_files=list(map(str,sorted(prior))),excluded_documents=len(excluded),
        evidence='64new biographies selected by metadata and hash. Token limit128. One published program and four fixed later heads.'))
    queries={k:v for k,v in c['human_queries'].items() if k!='center'}
    rng=np.random.default_rng(2026092103)
    for kind in ['interior','boundary']:
        for i in range(6):
            weights=rng.uniform(.05,.95,3)
            if kind=='boundary':weights[i%3]=(i//3)%2
            queries[f'{kind}_{i:02d}']=dict(zip(['pronouns','names','associated_words'],weights.tolist()))
    paths=[]
    for seed in [3,4,5]:
        v=dict(c,run_id=f'RG03_TRAJECTORY_CONFIRMATION_T{seed}_20260921',target_seed=seed,seeds=[seed],
            datasets=['human'],human_panel=str(panel),human_split='dev',human_per_cell=8,human_queries=queries,
            human_member_queries={},query_progress=True,budget_seconds=2300,
            purpose='Confirm source-state feedback for complete and unseen partial requests in three further natural dictionaries.',
            scope='Frozen64new biographies and12new participation requests; source, three dictionaries and four heads fixed.',
            evidence_level='frozen_new_context_request_confirmation')
        reference={}
        for site in ['embed','mlp_0','resid_0']:
            path=Path(f'D:/CCAD_Storage/training_curves/SCIENCE04_shift_t{seed}_v1_20260919/tangent_mixed/{site}_seed{seed}.pt')
            if not path.is_file():raise FileNotFoundError(path)
            reference[site]=str(path)
        v['task_adapted_reference']=dict(human=reference,grammar=c['task_adapted_reference']['grammar'])
        path=ROOT/f'configs/rg03_trajectory_confirmation_t{seed}.json';save(path,v);paths.append(path)
    assets=[panel,*paths,ROOT/'scripts/train_intervention_changes.py',ROOT/'scripts/adaptive_native_execution.py',
            ROOT/'scripts/analyze_request_trajectory.py',ROOT/'scripts/prepare_trajectory_confirmation.py',
            ROOT/'src/ccad/intervention_transport.py',ROOT/'src/ccad/request_realization.py',
            Path(c['source_manifest']),Path(c['source_parameters'])]
    save(OUT/'TRAJECTORY_CONFIRMATION_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        primary='Mean later-head participation nRMSE across three fixed further dictionaries; feedback versus local inverse.',
        secondary='Recorded-action mechanism comparison, original head, semantic parts, full request, trained-program and source-direction readout.',
        inference='2000paired profession/gender-stratified document and participation-request bootstrap draws. Source, target dictionary set and later heads fixed.',
        target_seeds=[3,4,5],steps=128,candidate_limit=128,member_allowance='Twice the number of source members at each intervention site.',
        information='Feedback and recorded-action use one extra source-program pass per requested input. Zero target response acquisition or training.',
        identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}))
    print(json.dumps(dict(documents=len(rows),requests=len(queries),tokens=sum(map(len,tokens)),targets=[3,4,5],excluded=len(excluded))))


if __name__=='__main__':main()
