from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from prepare_shared_column_panel import records


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round02'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round02')


def save(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    c=json.loads((ROOT/'configs/rg02_finite_support_development_t2.json').read_text())
    prior={p.resolve() for p in (ROOT/'artifacts').rglob('*.json') if 'panel' in p.name.lower()}
    for path in (ROOT/'configs').glob('*.json'):
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value,dict):continue
        for key in ['evaluation_panel','panel','fit_panel']:
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
    buckets={(p,g):[] for p in [5,25,12,24] for g in [0,1]}
    seen=set(excluded)
    for row in table.to_pylist():
        text=row['hard_text'];digest=hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        buckets[row['profession'],row['gender']].append(dict(text=text,document_sha256=digest,
            profession=row['profession'],gender=row['gender'],split='dev'))
    selected=[]
    for key,rows in buckets.items():
        assert len(rows)>=16,(key,len(rows))
        selected+=sorted(rows,key=lambda r:hashlib.sha256(('rg02finite/'+r['document_sha256']).encode()).hexdigest())[:16]
    tokenizer=AutoTokenizer.from_pretrained(c['model_local_dir'],local_files_only=True)
    tokens=tokenizer([r['text'] for r in selected],truncation=True,max_length=2048,add_special_tokens=True)['input_ids']
    for i,(row,token) in enumerate(zip(selected,tokens)):row.update(row_id=i,tokens=token)
    assert not {r['document_sha256'] for r in selected}&excluded
    panel=OUT/'FINITE_CONFIRMATION_PANEL.json'
    save(panel,dict(rows=selected,evidence='128new biographies excluded from all recorded panels, selected by metadata and hash before model evaluation.',
        exclusion_files=list(map(str,sorted(prior))),excluded_documents=len(excluded),remaining_counts={str(k):len(v) for k,v in buckets.items()}))
    original_requests=json.loads(Path(c['request_panel']).read_text())
    req=dict(queries=[q for q in original_requests['queries'] if not q.startswith(('interior','boundary','member'))],dose_queries={},member_queries={})
    rng=np.random.default_rng(2026092102)
    groups=['pronouns','names','associated_words']
    for kind in ['interior','boundary']:
        for i in range(6):
            name=f'{kind}_{i:02d}';values=rng.uniform(.05,.95,3)
            if kind=='boundary':values[i%3]=(i//3)%2
            req['queries'].append(name);req['dose_queries'][name]=dict(zip(groups,values.tolist()))
    source=json.loads(Path(c['source_manifest']).read_text())
    for i in range(8):
        name=f'member_subset_{i:02d}';req['queries'].append(name)
        req['member_queries'][name]={s:rng.integers(0,2,len(ids)).tolist() for s,ids in source['members'].items()}
    requests=OUT/'FINITE_CONFIRMATION_REQUESTS.json';save(requests,req)
    fields={'transfer_gain':str(BULK/'RG02_FINITE_EMBEDDING_DEVELOPMENT_T2_20260921/gain_field.pt'),
            'transfer_sparse':str(BULK/'RG02_FINITE_SUPPORT_DEVELOPMENT_T2_20260921/sparse_field.pt')}
    frozen={'saved_fixed':str(BULK/'RG02_FIXED_DIRECT_DEVELOPMENT_T2_20260921/fixed_direct_256.pt'),
            'saved_sparse':str(BULK/'RG02_FINITE_SUPPORT_DEVELOPMENT_T2_20260921/sparse_256.pt')}
    paths=[]
    for seed in [2,4,5]:
        v=c.copy()
        v.update(run_id=f'RG02_FINITE_CONFIRMATION_T{seed}_20260921',target_seed=seed,seeds=[seed],
            arms=[],steps=0,checkpoints=[],evaluation_panel=str(panel),request_panel=str(requests),evaluation_per_cell=0,
            evaluation_description='Frozen128new-biography27request confirmation. Target2learned coefficients; targets4/5reuse target2physical fields with no response fitting.',
            evidence_level='frozen_new_context_request_confirmation',audit_opened=True,candidate_family_frozen=True,
            baseline_methods=['geometric','program','readout'],
            frozen_coefficients=frozen if seed==2 else {},transfer_fields={} if seed==2 else fields,
            reference_program_directory=f'D:/CCAD_Storage/training_curves/SCIENCE04_shift_t{seed}_v1_20260919/tangent_mixed',
            statistics_unit='New biographies and requests; source explanation and finite target pool fixed.',
            purpose='Confirm frozen finite-response support fitting and field reuse using new biographies and requests.',
            scope='One fixed published55member program. Only target2provided responses for fitted fields; targets4/5are held from this fit.')
        assert Path(v['reference_program_directory']).is_dir()
        path=ROOT/f'configs/rg02_finite_confirmation_t{seed}.json';save(path,v);paths.append(path)
    smoke=json.loads((ROOT/'configs/rg02_fixed_direct_smoke_t2.json').read_text())
    smoke.update(run_id='RG02_SAVED_EXECUTION_SMOKE_T2_20260921',arms=[],steps=0,checkpoints=[],frozen_coefficients=frozen)
    save(ROOT/'configs/rg02_saved_execution_smoke_t2.json',smoke)
    assets=[panel,requests,*paths,ROOT/'scripts/fit_finite_embedding_transport.py',ROOT/'scripts/analyze_finite_embedding_transport.py',
            ROOT/'scripts/prepare_finite_execution_confirmation.py',*map(Path,fields.values()),*map(Path,frozen.values())]
    save(OUT/'FINITE_CONFIRMATION_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        primary='Mean later-head per-request nRMSE on12new participation requests. Target2 saved_sparse versus geometric, with saved_fixed mechanism contrast. Held-target4/5 transfer_sparse versus geometric and transfer_gain.',
        secondary='Semantic endpoints,8new member masks, full intervention, original head and representation error; strong program and source-direction readout retained.',
        inference='2000paired profession/gender-stratified biography and participation-request resamples. The one source, two held target dictionaries and four classifier heads remain fixed.',
        fit_target=2,held_targets=[4,5],fit_steps=256,member_allowance=20,
        identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}))
    print(json.dumps(dict(documents=len(selected),requests=len(req['queries']),excluded=len(excluded),targets=[2,4,5])))


if __name__=='__main__':
    main()
