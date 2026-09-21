from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from prepare_shared_column_panel import records


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reuse_generalization_20260921_round04'
BULK=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round04')


def save(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    for seed in [3,4,5]:
        paths=[BULK/f'RG04_HUMAN_{kind}_COUNTERPARTS_T2TO5_20260921'/f'counterparts_t{seed}.json'
               for kind in ['SOURCE','PROPAGATED']]
        a,b=[json.loads(p.read_text()) for p in paths]
        if a['contexts']!=b['contexts'] or a['source_identity']!=b['source_identity']:raise ValueError('Source bank mismatch')
        local=a['supports']['action_span_64'];prop=b['supports']['propagated_span_64']
        filled={part:{site:members if members else prop[part][site] for site,members in sites.items()} for part,sites in local.items()}
        if any(len(filled[p][s])!=len(prop[p][s]) for p in prop for s in prop[p]):raise ValueError('Site counts differ')
        bank=dict(a,supports=dict(action_span_64=local,propagated_span_64=prop,filled_action_span_64=filled),
                  construction='Local-action and propagated banks; the filled-action control uses propagated members only at empty local-action sites.',
                  input_identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        save(OUT/f'HUMAN_COUNTERPARTS_T{seed}.json',bank)
    config=json.loads((ROOT/'configs/rg03_trajectory_development.json').read_text())
    prior={p.resolve() for p in (ROOT/'artifacts').rglob('*.json') if 'panel' in p.name.lower()}
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
        if len(available)<4:raise ValueError(cell)
        rows+=sorted(available,key=lambda r:hashlib.sha256(('rg04propagation/'+r['document_sha256']).encode()).hexdigest())[:4]
    tokenizer=AutoTokenizer.from_pretrained(config['model_local_dir'],local_files_only=True)
    tokens=tokenizer([r['text'] for r in rows],truncation=True,max_length=128,add_special_tokens=True)['input_ids']
    for i,(row,ids) in enumerate(zip(rows,tokens)):row.update(row_id=i,tokens=ids)
    save(OUT/'HUMAN_COUNTERPART_PANEL.json',dict(rows=rows,written_at_utc=datetime.now(timezone.utc).isoformat(),
        excluded_documents=len(excluded),exclusion_files=list(map(str,sorted(prior))),
        selection='Four previously unused dev biographies per profession/gender cell, selected by fixed hash.'))
    print(json.dumps(dict(documents=len(rows),targets=[3,4,5],source_contexts=len(a['contexts']),excluded=len(excluded))))


if __name__=='__main__':main()
