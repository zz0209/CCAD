"""Recover exact retained non-audit document prefixes, then tokenize for GPT-2."""
import os,sys,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ccad.artifacts import sha256
from ccad.data_manifest import canonical_sha256
import numpy as np
from transformers import AutoTokenizer
start=time.perf_counter();parent=ROOT/'runs/R008a_paired_corpus_v3_20260903T234000Z';out=ROOT/'artifacts/final_three_research_20260909/paired_material';out.mkdir(exist_ok=False)
cfg=json.loads((parent/'config.resolved.json').read_text());manifest=json.loads((parent/'artifacts/token_manifest.json').read_text());docs=[json.loads(s) for s in (parent/'artifacts/documents.jsonl').open()]
old=AutoTokenizer.from_pretrained(cfg['tokenizer_local_dir'],local_files_only=True);new=AutoTokenizer.from_pretrained('D:/CCAD_Storage/models/gpt2-medium/6dcaa7a952f72f9298047fd5137cd6e4f05f41da',local_files_only=True)
records=[];outputs={};exclusions=[]
for split in ['mean','discovery','calibration']:
 info=manifest['outputs'][split];p=parent/info['path'];assert sha256(p)==info['sha256'];tokens=np.fromfile(p,dtype='<u2').tolist();assert tokens[0]==old.eos_token_id;cursor=1;packed=[new.eos_token_id];spans=[]
 selected=sorted((r for r in docs if r['split']==split),key=lambda r:hashlib.sha256((cfg['selection_salt']+'-order\0'+r['document_id']).encode()).hexdigest())
 for r in selected:
  n=r['included_token_count'];ids=tokens[cursor:cursor+n];cursor+=n+1
  if len(ids)!=n:
   exclusions.append(dict(split=split,document_id=r['document_id'],reason='Final document truncated by old packed stream; full included-token hash cannot be checked'));continue
  assert canonical_sha256(ids)==r['included_token_sha256']
  kept=ids[:];removed=0
  while True:
   text=old.decode(kept,clean_up_tokenization_spaces=False,skip_special_tokens=False)
   if old.encode(text,add_special_tokens=False,verbose=False)==kept:break
   kept=kept[:-1];removed+=1
   if removed>8:raise ValueError('Non-boundary tokenizer roundtrip mismatch')
  newids=new.encode(text,add_special_tokens=False,verbose=False);s=len(packed);packed+=newids+[new.eos_token_id]
  record=dict(r,recovered_original_tokens=n,reversible_prefix_tokens=len(kept),removed_boundary_tokens=removed,recovered_text_sha256=hashlib.sha256(text.encode()).hexdigest(),gpt2_start=s,gpt2_stop=s+len(newids),gpt2_tokens=len(newids));records.append(record);spans.append(record)
 packed=packed[:len(packed)//128*128];dest=out/(split+'.uint16.bin');np.asarray(packed,dtype='<u2').tofile(dest)
 seq=[]
 for i in range(len(packed)//128):seq.append(dict(sequence_index=i,document_ids=[r['document_id'] for r in spans if r['gpt2_start']<(i+1)*128 and r['gpt2_stop']>i*128]))
 (out/(split+'_sequence_records.json')).write_text(json.dumps(seq,indent=2)+'\n')
 outputs[split]=dict(path=str(dest),sha256=sha256(dest),tokens=len(packed),sequences=len(packed)//128,documents=len(spans),source_token_sha256=info['sha256'])
 print(json.dumps(outputs[split]),flush=True)
(out/'document_prefix_records.json').write_text(json.dumps(dict(documents=records,exclusions=exclusions),indent=2)+'\n')
(out/'TOKEN_MANIFEST.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),source_run=parent.name,outputs=outputs,wall_seconds=time.perf_counter()-start,audit_tokens_read=False,network_bytes=0,scope='Exact original non-audit document-prefix token hashes verified. Reversible Pythia decode and GPT-2 encode; original hash-split membership retained. Last truncated document excluded per split because full prefix hash unavailable; rare incomplete UTF-8 boundary tokens removed and recorded. No claim to recover full original documents.'),indent=2)+'\n')
