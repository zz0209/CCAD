"""Re-tokenize retained disjoint natural documents for a pinned model tokenizer."""
import os,json,sys,time,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ccad.artifacts import sha256
import numpy as np
from transformers import AutoTokenizer
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,default=Path('D:/CCAD_Storage/models/gpt2-medium/6dcaa7a952f72f9298047fd5137cd6e4f05f41da'))
parser.add_argument('--train-documents',type=Path,default=ROOT/'runs/FINAL5_R14_unique24m_repack_v3_20260908/artifacts/sampled_documents.jsonl')
parser.add_argument('--validation-documents',type=Path,default=ROOT/'runs/R011_NR1_long_budget_corpus_v2_20260904T012000Z/artifacts/sampled_documents.jsonl')
parser.add_argument('--paired-records',type=Path,default=ROOT/'runs/R008a_paired_corpus_v3_20260903T234000Z/artifacts/documents.jsonl')
parser.add_argument('--output',type=Path,default=ROOT/'artifacts/final_three_research_20260909/training_material')
parser.add_argument('--train-tokens',type=int,default=16777216)
parser.add_argument('--validation-tokens',type=int,default=32768)
args=parser.parse_args();start=time.perf_counter();out=args.output;out.mkdir(exist_ok=False,parents=True)
model=args.model
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
dtype=np.dtype('<u4' if len(tokenizer)>65535 else '<u2');storage_name='uint32' if dtype.itemsize==4 else 'uint16'
paths={'train':args.train_documents,'validation':args.validation_documents}
paired=args.paired_records
paired_rows=[json.loads(s) for s in paired.open(encoding='utf-8')]
forbidden_ids={r['document_id'] for r in paired_rows};forbidden_hash={r['text_sha256'] for r in paired_rows}
allrecords=[];outputs={};seen={};inputs=[]
for split,source in paths.items():
 target=(args.train_tokens if split=='train' else args.validation_tokens);buf=[];records=[];total=0
 inputs.append(dict(path=str(source),sha256=sha256(source),bytes=source.stat().st_size))
 for line in source.open(encoding='utf-8'):
  r=json.loads(line)
  if r['split']!=split:continue
  assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
  if r['document_id'] in forbidden_ids or r['text_sha256'] in forbidden_hash:raise ValueError('Paired/train overlap')
  if r['text_sha256'] in seen:raise ValueError('Duplicate train/validation text')
  tokens=tokenizer.encode(r['text'],add_special_tokens=False,verbose=False)+[tokenizer.eos_token_id]
  tokens=tokens[:target-total];a=np.asarray(tokens,dtype=dtype);buf.append(a)
  record={k:v for k,v in r.items() if k!='text'};record.update(included_token_count=len(tokens),start=total,stop=total+len(tokens),included_token_sha256=hashlib.sha256(a.tobytes()).hexdigest())
  records.append(record);seen[r['text_sha256']]=split;total+=len(tokens)
  if total>=target:break
 assert total==target,(split,total,target)
 data=np.concatenate(buf);path=out/(split+'.'+storage_name+'.bin');data.tofile(path);allrecords.extend(records)
 outputs[split]=dict(path=str(path),sha256=sha256(path),tokens=total,documents=len(records),bytes=path.stat().st_size,dtype=dtype.str)
 print(json.dumps(dict(stage='RETOKENIZED',split=split,**outputs[split],wall_seconds=time.perf_counter()-start)),flush=True)
(out/'document_token_records.json').write_text(json.dumps(dict(documents=allrecords),indent=2)+'\n')
manifest=dict(written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),model=str(model),tokenizer_sha256=sha256(model/'tokenizer.json'),source_inputs=inputs,paired_exclusion=dict(path=str(paired),sha256=sha256(paired)),outputs=outputs,wall_seconds=time.perf_counter()-start,context_length=128,token_dtype=dtype.str,generator_script=str(Path(__file__).resolve()),generator_sha256=sha256(Path(__file__)),network_bytes=0,scope='Original natural document order and original train/validation labels retained. Retokenization uses the explicitly selected pinned model; no benchmark text. Training uses the declared token prefix of retained R14 train; validation uses retained R011 validation. Exact text and document disjointness checked against all original paired splits by metadata only.')
(out/'TOKEN_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
