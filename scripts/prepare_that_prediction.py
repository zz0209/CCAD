"""Frozen source-blind natural that context selection, with all eligibility retained."""
import argparse,hashlib,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());rule=cfg['frozen_scope'];run=ROOT/'runs'/cfg['run_id'];out=ROOT/rule['output'];out.mkdir(exist_ok=True)
 assert json.loads((run/'status.json').read_text())['status']=='PASS'
 assert json.loads((run/'config.resolved.json').read_text())==cfg
 sys.path.insert(0,cfg['dependency_overlay_dir']);from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg['tokenizer_local_dir'],local_files_only=True);that=tok.encode(' that',add_special_tokens=False);assert len(that)==1
 ts=np.fromfile(run/'artifacts/calibration.uint16.bin',dtype='<u2').reshape(-1,128);seq=json.loads((run/'artifacts/sequence_records.json').read_text())['sequences'];eligible={k:[] for k in rule['word_lists']}
 for rec in seq:
  if len(rec['document_ids'])!=1:continue
  s=rec['sequence_index'];doc=rec['document_ids'][0]
  for p in np.flatnonzero(ts[s]==that[0]):
   p=int(p);ids=ts[s,:p+1].tolist()
   if len(ids)<12 or tok.eos_token_id in ids:continue
   prefix=tok.decode(ids[:-1]);match=re.search(r'([A-Za-z]+)\s*$',prefix)
   if not match:continue
   word=match[1].lower()
   for family,words in rule['word_lists'].items():
    if word not in words:continue
    key=hashlib.sha256(f"{rule['selection_salt']}/{doc}/{s}/{p}".encode()).hexdigest()
    eligible[family].append(dict(subject=family,family=family,verb=word,sequence=s,position=p,document_id=doc,selection_hash=key,token_ids=ids,text=tok.decode(ids)))
 selected={};used=set();cases=[]
 for family in rule['word_lists']:
  selected[family]=[]
  for r in sorted(eligible[family],key=lambda r:r['selection_hash']):
   if r['document_id'] in used:continue
   selected[family].append(r);used.add(r['document_id']);cases.append(dict(r,template=len(cases)))
   if len(selected[family])==rule['maximum_per_family']:break
 inputs=[a.config,Path(__file__),run/'artifacts/calibration.uint16.bin',run/'artifacts/sequence_records.json']
 payload=dict(cases=cases,eligible=eligible,selected_counts={k:len(v) for k,v in selected.items()},scope=rule,inputs=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in inputs])
 (out/'natural_inputs.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(dict(eligible={k:len(v) for k,v in eligible.items()},selected=payload['selected_counts'],cases=len(cases))))
if __name__=='__main__':main()
