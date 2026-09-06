"""Frozen lexical selection of natural colon prefixes; no model/SAE calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());rule=cfg['frozen_scope'];run=ROOT/'runs'/cfg['run_id']
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    assert json.loads((run/'config.resolved.json').read_text())==cfg
    out=ROOT/rule['output'];out.mkdir(exist_ok=True)
    sys.path.insert(0,cfg['dependency_overlay_dir'])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['tokenizer_local_dir'],local_files_only=True)
    colon=tok.encode(':',add_special_tokens=False);assert len(colon)==1
    ts=np.fromfile(run/'artifacts/calibration.uint16.bin',dtype='<u2').reshape(-1,128)
    seq=json.loads((run/'artifacts/sequence_records.json').read_text())['sequences']
    candidates=[];counts=dict(multi_document_sequences=0,colon_positions=0,short_or_eos=0,unclassified=0)
    for rec in seq:
        if len(rec['document_ids'])!=1:counts['multi_document_sequences']+=1;continue
        si=rec['sequence_index'];doc=rec['document_ids'][0]
        for pos in np.flatnonzero(ts[si]==colon[0]):
            pos=int(pos);counts['colon_positions']+=1;ids=ts[si,:pos+1].tolist()
            if len(ids)<rule['minimum_prefix_tokens'] or tok.eos_token_id in ids:counts['short_or_eos']+=1;continue
            text=tok.decode(ids);prior=text[:-1];line=prior.rsplit('\n',1)[-1].strip()
            words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",line)
            if '\n' in prior and 1<=len(words)<=4 and not re.search(r'[:/\d]',line):family='short_field'
            elif len(words)>=8 and not re.search(r'https?://',line,re.I):family='long_clause'
            else:counts['unclassified']+=1;continue
            key=hashlib.sha256(f"{rule['selection_salt']}\0{doc}\0{si}\0{pos}".encode()).hexdigest()
            candidates.append(dict(document_id=doc,sequence_index=si,token_index=pos,family=family,selection_hash=key,text=text,token_ids=ids))
    selected={};used=set()
    for family in ['short_field','long_clause']:
        selected[family]=[]
        for c in sorted((c for c in candidates if c['family']==family),key=lambda c:c['selection_hash']):
            if c['document_id'] in used:continue
            selected[family].append(c);used.add(c['document_id'])
            if len(selected[family])==rule['maximum_pairs']:break
    pairs=min(map(len,selected.values()));cases=[]
    for i in range(pairs):
        for family in ['short_field','long_clause']:cases.append(dict(selected[family][i],pair=i,role=family,verb='not_applicable'))
    write(out/'prepared_inputs.json',dict(cases=cases,hypotheses=rule['hypotheses'],scope=rule['scope']))
    counts.update(candidate_counts={f:sum(c['family']==f for c in candidates) for f in selected},selected_counts={f:len(v) for f,v in selected.items()},pairs=pairs,unpaired_selected={f:len(v)-pairs for f,v in selected.items()},missing_pairs=rule['maximum_pairs']-pairs)
    paths=[a.config,Path(__file__),run/'artifacts/calibration.uint16.bin',run/'artifacts/documents.jsonl',run/'artifacts/sequence_records.json']
    write(out/'selection.json',dict(counts=counts,candidates=candidates,selected=selected,forwards=0,inputs=[dict(path=str(p),sha256=digest(p)) for p in paths],prepared_sha256=digest(out/'prepared_inputs.json')))
    print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
