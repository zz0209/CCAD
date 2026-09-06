"""Frozen source-only natural and-prefix selection; no model or SAE calls."""
import hashlib,json,re,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    start=time.perf_counter();cfg=json.loads((ROOT/'configs/f4_and_natural_corpus_v1.json').read_text());rule=cfg['frozen_scope']
    run=ROOT/'runs'/cfg['run_id'];out=ROOT/'artifacts/and_natural_confirmation_20260906'
    if json.loads((run/'status.json').read_text())['status']!='PASS':raise ValueError('Corpus not complete')
    if json.loads((run/'config.resolved.json').read_text())!=cfg:raise ValueError('Frozen corpus config changed')
    sys.path[:0]=[cfg['dependency_overlay_dir']]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['tokenizer_local_dir'],local_files_only=True)
    conjunctions=[tok.encode(s,add_special_tokens=False) for s in rule['conjunctions']]
    clause=[tok.encode(s,add_special_tokens=False) for s in rule['clause_start_tokens']]
    if any(len(x)!=1 for x in conjunctions+clause):raise ValueError('Non-single token')
    tokens=np.fromfile(run/'artifacts/calibration.uint16.bin',dtype='<u2').reshape(-1,128)
    seq=json.loads((run/'artifacts/sequence_records.json').read_text())['sequences'];candidates=[];counts=dict(multiple_document_sequences=0,eos_prefix=0)
    for record in seq:
        if len(record['document_ids'])!=1:counts['multiple_document_sequences']+=1;continue
        s=record['sequence_index'];t=tokens[s];doc=record['document_ids'][0]
        for j in np.flatnonzero(t==conjunctions[0][0]):
            j=int(j)
            if j+1<rule['minimum_prefix_tokens']:continue
            if tok.eos_token_id in t[:j+1]:counts['eos_prefix']+=1;continue
            past32=tok.decode(t[max(0,j-32):j].tolist());past16=tok.decode(t[max(0,j-16):j].tolist())
            between=bool(re.search(r'\bbetween\b',past32,re.I));commas=past16.count(',')
            family='constrained' if between or commas>=2 else 'ordinary'
            key=hashlib.sha256(f"{rule['selection_salt']}\0{doc}\0{s}\0{j}".encode()).hexdigest()
            candidates.append(dict(document_id=doc,sequence_index=s,token_index=j,family=family,has_between=between,preceding16_comma_count=commas,selection_hash=key))
    selected=[];used=set()
    for family in ['ordinary','constrained']:
        for c in sorted((c for c in candidates if c['family']==family),key=lambda c:c['selection_hash']):
            if c['document_id'] in used:continue
            if sum(x['family']==family for x in selected)>=rule['maximum_pairs_per_family']:break
            selected.append(c);used.add(c['document_id'])
    cases=[]
    for i,c in enumerate(selected):
        prefix=tokens[c['sequence_index'],:c['token_index']].tolist()
        for text,ids in zip(rule['conjunctions'],conjunctions):
            ts=prefix+ids
            cases.append(dict(c,pair_index=i,topic=c['document_id'],conjunction=text,text=tok.decode(ts),token_ids=ts))
    payload=dict(cases=cases,clause_token_ids=[x[0] for x in clause],item_token_ids=[None]*len(selected),endpoint='pronoun_logodds')
    (out/'prepared_inputs.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    counts.update(candidates=len(candidates),candidate_documents=len(set(c['document_id'] for c in candidates)),families={f:dict(candidates=sum(c['family']==f for c in candidates),selected=sum(c['family']==f for c in selected),missing=rule['maximum_pairs_per_family']-sum(c['family']==f for c in selected)) for f in ['ordinary','constrained']})
    (out/'selection.json').write_text(json.dumps(dict(counts=counts,candidates=candidates,selected=selected,seconds=time.perf_counter()-start,forwards=0,inputs=[dict(path=str(p.relative_to(ROOT)),sha256=H(p)) for p in [run/'config.resolved.json',run/'artifacts/calibration.uint16.bin',run/'artifacts/documents.jsonl',run/'artifacts/sequence_records.json',Path(__file__)]],prepared_sha256=H(out/'prepared_inputs.json')),indent=2)+'\n')
    print(json.dumps(counts));print('\n'.join(f"{c['family']} {c['sequence_index']}:{c['token_index']} {c['text']}" for c in cases[::2]))


if __name__=='__main__':main()
