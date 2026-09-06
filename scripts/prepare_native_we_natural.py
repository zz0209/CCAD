"""Source-only selection of natural first-post-we predicate positions."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    start=time.perf_counter();cp=ROOT/'configs/native_we_natural_corpus_v1.json';cfg=json.loads(cp.read_text());rule=cfg['frozen_scope']
    run=ROOT/'runs'/cfg['run_id'];out=ROOT/'artifacts/interpretability_closeout_20260906'
    if json.loads((run/'status.json').read_text())['status']!='PASS' or json.loads((run/'config.resolved.json').read_text())!=cfg:raise ValueError('Frozen corpus not complete')
    sys.path[:0]=[cfg['dependency_overlay_dir']]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['tokenizer_local_dir'],local_files_only=True)
    subject_tokens=[tok.encode(' '+s,add_special_tokens=False) for s in rule['subjects']]
    if any(len(x)!=1 for x in subject_tokens):raise ValueError('Subject is not one token')
    verbs={};split_verbs=[]
    for word in rule['verbs']:
        ids=tok.encode(' '+word,add_special_tokens=False)
        if len(ids)==1:verbs[ids[0]]=word
        else:split_verbs.append(word)
    tokens=np.fromfile(run/'artifacts/calibration.uint16.bin',dtype='<u2').reshape(-1,128)
    sequences=json.loads((run/'artifacts/sequence_records.json').read_text())['sequences'];candidates=[];counts=dict(multiple_document_sequences=0,we_positions=0,eos_prefix=0,unsupported_following_token=0)
    for record in sequences:
        if len(record['document_ids'])!=1:counts['multiple_document_sequences']+=1;continue
        si=record['sequence_index'];ts=tokens[si];doc=record['document_ids'][0]
        for j in np.flatnonzero(ts==subject_tokens[0][0]):
            j=int(j);counts['we_positions']+=1
            if j<rule['minimum_subject_index'] or j+1>=128:continue
            if tok.eos_token_id in ts[:j+2]:counts['eos_prefix']+=1;continue
            if int(ts[j+1]) not in verbs:counts['unsupported_following_token']+=1;continue
            key=hashlib.sha256(f"{rule['selection_salt']}\0{doc}\0{si}\0{j}".encode()).hexdigest()
            candidates.append(dict(document_id=doc,sequence_index=si,subject_index=j,position=j+1,verb=verbs[int(ts[j+1])],selection_hash=key))
    selected=[];used=set()
    for c in sorted(candidates,key=lambda c:c['selection_hash']):
        if c['document_id'] in used:continue
        selected.append(c);used.add(c['document_id'])
        if len(selected)>=rule['maximum_documents']:break
    cases=[]
    for i,c in enumerate(selected):
        natural=tokens[c['sequence_index'],:c['position']+1].tolist()
        for subject,ids in zip(rule['subjects'],subject_tokens):
            ts=natural.copy();ts[c['subject_index']]=ids[0]
            cases.append(dict(c,template=i,subject=subject,text=tok.decode(ts),token_ids=ts))
    counts.update(eligible_positions=len(candidates),eligible_documents=len(set(c['document_id'] for c in candidates)),selected_documents=len(selected),missing=rule['maximum_documents']-len(selected),split_verbs=split_verbs)
    (out/'native_prepared_inputs.json').write_text(json.dumps(dict(cases=cases,subjects=rule['subjects']),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    result=dict(counts=counts,candidates=candidates,selected=selected,seconds=time.perf_counter()-start,forwards=0,inputs=[dict(path=str(p.relative_to(ROOT)),sha256=H(p)) for p in [cp,run/'artifacts/calibration.uint16.bin',run/'artifacts/sequence_records.json',run/'artifacts/documents.jsonl',Path(__file__)]],prepared_sha256=H(out/'native_prepared_inputs.json'))
    (out/'native_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts));print('\n'.join(c['text'] for c in cases if c['subject']=='we'))


if __name__=='__main__':main()
