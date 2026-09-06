"""Source-blind input construction for frozen long-source correspondence."""
import json
from pathlib import Path
import re
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    from transformers import AutoTokenizer
    asset=json.loads((ROOT/'configs/r011_nr1_k128_paired_codes_v1.json').read_text())
    tok=AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True)
    prior=json.loads((ROOT/'configs/native_that_roles_s1a2379_v1.json').read_text())
    cases=[]
    for frame in ['After reviewing the records, the analyst {verb} that','The editor says the committee {verb} that']:
        for left,right in prior['pairs']:
            pair=len(cases)//2
            cases.extend(dict(pair=pair,role=role,verb=v,family='new_frame',text=frame.format(verb=v)) for role,v in [('report',left),('attitude',right)])
    for v in ['hoped','feared','suspected','believed','wished','expected']:
        pair=len(cases)//2
        cases.extend([dict(pair=pair,role='direct',verb=v,family='syntactic_control',text=f'The investigator {v} that'),
                      dict(pair=pair,role='relative',verb=v,family='syntactic_control',text=f'The investigator {v} the account that')])
    words=dict(report=['say','says','said','report','reports','reported','show','shows','showed','found','find','finds','confirm','confirmed','confirms','observe','observed','observes','note','noted','notes','demonstrate','demonstrated','demonstrates'],
               attitude=['believe','believes','believed','hope','hopes','hoped','fear','fears','feared','suspect','suspects','suspected','expect','expects','expected','think','thinks','thought','wish','wishes','wished','feel','feels','felt'])
    corpus=ROOT/'runs/NATIVE_we_natural_corpus_v1_20260906/artifacts'
    tm=json.loads((corpus/'token_manifest.json').read_text())
    tokens=np.memmap(corpus.parent/tm['outputs']['calibration']['path'],dtype='<u2',mode='r').reshape(-1,128)
    seqs=json.loads((corpus/'sequence_records.json').read_text())['sequences'];docmap={r['sequence_index']:r['document_ids'] for r in seqs}
    that=tok.encode(' that',add_special_tokens=False)[0];eligible={r:[] for r in words}
    for s,p in np.argwhere(tokens==that):
        s,p=int(s),int(p);eos=np.flatnonzero(tokens[s,:p]==0);begin=max(0,p-95,int(eos[-1])+1 if len(eos) else 0)
        if p-begin<12:continue
        prefix=tok.decode(tokens[s,begin:p].astype(int).tolist());match=re.search(r'([A-Za-z]+)\s*$',prefix)
        if not match:continue
        word=match[1].lower()
        for role,allowed in words.items():
            if word in allowed:
                key=hashlib.sha256((' '.join(docmap[s])+f'/{s}/{p}').encode()).hexdigest()
                eligible[role].append(dict(role=role,verb=word,sequence=s,position=p,document_ids=docmap[s],selection_hash=key,
                    family='natural_reused_corpus',token_ids=tokens[s,begin:p+1].astype(int).tolist(),text=tok.decode(tokens[s,begin:p+1].astype(int).tolist())))
    used=set();selected={r:[] for r in words}
    for role in words:
        for r in sorted(eligible[role],key=lambda r:r['selection_hash']):
            if used.intersection(r['document_ids']):continue
            selected[role].append(r);used.update(r['document_ids'])
            if len(selected[role])==6:break
    n=min(map(len,selected.values()))
    for i in range(n):
        pair=len(cases)//2
        cases.extend(dict(selected[role][i],pair=pair) for role in words)
    out=ROOT/'artifacts/long_source2379_contexts_20260906';out.mkdir(exist_ok=False)
    payload=dict(cases=cases,word_lists=words,eligible=eligible,natural_selected=selected,natural_pairs_used=n,
                 selection='Lexical final predecessor, >=12within-sequence/document tokens, hash order, no repeated document across natural cases; no activations/target data.',
                 source_corpus=str(corpus.parent),scope='Authored new frames + same-verb syntax + natural previously used unrelated corpus. Development, not fresh independent confirmation.')
    (out/'inputs.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),natural_eligible={k:len(v) for k,v in eligible.items()},natural_pairs=n,path=str(out/'inputs.json'))))


if __name__=='__main__':main()
