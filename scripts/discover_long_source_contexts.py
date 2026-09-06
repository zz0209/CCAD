"""Source-only natural-context discovery from existing sparse codes, no model calls."""
from pathlib import Path
import json,os,time,hashlib,math
from collections import Counter
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]

def main():
 start=time.perf_counter();out=ROOT/'artifacts/long_source_breadth_20260906';out.mkdir(exist_ok=False)
 inputs=[]
 def checked(path,expected=None):
  p=Path(path);p=p if p.is_absolute() else ROOT/p;digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if expected:assert digest==expected
  inputs.append(dict(path=str(p),sha256=digest));return p
 def load(path):return json.loads(checked(path).read_text(encoding='utf-8-sig'))
 checked(Path(__file__));a=load('configs/r011_nr1_k128_paired_codes_v1.json');b=load('configs/f4_long_k128_newseeds_paired_codes_v1.json')
 tm=load(a['token_manifest_path']);tmeta=tm['outputs']['discovery'];tokens=np.memmap(checked(ROOT/'runs'/a['paired_corpus_run']/tmeta['path'],tmeta['sha256']),dtype='<u2',mode='r').reshape(-1,128)
 selected=np.linspace(0,tokens.size-1,8192,dtype=int);tok=AutoTokenizer.from_pretrained(a['model_local_dir'],local_files_only=True)
 for seed,asset in [(2,a),(3,b)]:
  manifest=load(Path(asset['bulk_output_dir'])/'asset_manifest.json');files=next(s for s in manifest['splits'] if s['split']=='discovery')['files'];parts={}
  for f in files:
   if f['seed']==seed:parts[f['dtype']]=np.memmap(checked(f['path'],f['sha256']),dtype='<u2' if f['dtype']=='uint16' else '<f4',mode='r',shape=tuple(f['shape']))[selected]
  ii=parts['uint16'];aa=parts['float32'];hits=[[] for _ in range(3072)]
  for j in range(len(selected)):
   for atom,act in zip(ii[j],aa[j]):
    if act>0:hits[int(atom)].append((float(act),int(selected[j])))
  candidates=[]
  for atom,records in enumerate(hits):
   top=sorted(records,reverse=True)[:24]
   if len(top)<12:continue
   dominant,n=Counter(int(tokens.flat[r]) for v,r in top).most_common(1)[0];seqs=len({r//128 for v,r in top})
   if seqs<4:continue
   examples=[]
   for value,r in top[:8]:
    seq,pos=divmod(r,128);prefix=tokens[seq,:pos].tolist();eos=[k for k,v in enumerate(prefix) if v==0];begin=max(pos-24,eos[-1]+1 if eos else 0,0)
    examples.append(dict(sequence=seq,position=pos,activation=value,token=tok.decode([int(tokens[seq,pos])]),left=tok.decode(tokens[seq,begin:pos].tolist()),right=tok.decode(tokens[seq,pos+1:min(128,pos+17)].tolist())))
   candidates.append(dict(atom=atom,frequency=len(records),dominant_token=tok.decode([dominant]),dominant_token_id=dominant,concentration=n/len(top),score=n/len(top)*math.log(seqs),examples=examples))
  candidates.sort(key=lambda r:(-r['score'],r['atom']))
  (out/f'source{seed}.json').write_text(json.dumps(dict(source_seed=seed,scope='8192 uniformly sampled discovery positions; lexical concentration is candidate ranking only, no target behavior or semantic validation',all_frequencies=[len(x) for x in hits],candidates=candidates),ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'inputs.json').write_text(json.dumps(dict(inputs=inputs,selected_positions=selected.tolist(),wall_seconds=time.perf_counter()-start,no_model_forward=True),indent=2))
 print(json.dumps(dict(status='PASS',source_seeds=[2,3],positions_per_seed=len(selected),wall_seconds=time.perf_counter()-start,output=str(out))))
if __name__=='__main__':main()
