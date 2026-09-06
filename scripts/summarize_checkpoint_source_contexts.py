import json,numpy as np
from pathlib import Path
r=Path.cwd();out=r/'artifacts/five_seed_training_20260906';run=r/'runs/R012_source_census_s1_v2_20260906';qs=json.loads((run/'selected_queries.json').read_text());cfg=json.loads((run/'config.resolved.json').read_text());asset=json.loads((r/cfg['asset_config']).read_text());tm=json.loads((r/asset['token_manifest_path']).read_text());t=tm['outputs']['discovery'];tokens=np.memmap(r/'runs'/asset['paired_corpus_run']/t['path'],dtype='<u2',mode='r').reshape(-1,128)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True)
with np.load(run/'source_codes.npz') as z:ids=z['indices'];acts=z['activations'];pos=z['positions']
result=[]
for q in qs['queries']:
 eligible=np.flatnonzero(tokens.ravel()[pos]==q['token_id']);a=np.array([np.sum(np.where(ids==atom,acts,0),axis=1) for atom in q['atoms']]).T;scale=np.sqrt(np.mean(a[eligible]**2,axis=0));score=(a[:,0]/max(scale[0],1e-12)-a[:,1]/max(scale[1],1e-12));groups={}
 for role,order in [('A_high',eligible[np.argsort(-score[eligible])]),('B_high',eligible[np.argsort(score[eligible])])]:
  used=set();items=[]
  for idx in order:
   seq,k=divmod(int(pos[idx]),128)
   if seq in used or k<10:continue
   eos=np.flatnonzero(tokens[seq,:k]==0);begin=max(0,k-48,int(eos[-1])+1 if len(eos) else 0)
   if k-begin<10:continue
   used.add(seq);items.append(dict(sequence=seq,position=k,activations=a[idx].tolist(),standardized_contrast=float(score[idx]),prefix=tok.decode(tokens[seq,begin:k+1].astype(int).tolist()),following_for_display_only=tok.decode(tokens[seq,k+1:min(k+9,128)].astype(int).tolist())))
   if len(items)==3:break
  groups[role]=items
 result.append(dict(token=q['token'],atoms=q['atoms'],sampled_token_positions=len(eligible),rms_scale=scale.tolist(),contrasts=groups))
(out/'source_contrasts.json').write_text(json.dumps(dict(queries=result,scope='Post-selection source-only development examples. Rank source A/RMS-A minus B/RMS-B within token; 3 distinct sequences per extreme, prefix>=10 tokens. Following text is display-only and cannot explain the current causal prefix. No target or functional outcome used.'),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
