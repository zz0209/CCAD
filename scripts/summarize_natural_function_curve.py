"""All fixed training stages, source/raw checks and paired endpoint contrasts."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 r=ROOT;out=r/'artifacts/natural_function_curve_20260906';cfg=json.loads((r/'configs/f4_natural_function_curve_v1.json').read_text());quality={(x['seed'],x['step']):x for x in [json.loads(t) for t in (r/'runs/R012_same_stream_curve_v1_20260906/metrics.raw.jsonl').read_text().splitlines()]};records=[];arrays={};reference=None;identities=[]
 for cell in cfg['cells']:
  run=r/'runs'/cell['run'];summary=json.loads((run/'metrics.summary.json').read_text());assert summary['status']=='PASS';rr=[json.loads(t) for t in (run/'metrics.raw.jsonl').read_text().splitlines()];rows=[x for x in rr if x['method'] in cfg['original_methods']];lookup={(x['method'],x['operator'],x['case_id']):x for x in rows};methods=cfg['original_methods'];ops=sorted({x['operator'] for x in rows});ids=sorted({x['case_id'] for x in rows});pairs=sorted({x['pair'] for x in rows});assert len(rows)==768 and len(ids)==24 and len(pairs)==12
  if reference is None:reference=lookup
  for key,x in lookup.items():
   for name in ['source_difference','common_dose','source_kl','text','donor']:assert x[name]==reference[key][name],(cell,key,name)
   if x['method']=='raw':assert x['candidate_kl']==reference[key]['candidate_kl'],(cell,key,'raw')
  with np.load(run/'coefficients.npz') as co:support={m:int(np.count_nonzero(np.linalg.norm(co[m],axis=1))) for m in methods}
  with np.load(run/'probabilities.npz') as probs:
   for x in rows:
    t=x['observed_next_token_id'];i=x['case_id'];op=x['operator'];m=x['method'];x['nll_error']=None if t is None else abs(float(np.log(max(float(probs[f'source_{op}_{i}'][t]),1e-300))-np.log(max(float(probs[f'{m}_{op}_{i}'][t]),1e-300))))
  per={}
  for m in methods:
   a={field:np.array([[lookup[m,op,i][field] if lookup[m,op,i][field] is not None else np.nan for i in ids] for op in ops]) for field in ['normalized_kl_error','candidate_kl','nll_error']};arrays[cell['seed'],cell['step'],m]=a;per[m]=dict(support=support[m],worst_median_kl_ratio=float(np.nanmax(np.nanmedian(a['normalized_kl_error'],axis=1))),worst_median_absolute_kl=float(np.nanmax(np.nanmedian(a['candidate_kl'],axis=1))),worst_median_nll_error=float(np.nanmax(np.nanmedian(a['nll_error'],axis=1))),per_operator={op:dict(median_kl_ratio=float(np.nanmedian(a['normalized_kl_error'][j])),median_absolute_kl=float(np.nanmedian(a['candidate_kl'][j])),median_nll_error=float(np.nanmedian(a['nll_error'][j])),missing_ratios=int(np.isnan(a['normalized_kl_error'][j]).sum())) for j,op in enumerate(ops)})
  q=quality[cell['seed'],cell['step']];records.append(dict(seed=cell['seed'],step=cell['step'],tokens=q['packed_train_tokens'],run=cell['run'],reused=cell['reuse'],quality=q['quality'],decoder_norm_max_error=q['decoder_norm_max_error'],methods=per));identities.append(dict(run=cell['run'],raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),coefficients_sha256=hashlib.sha256((run/'coefficients.npz').read_bytes()).hexdigest(),new_forwards=0 if cell['reuse'] else summary['forwards'],new_wall_seconds=0 if cell['reuse'] else summary['wall_seconds']))
 pairpos=np.array([[i for i in ids if reference[methods[0],ops[0],i]['pair']==pair] for pair in pairs]);rng=np.random.default_rng(20260906);bootids=pairpos[rng.integers(0,len(pairs),size=(2000,len(pairs)))].reshape(2000,-1);changes=[]
 def score(a):return float(np.nanmax(np.nanmedian(a,axis=1)))
 for seed in [1,2]:
  for m in methods:
   metrics={}
   for field in ['normalized_kl_error','candidate_kl','nll_error']:
    early=arrays[seed,256,m][field];late=arrays[seed,8192,m][field];boot=[np.nanmax(np.nanmedian(a[:,bootids],axis=-1),axis=0) for a in [early,late]];pd=[score(late[:,pp])-score(early[:,pp]) for pp in pairpos];metrics[field]=dict(initial=score(early),final=score(late),final_over_initial=score(late)/score(early) if score(early)>0 else None,final_minus_initial=score(late)-score(early),paired_bootstrap_95_percentile=np.quantile(boot[1]-boot[0],[.025,.975]).tolist(),pair_wins=sum(x<0 for x in pd),pair_losses=sum(x>0 for x in pd),ties=sum(x==0 for x in pd))
   changes.append(dict(seed=seed,method=m,metrics=metrics))
 payload=dict(records=records,changes=changes,identities=identities,source_and_raw_exact=True,scope=cfg['scope'],new_forwards=sum(x['new_forwards'] for x in identities),new_run_wall_seconds=sum(x['new_wall_seconds'] for x in identities),witness_new_forwards=75,statistics='Descriptive2000whole-document-pair bootstrap; checkpoint dependence preserved. Conditional on two target trajectories and one source, not independent ten-model replication.')
 (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(new_forwards=payload['new_forwards'],new_run_wall_seconds=payload['new_run_wall_seconds'],cells=[dict(seed=x['seed'],step=x['step'],ce=x['quality']['ce_recovered'],fve=x['quality']['fve'],primary={m:x['methods'][m]['worst_median_kl_ratio'] for m in ['dynamic_pair_ridge','shared16','full','raw']}) for x in records],changes=[x for x in changes if x['method'] in ['dynamic_pair_ridge','shared16','full']]),indent=2))
if __name__=='__main__':main()
