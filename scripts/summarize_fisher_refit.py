"""Prespecified fresh-input endpoints, with document-pair bootstrap."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 r=ROOT;run=r/'runs/F4_fisher_refit_apply_s1_v1_20260906';assert json.loads((run/'status.json').read_text())['status']=='PASS';out=r/'artifacts/fisher_refit_20260906';freeze=json.loads((out/'freeze.json').read_text());rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()];methods=sorted({r['method'] for r in rows});ops=sorted({r['operator'] for r in rows});pairs=sorted({r['pair'] for r in rows});cases=sorted({r['case_id'] for r in rows});lookup={(r['method'],r['operator'],r['case_id']):r for r in rows};nextcount=0
 with np.load(run/'probabilities.npz') as probs:
  for row in rows:
   i,op,m=row['case_id'],row['operator'],row['method'];t=row['observed_next_token_id'];row['observed_nll_response_error']=None;row['source_nll_response']=None;row['candidate_nll_response']=None
   if t is not None:
    ps,pc,pb=[float(probs[k][t]) for k in [f'source_{op}_{i}',f'{m}_{op}_{i}',f'base_{i}']];source=-np.log(max(ps,1e-300))+np.log(max(pb,1e-300));candidate=-np.log(max(pc,1e-300))+np.log(max(pb,1e-300));row.update(observed_nll_response_error=abs(float(candidate-source)),source_nll_response=float(source),candidate_nll_response=float(candidate))
  nextcount=sum(lookup[methods[0],ops[0],i]['observed_next_token_id'] is not None for i in cases)
 arrays={m:{metric:np.array([[lookup[m,op,i][metric] if lookup[m,op,i][metric] is not None else np.nan for i in cases] for op in ops]) for metric in ['normalized_kl_error','candidate_kl','observed_nll_response_error']} for m in methods}
 def score(a):return float(np.nanmax(np.nanmedian(a,axis=-1)))
 stats={m:dict(primary_worst_operator_median_kl_ratio=score(a['normalized_kl_error']),worst_operator_median_absolute_kl=score(a['candidate_kl']),worst_operator_median_observed_nll_error=score(a['observed_nll_response_error']),per_operator={op:dict(median_kl_ratio=float(np.nanmedian(a['normalized_kl_error'][j])),median_absolute_kl=float(np.nanmedian(a['candidate_kl'][j])),median_observed_nll_error=float(np.nanmedian(a['observed_nll_response_error'][j])),missing_source_ratios=int(np.isnan(a['normalized_kl_error'][j]).sum())) for j,op in enumerate(ops)}) for m,a in arrays.items()}
 rng=np.random.default_rng(20260906);pair_positions=[[cases.index(i) for i in cases if lookup[methods[0],ops[0],i]['pair']==pair] for pair in pairs];assert all(len(p)==2 for p in pair_positions);positions=np.array(pair_positions)[rng.integers(0,len(pairs),size=(2000,len(pairs)))].reshape(2000,-1);comparisons={}
 for metric in ['normalized_kl_error','candidate_kl','observed_nll_response_error']:
  boot={m:np.nanmax(np.nanmedian(a[metric][:,positions],axis=-1),axis=0) for m,a in arrays.items()};comparisons[metric]={}
  for baseline in methods:
   if baseline=='pointwise_fisher_refit':continue
   delta=boot['pointwise_fisher_refit']-boot[baseline];point=arrays['pointwise_fisher_refit'][metric];other=arrays[baseline][metric];pairdiff=[score(point[:,pp])-score(other[:,pp]) for pp in pair_positions]
   comparisons[metric][baseline]=dict(point_minus_baseline=score(point)-score(other),paired_bootstrap_95_percentile=np.quantile(delta,[.025,.975]).tolist(),document_pair_wins=sum(d<0 for d in pairdiff),document_pair_losses=sum(d>0 for d in pairdiff),ties=sum(d==0 for d in pairdiff))
 (out/'endpoint_rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
 summary=dict(freeze=freeze,scope='Single source3->target1, fresh disjoint documents for this frozen method. Twelve or fewer reciprocal document pairs, not seed-population inference; no retuning.',cases=len(cases),document_pairs=len(pairs),observed_next_tokens=nextcount,methods=stats,comparisons=comparisons,input_hashes={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'metrics.raw.jsonl',run/'probabilities.npz',run/'coefficients.npz',out/'freeze.json',Path(__file__)]})
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),document_pairs=len(pairs),observed_next_tokens=nextcount,methods={m:{k:v for k,v in s.items() if k!='per_operator'} for m,s in stats.items()},comparisons={metric:{b:vals for b,vals in cs.items() if b in ['shared16','euclidean_refit','constant_fisher_refit']} for metric,cs in comparisons.items()}),indent=2))
if __name__=='__main__':main()
