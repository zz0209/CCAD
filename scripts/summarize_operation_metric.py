"""Describe all cells of the fixed single-metric development comparison."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 o=ROOT/'artifacts/operation_metric_20260906';p=json.loads((o/'calibration_results.json').read_text());cells=p['cells'];assert len(cells)==16;rows=[]
 for c in cells:
  m=c['methods'];a,b=m['shared16'],m['energy16']
  with np.load(ROOT/'runs'/c['run']/'coefficients.npz') as z:
   sa=set(np.flatnonzero(np.linalg.norm(z['shared16'],axis=1)));sb=set(np.flatnonzero(np.linalg.norm(z['energy16'],axis=1)))
  rows.append(dict(query=c['query'],seed=c['seed'],run=c['run'],old=a['mean_iid_donor_hook_error'],new=b['mean_iid_donor_hook_error'],ratio=b['mean_iid_donor_hook_error']/a['mean_iid_donor_hook_error'],component_ratios=(np.array(b['component_variance'])/a['component_variance']).tolist(),standardized_ratio=b['standardized_error']/a['standardized_error'],old_support=a['support'],new_support=b['support'],support_jaccard=len(sa&sb)/len(sa|sb),old_worst=max(a['operator_iid_donor_hook_error'].values()),new_worst=max(b['operator_iid_donor_hook_error'].values())))
 agg={}
 for q in ['all',0,1,2,3]:
  rr=[x for x in rows if q=='all' or x['query']==q];old=np.mean([x['old'] for x in rr]);new=np.mean([x['new'] for x in rr]);agg[q]=dict(old=float(old),new=float(new),ratio=float(new/old),improved=sum(x['new']<x['old']*(1-1e-10) for x in rr),worse=sum(x['new']>x['old']*(1+1e-10) for x in rr),ties=sum(abs(x['ratio']-1)<=1e-10 for x in rr),total=len(rr),both_components_improve=sum(max(x['component_ratios'])<1-1e-10 for x in rr),worst_improves=sum(x['new_worst']<x['old_worst']*(1-1e-10) for x in rr))
 costs=[json.loads((ROOT/'runs'/c['run']/'metrics.summary.json').read_text()) for c in cells]
 summary=dict(rows=rows,aggregate=agg,legacy_coefficients_exact=p['legacy_coefficients_exact'],wall_seconds=sum(x['wall_seconds'] for x in costs),wrapper_seconds=p['seconds'],lm_forwards=sum(x['forwards'] for x in costs),calibration_results_sha256=hashlib.sha256((o/'calibration_results.json').read_bytes()).hexdigest(),scope='Existing calibration surrogate development; no behavioral or statistical-population claim')
 (o/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
