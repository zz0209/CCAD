"""Fixed-panel endpoint contrasts with query-stratified whole-pair uncertainty."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIELDS=['candidate_kl','normalized_kl_error','nll_error']
def score(a):return float(np.nanmax(np.nanmedian(a,axis=1)))
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def main():
 r=ROOT;out=r/'artifacts/five_seed_function_20260906';cfg=json.loads((r/'configs/f4_five_seed_function_v1.json').read_text());methods=cfg['methods'];records=[];arrays={};references={};boots={};identities=[];wrong=[];apostrophe=[];rng=np.random.default_rng(20260906)
 for cell in cfg['cells']:
  qi=cell['query'];run=r/'runs'/cell['apply_run'];summary=json.loads((run/'metrics.summary.json').read_text());assert summary['status']=='PASS';assert json.loads((run/'contract_validation.json').read_text())['ok'];rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()];lookup={(x['method'],x['operator'],x['case_id']):x for x in rows};ids=sorted({x['case_id'] for x in rows});ops=sorted({x['operator'] for x in rows});pairs=sorted({x['pair'] for x in rows});assert len(rows)==len(methods)*len(ops)*len(ids) and len(ids)==2*len(pairs)
  if qi not in references:
   references[qi]=lookup;pp=np.array([[i for i in ids if lookup[methods[0],ops[0],i]['pair']==p] for p in pairs]);boots[qi]=pp[rng.integers(0,len(pairs),size=(2000,len(pairs)))].reshape(2000,-1)
  for key,x in lookup.items():
   for name in ['source_difference','source_activation','common_dose','source_kl','text','donor']:assert x[name]==references[qi][key][name],(cell,key,name)
   if x['method']=='raw':assert x['candidate_kl']==references[qi][key]['candidate_kl'],(cell,key,'raw')
  with np.load(run/'probabilities.npz') as probs:
   token_ids=probs['retained_token_ids'].tolist();token_index={t:i for i,t in enumerate(token_ids)};probe=json.loads((run/'config.resolved.json').read_text())['probe_token_ids'];assert len(probe)==1
   for x in rows:
    t=x['observed_next_token_id'];i=x['case_id'];op=x['operator'];m=x['method'];x['nll_error']=None if t is None else abs(float(np.log(max(float(probs[f'source_{op}_{i}'][token_index[t]]),1e-300))-np.log(max(float(probs[f'{m}_{op}_{i}'][token_index[t]]),1e-300))))
    if qi==1 and cell['seed']==2 and cell['step']==256 and m=='raw':
     ti=token_index[probe[0]];apostrophe.append(dict(case_id=i,operator=op,text=x['text'],source_difference=x['source_difference'],source_activation=x['source_activation'],dose=x['common_dose'],base_t=float(probs[f'base_{i}'][ti]),source_t=float(probs[f'source_{op}_{i}'][ti]),log_t_response=float(np.log(max(float(probs[f'source_{op}_{i}'][ti]),1e-300))-np.log(max(float(probs[f'base_{i}'][ti]),1e-300)))))
  with np.load(run/'coefficients.npz') as co:support={m:int(np.count_nonzero(np.linalg.norm(co[m],axis=1))) for m in methods}
  per={}
  for m in methods:
   a={f:np.array([[lookup[m,op,i][f] if lookup[m,op,i][f] is not None else np.nan for i in ids] for op in ops]) for f in FIELDS};arrays[qi,cell['seed'],cell['step'],m]=a;per[m]=dict(support=support[m],**{f:score(v) for f,v in a.items()},per_operator={op:{f:float(np.nanmedian(a[f][j])) for f in FIELDS} for j,op in enumerate(ops)},missing={f:int(np.isnan(v).sum()) for f,v in a.items()})
  wr=[x for x in rows if x['method']=='wrong_query'];wrong.append(dict(run=cell['apply_run'],matched=sum(x['control_energy_matched'] for x in wr),total=len(wr),zero_predicted_energy=sum(x['control_energy_scale']==0 and x['source_delta_energy']>1e-20 for x in wr)))
  records.append(dict(query=qi,token=cell['token'],seed=cell['seed'],step=cell['step'],run=cell['apply_run'],recipients=len(ids),pairs=len(pairs),methods=per));identities.append(dict(run=cell['apply_run'],raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),probabilities_sha256=hashlib.sha256((run/'probabilities.npz').read_bytes()).hexdigest(),forwards=summary['forwards'],wall_seconds=summary['wall_seconds']))
 changes=[];boot_changes={}
 for qi in range(4):
  for seed in [2,3,4,5]:
   for m in methods:
    mm={}
    for field in FIELDS:
     e,l=[arrays[qi,seed,step,m][field] for step in [256,8192]];b=[np.nanmax(np.nanmedian(a[:,boots[qi]],axis=-1),axis=0) for a in [e,l]];delta=b[1]-b[0];boot_changes[qi,seed,m,field]=delta;pair_d=[score(l[:,j:j+2])-score(e[:,j:j+2]) for j in range(0,e.shape[1],2)];mm[field]=dict(early=score(e),late=score(l),delta=score(l)-score(e),late_over_early=score(l)/score(e) if score(e)>0 else None,bootstrap_95=np.quantile(delta,[.025,.975]).tolist(),pair_wins=sum(x<0 for x in pair_d),pair_losses=sum(x>0 for x in pair_d),ties=sum(x==0 for x in pair_d))
    changes.append(dict(query=qi,seed=seed,method=m,metrics=mm))
 aggregate={};per_query={};method_comparisons={}
 for m in methods:
  aggregate[m]={};per_query[m]={}
  for field in FIELDS:
   chosen=[x['metrics'][field] for x in changes if x['method']==m];bv=np.mean([boot_changes[q,s,m,field] for q in range(4) for s in [2,3,4,5]],axis=0);early=float(np.mean([x['early'] for x in chosen]));late=float(np.mean([x['late'] for x in chosen]));aggregate[m][field]=dict(early=early,late=late,delta=late-early,late_over_early=late/early if early>0 else None,bootstrap_95=np.quantile(bv,[.025,.975]).tolist(),improved_cells=sum(x['delta']<0 for x in chosen),total_cells=len(chosen))
  for qi in range(4):
   vv=[x['metrics']['candidate_kl'] for x in changes if x['method']==m and x['query']==qi];bv=np.mean([boot_changes[qi,seed,m,'candidate_kl'] for seed in [2,3,4,5]],axis=0);per_query[m][qi]=dict(early=float(np.mean([v['early'] for v in vv])),late=float(np.mean([v['late'] for v in vv])),delta=float(np.mean([v['delta'] for v in vv])),bootstrap_95=np.quantile(bv,[.025,.975]).tolist(),improved_targets=sum(v['delta']<0 for v in vv))
 for base in methods:
  if base=='shared16':continue
  deltas=[];boot=[]
  for qi in range(4):
   for seed in [2,3,4,5]:
    a,b=[arrays[qi,seed,8192,m]['candidate_kl'] for m in ['shared16',base]];deltas.append(score(a)-score(b));boot.append(np.nanmax(np.nanmedian(a[:,boots[qi]],axis=-1),axis=0)-np.nanmax(np.nanmedian(b[:,boots[qi]],axis=-1),axis=0))
  method_comparisons[base]=dict(late_shared_minus_baseline=float(np.mean(deltas)),bootstrap_95=np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist(),shared_lower_cells=sum(x<0 for x in deltas),total_cells=16)
 payload=clean(dict(records=records,changes=changes,aggregate=aggregate,per_query=per_query,late_comparisons=method_comparisons,apostrophe_t=apostrophe,wrong_query_energy_checks=wrong,identities=identities,source_and_raw_exact_per_query=True,new_application_forwards=sum(x['forwards'] for x in identities),new_application_wall_seconds=sum(x['wall_seconds'] for x in identities),statistics=cfg['statistics'],scope=cfg['scope'],control_scope='Component-swapped map with oracle source-energy normalization. Extra source information; diagnostic, not equal-information method ranking. Single-component matched magnitude leaves sign; sum/difference test mixtures. Full KL computed before marginal-only probability storage.'))
 (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(shared=aggregate['shared16'],by_query=per_query['shared16'],late_comparisons=method_comparisons,forwards=payload['new_application_forwards'],seconds=payload['new_application_wall_seconds']),indent=2))
if __name__=='__main__':main()
