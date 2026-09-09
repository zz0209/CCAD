"""Describe retained candidate choices without changing them or accessing new LM data."""
from pathlib import Path
import argparse,json,hashlib,csv
from collections import defaultdict
import numpy as np

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ranks(v):
 v=np.asarray(v);out=np.zeros(len(v),float)
 for x in set(v.tolist()):out[v==x]=(np.count_nonzero(v<x)+.5*(np.count_nonzero(v==x)-1))
 return out
def spearman(x,y):
 a=ranks(x);b=ranks(y)
 return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None
def describe(run):
 cfg=json.loads((run/'config.resolved.json').read_text());choices=json.loads((run/'selection_choices.json').read_text())['choices']
 rows=[json.loads(line) for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
 qrows=defaultdict(list)
 for row in rows:qrows[row['query']].append(row)
 selectors=sorted(choices[0]['selected']);records=[];metrics={};candidate_records=[]
 for c in choices:
  for s in selectors:
   order=sorted(c['scores'][s],key=lambda n:(-c['scores'][s][n],n));actual=[c['held_summary'][n]['iia'] for n in order]
   rec=dict(run=run.name,query=c['query'],task=c['task'],objective=c['objective'],source_seed=c['source_seed'],target_seed=c['target_seed'],selector=s,selected=order[0],top1_iia=actual[0],top3_uniform_iia=float(np.mean(actual[:3])),ranking_spearman=spearman([c['scores'][s][n] for n in order],actual),source_iia=c['source_held_iia'],oracle_iia=c['oracle_native_held_iia'],regret=c['oracle_native_held_iia']-actual[0],predicted_confidence=c['scores'][s][order[0]])
   records.append(rec)
  for name,cs in c['candidate_stats'].items():
   candidate_records.append(dict(query=c['query'],task=c['task'],objective=c['objective'],method=name,iia=c['held_summary'][name]['iia'],**{k:cs[k] for k in ['members','l1','relative_realization_l2','natural_mse']}))
 for s in selectors:
  rr=[r for r in records if r['selector']==s];metrics[s]={key:float(np.mean([r[key] for r in rr if r[key] is not None])) for key in ['top1_iia','top3_uniform_iia','ranking_spearman','regret'] if any(r[key] is not None for r in rr)}
  if s in ['base_linear','anchored_base_linear','source_endpoint','source_path']:
   pred=[];observed=[];margins=[]
   for r in rows:
    if s in r.get('predictions',{}):pred.append(r['predictions'][s]);observed.append(float(r['iia']));margins.append(r['margin'])
   p=1/(1+np.exp(-np.clip(pred,-80,80)));y=np.asarray(observed);m=np.asarray(margins)
   bins=[]
   for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
    ix=(p>=lo)&(p<hi if hi<1 else p<=hi)
    if ix.any():bins.append(dict(lower=float(lo),upper=float(hi),rows=int(ix.sum()),mean_score=float(p[ix].mean()),actual_iia=float(y[ix].mean())))
   metrics[s].update(margin_mae=float(np.mean(np.abs(np.asarray(pred)-m))),binary_brier=float(np.mean((p-y)**2)),calibration_bins=bins,calibration_scope='Sigmoid of a predicted donor-minus-base logit margin is a two-answer preference score. Reliability is empirical, not an assumed calibrated success probability.')
   metrics[s]['refusal_curve']=[dict(threshold=float(t),query_coverage=float(np.mean([r['predicted_confidence']>=t for r in rr])),selected_iia=float(np.mean([r['top1_iia'] for r in rr if r['predicted_confidence']>=t])) if any(r['predicted_confidence']>=t for r in rr) else None) for t in [0,.25,.5,.6,.7,.8,.9]]
 return dict(run=run.name,config=cfg,queries=len(choices),metrics=metrics,records=records,candidate_records=candidate_records,source_iia=float(np.mean([c['source_held_iia'] for c in choices])),raw_iia=float(np.mean([c['held_summary']['raw']['iia'] for c in choices])),full_target_iia=float(np.mean([c['held_summary']['full_target']['iia'] for c in choices])),source_files=[dict(path=str(run/f),sha256=digest(run/f)) for f in ['config.resolved.json','selection_choices.json','metrics.raw.jsonl','metrics.summary.json']])
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 result=dict(runs=[describe(Path(r)) for r in a.runs],scope='Descriptive query-macro summaries. Source and target seeds, shared tasks and prompts are dependent; these rows are not independent population replicates. Held-dev is development, even though each saved choice precedes held candidate execution.')
 (a.output/'selection_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 records=[r for run in result['runs'] for r in run['records']]
 with (a.output/'selector_by_query.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 for run in result['runs']:print(json.dumps(dict(run=run['run'],queries=run['queries'],source=run['source_iia'],raw=run['raw_iia'],selectors={s:round(m['top1_iia'],5) for s,m in run['metrics'].items()})))
if __name__=='__main__':main()
