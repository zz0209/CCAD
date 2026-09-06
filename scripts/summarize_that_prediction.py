"""All-document input/output prediction and frozen cross-seed transfer report."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/that_prediction_20260906'
def load(p):return json.loads(p.read_text())
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 native=ROOT/'runs/NATIVE_that_prediction_s1a2379_v1_20260906';ns=load(native/'metrics.summary.json');nr=[json.loads(s) for s in (native/'metrics.raw.jsonl').read_text().splitlines()];ids=load(native/'contrast_tokens.json');groups={};details=[]
 for name in ['attitude','report','noun_control']:
  rr=[r for r in nr if r['subject']==name];a=[r['activation'] for r in rr];ad=[r['effects']['native_add']['contrast_delta'] for r in rr];rm=[r['effects']['native_remove']['contrast_delta'] for r in rr]
  groups[name]=dict(n=len(rr),activation_median=st.median(a),activation_min=min(a),activation_max=max(a),add_positive=sum(v>0 for v in ad),remove_negative=sum(v<0 for v in rm),joint=sum(x>0 and y<0 for x,y in zip(ad,rm)),zero_activation=sum(v==0 for v in a),add_median=st.median(ad),remove_median=st.median(rm))
  details.extend(dict(family=name,text=r['text'],document_id=r['document_id'],activation=r['activation'],add=r['effects']['native_add']['contrast_delta'],remove=r['effects']['native_remove']['contrast_delta']) for r in rr)
 cells=[];runs=[];percase=[];source=None
 for seed in [2,3,4,5]:
  run=ROOT/'runs'/f'F4_that_prediction_apply_s{seed}_v1_20260906';summ=load(run/'metrics.summary.json');assert summ['status']=='PASS';cfg=load(run/'config.resolved.json');parent=ROOT/cfg['frozen_fit']['path'];p=np.load(run/'probabilities.npz');rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
  with np.load(run/'coefficients.npz') as a,np.load(parent/'coefficients.npz') as b:
   assert all(np.array_equal(a[k],b[k]) for k in a.files);support=int(np.count_nonzero(a['sparse16']))
  ref=[{k:r[k] for k in ['case_id','source_activation','source_difference','source_kl','candidate_kl','dose']} for r in rows if r['method']=='raw']
  if source is None:source=ref
  else:assert source==ref
  def contrast(key):
   x=p[key].astype(float);return float(np.log(x[ids['first']].sum()/x[ids['third']].sum()))
  for r in rows:
   i=r['case_id'];base=contrast(f'base_{i}');actual=contrast(f'source_{i}')-base;pred=contrast(f"{r['method']}_{i}")-base
   percase.append(dict(seed=seed,case_id=i,pair=r['pair'],family=r['family'],method=r['method'],source_response=actual,candidate_response=pred,absolute_contrast_error=abs(pred-actual),source_difference=r['source_difference'],predicted_difference=r['predicted_difference'],expected_sign=r['expected_donor_sign'],source_prediction_matches=None if r['expected_donor_sign'] is None else actual*r['expected_donor_sign']>0,candidate_prediction_matches=None if r['expected_donor_sign'] is None else pred*r['expected_donor_sign']>0))
  for family in ['attitude_report','noun_challenge','all']:
   cell=dict(seed=seed,family=family,sparse_members=support,methods={})
   for m in ['best_atom','geometric_atom','sparse16','full','raw']:
    rr=[r for r in rows if r['method']==m and (family=='all' or r['family']==family)];tt=[r for r in percase if r['seed']==seed and r['method']==m and (family=='all' or r['family']==family)]
    errors=[r['absolute_contrast_error'] for r in tt];v=[r['normalized_kl_error'] for r in rr if r['normalized_kl_error'] is not None]
    cell['methods'][m]=dict(n=len(rr),median_absolute_contrast_error=st.median(errors),mean_absolute_contrast_error=st.mean(errors),median_kl_ratio=st.median(v) if v else None,median_absolute_kl=st.median(r['candidate_kl'] for r in rr),source_response_matches=sum(r['source_response']*r['candidate_response']>0 for r in tt),lexical_prediction_matches=sum(r['candidate_prediction_matches'] is True for r in tt),lexical_denominator=sum(r['expected_sign'] is not None for r in tt),weak_source=sum(r['source_kl']<1e-5 for r in rr))
   a={r['case_id']:r for r in percase if r['seed']==seed and r['method']=='sparse16' and (family=='all' or r['family']==family)}
   cell['sparse_contrast_wins']={m:sum(a[r['case_id']]['absolute_contrast_error']<r['absolute_contrast_error'] for r in percase if r['seed']==seed and r['method']==m and (family=='all' or r['family']==family)) for m in ['best_atom','geometric_atom','full','raw']}
   cells.append(cell)
  runs.append(dict(seed=seed,run=str(run),forwards=summ['forwards'],wall_seconds=summ['wall_seconds'],metrics_sha256=H(run/'metrics.raw.jsonl')))
 allsource=[r for r in percase if r['seed']==2 and r['method']=='raw' and r['family']=='attitude_report']
 result=dict(native=groups,native_details=details,native_cost=ns,cells=cells,runs=runs,percase=percase,source_donor_lexical_matches=sum(r['source_prediction_matches'] for r in allsource),source_donor_lexical_denominator=len(allsource),checks=dict(all_coefficients_exact=True,source_raw_exact_across_targets=True),scope='One source/four dependent targets,12 reciprocal document pairs. Source native add/remove and source-aligned donor distinct. Output contrast frozen before newtext, maps predate newtext; target application planned after source result without changing inputset or fitting. No semantic uniqueness/native target claim.')
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# That source: natural predictions and frozen functional correspondence','',result['scope'],'','Contrast = log(P(I,we,you,he,she,it,they)/P(there)), fixed before new corpus. These are token sets, not guaranteed semantic roles. All24documents retained.','', '| Family | Activation median [range] | Add positive | Remove negative | Joint |','|---|---:|---:|---:|---:|']
 for k,g in groups.items():lines.append(f"| {k} | {g['activation_median']:.4f} [{g['activation_min']:.4f},{g['activation_max']:.4f}] | {g['add_positive']}/{g['n']} | {g['remove_negative']}/{g['n']} | {g['joint']}/{g['n']} |")
 lines+=['','Fresh source-native results support a lexical response tendency with explicit counterexamples. Noun controls remain active and have similar direction: not attitude-exclusive or pure syntax. Native source prediction was developed posthoc on12old authored examples, then frozen for fresh documents.','', '| Target | Family | Members | Sparse contrast MAE | Best atom | Full | Raw | Sparse wins vs atom |','|---|---|---:|---:|---:|---:|---:|---:|']
 for c in cells:
  v=c['methods'];lines.append(f"| {c['seed']} | {c['family']} | {c['sparse_members']} | {v['sparse16']['mean_absolute_contrast_error']:.6f} | {v['best_atom']['mean_absolute_contrast_error']:.6f} | {v['full']['mean_absolute_contrast_error']:.6f} | {v['raw']['mean_absolute_contrast_error']:.6f} | {c['sparse_contrast_wins']['best_atom']}/{v['sparse16']['n']} |")
 lines+=['',f"Source donor lexical sign prediction: {result['source_donor_lexical_matches']}/{len(allsource)}. Whole-distribution KL, both families, all5methods, weak sources and every contrast are in summary.json. Counts are descriptive; reciprocal donors and sharedsource are dependent.",'','## All natural source contexts','']
 for r in details:lines.append(f"- {r['family']} | act {r['activation']:.5f}; add {r['add']:+.5f}; remove {r['remove']:+.5f} | {r['text'].replace(chr(10),' ')}")
 (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(native=groups,cells=cells,donor_signs=result['source_donor_lexical_matches'],forwards=sum(r['forwards'] for r in runs),seconds=sum(r['wall_seconds'] for r in runs)),indent=2))
if __name__=='__main__':main()
