"""Equal-energy direction controls with exact replay and all-context accounting."""
from pathlib import Path
import json,hashlib,statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 run=ROOT/'runs/NATIVE_that_controls_s1a2379_v1_20260906';parent=ROOT/'runs/NATIVE_that_prediction_s1a2379_v1_20260906';out=ROOT/'artifacts/that_controls_20260906';rows=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()];summary=json.loads((run/'metrics.summary.json').read_text());metadata=json.loads((run/'control_metadata.json').read_text())
 assert json.loads((run/'status.json').read_text())['status']=='PASS'
 with np.load(parent/'probabilities.npz') as old,np.load(run/'probabilities.npz') as new:
  assert all(np.array_equal(old[k],new[k]) for k in old.files)
 maxerr=max(abs(v-r['source_delta_norm']) for r in rows for v in r['control_delta_norms'].values());assert maxerr<1e-5
 detail=[];cells=[]
 for i,r in enumerate(rows):
  for name in ['native','atom_988','random_0','random_1']:
   add=r['effects'][name+'_add'];remove=r['effects'][name+'_remove'];detail.append(dict(case_id=i,family=r['subject'],text=r['text'],direction=name,source_activation=r['activation'],alternative_activation=r['control_activations']['atom_988'],delta_norm=r['source_delta_norm'],add=add['contrast_delta'],remove=remove['contrast_delta'],signed_slope=(add['contrast_delta']-remove['contrast_delta'])/2,even_response=(add['contrast_delta']+remove['contrast_delta'])/2,joint=add['contrast_delta']>0 and remove['contrast_delta']<0,mean_kl=(add['kl_to_base']+remove['kl_to_base'])/2))
 for family in ['attitude','report','noun_control','all']:
  for direction in ['native','atom_988','random_0','random_1']:
   rr=[r for r in detail if r['direction']==direction and (family=='all' or r['family']==family)];slopes=[r['signed_slope'] for r in rr]
   cells.append(dict(family=family,direction=direction,n=len(rr),joint=sum(r['joint'] for r in rr),joint_opposite=sum(r['add']<0 and r['remove']>0 for r in rr),positive_slope=sum(v>0 for v in slopes),median_signed_slope=st.median(slopes),mean_signed_slope=st.mean(slopes),median_absolute_slope=st.median(abs(v) for v in slopes),mean_absolute_slope=st.mean(abs(v) for v in slopes),median_mean_kl=st.median(r['mean_kl'] for r in rr),mean_mean_kl=st.mean(r['mean_kl'] for r in rr)))
 source={r['case_id']:r for r in detail if r['direction']=='native'}
 comparisons={name:dict(source_larger_signed=sum(source[r['case_id']]['signed_slope']>r['signed_slope'] for r in detail if r['direction']==name),source_larger_absolute=sum(abs(source[r['case_id']]['signed_slope'])>abs(r['signed_slope']) for r in detail if r['direction']==name),source_larger_kl=sum(source[r['case_id']]['mean_kl']>r['mean_kl'] for r in detail if r['direction']==name)) for name in ['atom_988','random_0','random_1']}
 result=dict(cells=cells,detail=detail,comparisons=comparisons,control_metadata=metadata,source_replay_exact=True,max_control_norm_absolute_error=maxerr,execution=summary,scope='24exposed natural documents, one source, same-token alternative and2fixed isotropic random directions. Equal hook norm, not equal output KL. Alternative988 is decoder perturbation at source2379dose, not its native deletion. Signed +/- central response per case, no small-dose linearity assumption. Descriptive, not a random-direction population test.',inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [Path(__file__),run/'metrics.raw.jsonl',run/'control_directions.npz',parent/'probabilities.npz']])
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Direction-specificity development controls','',result['scope'],'','| Family | Direction | Joint expected signs | Signed response median | Absolute response median | Mean +/- KL median |','|---|---|---:|---:|---:|---:|']
 for r in cells:lines.append(f"| {r['family']} | {r['direction']} | {r['joint']}/{r['n']} | {r['median_signed_slope']:.6f} | {r['median_absolute_slope']:.6f} | {r['median_mean_kl']:.6f} |")
 lines+=['','Response = (add contrast - remove contrast)/2 at the same capped source norm; not divided by dose and not claimed infinitesimal derivative. Absolute response prevents arbitrary random orientation from hiding a large opposite effect.','',f"All96old baseline/noop/native probability arrays replay exactly. Maximum norm difference {maxerr:.9g}. Decoder cosine source/988={metadata['cosines']['source']['atom_988']:.6f}; full direction identities andcosines stored.",'','| Control | Source larger signed | Source larger absolute | Source larger KL |','|---|---:|---:|---:|']
 for name,c in comparisons.items():lines.append(f"| {name} | {c['source_larger_signed']}/24 | {c['source_larger_absolute']}/24 | {c['source_larger_kl']}/24 |")
 (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(cells=cells,comparisons=comparisons,cosines=metadata['cosines'],normerror=maxerr,cost=summary['wall_seconds']),indent=2))
if __name__=='__main__':main()
