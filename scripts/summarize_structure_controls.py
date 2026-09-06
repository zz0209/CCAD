"""Small CPU aggregation and source-only instance extraction for the core audit."""
import json,hashlib,os,sys
from pathlib import Path
from statistics import median
from collections import defaultdict
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/contribution_structure_20260906'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 dp=ROOT/'runs/F4_contribution_structure_diagnostic_v1_20260906';mp=ROOT/'runs/F4_contribution_matched_control_v1_20260906'
 read=lambda p:json.loads(p.read_text())
 rows=[json.loads(x) for x in (mp/'metrics.raw.jsonl').read_text().splitlines()]
 diag=[json.loads(x) for x in (dp/'metrics.raw.jsonl').read_text().splitlines()]
 src=read(dp/'source_structure.json');groups=read(mp/'source_groups.json')['groups']
 matched=[g for gs in groups.values() for g in gs if g['kind']=='matched_random']
 lookup={(r['source_seed'],r['target_seed'],r['query_id'],r['kind']):r for r in rows}
 def contrast(control,source=None):
  rr=[r for r in rows if r['kind']=='coherent' and (source is None or r['source_seed']==source)]
  diffs=[r['methods']['native64']['relative_error']-lookup[r['source_seed'],r['target_seed'],r['query_id'],control]['methods']['native64']['relative_error'] for r in rr]
  return dict(n=len(diffs),coherent_better=sum(v<0 for v in diffs),paired_median_difference=median(diffs),differences=diffs)
 balance=[]
 for g in matched:
  b=g['control_balance'];c=np.array(b['coherent']);r=np.array(b['selected_before_rescaling']);scale=b['energy_multiplier']
  balance.append(dict(query_id=g['query_id'],source_seed=g['source_seed'],coherent=c.tolist(),matched=r.tolist(),
    discovery_energy_ratio_after_scale=float(r[0]*scale**2/c[0]),effective_rank_ratio=float(r[1]/c[1]),
    max_spectral_share_error=float(np.max(np.abs(r[[2,3]]-c[[2,3]]))),
    max_raw_pc_share_error=float(np.max(np.abs(r[[4,5,6]]-c[[4,5,6]]))),
    firing_frequency_ratio=float(r[7]/c[7])))
 summary=dict(method_medians={k:{m:median(r['methods'][m]['relative_error'] for r in rows if r['kind']==k) for m in rows[0]['methods']} for k in ['coherent','random','matched_random']},
   contrasts={k:contrast(k) for k in ['random','matched_random']},per_source={str(s):{k:contrast(k,s) for k in ['random','matched_random']} for s in range(1,6)},
   raw_explained_variance=src['raw_explained_variance'],source_and_complement={},balance=balance,
   balance_summary=dict(max_energy_ratio_error=max(abs(b['discovery_energy_ratio_after_scale']-1) for b in balance),
      effective_rank_abs_error_median=median(abs(b['coherent'][1]-b['matched'][1]) for b in balance),
      rank_ratio_within_20pct=sum(.8<=b['effective_rank_ratio']<=1.2 for b in balance),
      spectra_max_error_below_05=sum(b['max_spectral_share_error']<=.05 for b in balance),
      raw_pc_max_error_below_05=sum(b['max_raw_pc_share_error']<=.05 for b in balance),n=len(balance)))
 for kind in ['coherent','random']:
  ss=[r for r in src['rows'] if r['kind']==kind and r['size']==16 and r['split']=='calibration']
  rr=[r for r in diag if r['kind']==kind and r['size']==16]
  summary['source_and_complement'][kind]=dict(effective_rank=median(r['effective_rank'] for r in ss),source_energy=median(r['energy'] for r in ss),
   top1_share=median(r['rank_one_share'] for r in ss),source_raw_pc_share={str(k):median(r['raw_pc_shares'][str(k)] for r in ss) for k in [1,4,16,64]},
   remaining_error={str(k):median(r['components'][str(k)]['complement_relative_error'] for r in rr) for k in [1,4,16,64]},
   global_field_remaining_error=median(r['global_field_complement_relative_error'] for r in rr))
 # The first source-frozen anchor is used for illustration, never selected by transfer.
 qq=[g for g in groups['1'] if g['query_id']==0]
 parent=ROOT/'runs/F4_contribution_completion_v1_20260906'
 with np.load(parent/'decoders_means.npz') as ar:dec=ar['decoder_1']
 with np.load(parent/'codes_s1_discovery.npz') as ar:z=ar['codes'].astype(float);tokens=ar['tokens']
 from transformers import AutoTokenizer
 acfg=read(ROOT/'configs/r011_nr1_k128_paired_codes_v1.json')
 tok=AutoTokenizer.from_pretrained(acfg['model_local_dir'],local_files_only=True)
 instance=dict(query_id=0,source_seed=1,anchor=1155,selection='First original source-frozen anchor, independent of target performance',groups=[])
 for g in qq:
  ids=g['source_atoms'];w=np.array(g.get('source_weights',[1]*16));x=z[:,ids]*w
  c=np.cov(x,rowvar=False,bias=True);d=dec[ids];gg=c*(d@d.T);sd=np.sqrt(np.diag(gg));corr=gg/sd[:,None]/sd[None,:]
  ce,cu=np.linalg.eigh(c);cs=(cu*np.sqrt(np.maximum(ce,0)))@cu.T;sp=np.maximum(np.linalg.eigvalsh(cs@(d@d.T)@cs),0)[::-1]
  labels=[]
  for atom in ids:
   mass=defaultdict(float)
   for token,value in zip(tokens,z[:,atom]):mass[int(token)]+=float(value)
   top=sorted(mass,key=lambda t:-mass[t])[:3]
   labels.append(dict(atom=atom,top_tokens=[tok.decode([t]).replace('\n','\\n') for t in top],firing=float(np.mean(z[:,atom]>0))))
  instance['groups'].append(dict(kind=g['kind'],members=labels,correlation=corr.tolist(),spectrum=sp.tolist(),energy=float(sp.sum()),
      effective_rank=float(sp.sum()**2/(sp@sp)),native64_errors=[lookup[1,t,0,g['kind']]['methods']['native64']['relative_error'] for t in range(2,6)]))
 cases=read(ROOT/'runs/F4_contribution_natural_v2_20260906/source_cases.json')['cases']
 instance['natural_contexts']=[{k:c[k] for k in ['recipient_text','donor_text','token','source_anchor_recipient','source_anchor_donor']} for c in cases if c['query_id']==0]
 # Normalize only shown group spectra by their own total; all member labels retained.
 payload=dict(summary=summary,instance=instance,all_rows=rows,source_structure=src['rows'],diagnostic=diag)
 (OUT/'data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 paths=[dp/'metrics.raw.jsonl',dp/'source_structure.json',mp/'metrics.raw.jsonl',mp/'source_groups.json',parent/'codes_s1_discovery.npz',parent/'decoders_means.npz',Path(__file__)]
 (OUT/'source_manifest.json').write_text(json.dumps(dict(files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],instance_rule=instance['selection'],statistics='640 dependent query-target cells;five source summaries,not independent seed directions',scope='Posthoc development; no new natural LM evaluation for matched groups'),indent=2))
 print(json.dumps({k:summary[k] for k in ['method_medians','balance_summary']},indent=2))
 print(json.dumps({'instance':[{k:g[k] for k in ['kind','energy','effective_rank','native64_errors']} for g in instance['groups']]},indent=2))
if __name__=='__main__':main()
