"""Bounded artifact-only pilot: family risk versus aggregate selection criteria."""
import argparse,json,os,platform,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.family_risk import relative_family_risk
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg);code=[]
 for rel in ['scripts/run_family_energy_diagnostic.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/family_risk.py','src/ccad/artifacts.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='family.energy.diagnostic.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],milestone='C1-C3',evidence_level='exposed_artifact_method_selection_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='reuse original independent mean',threshold_source_split='fixed configuration before this pilot',statistics_unit=cfg['scope'],device='cpu',seeds=[1,2,3],resource_lease='none',resource_lease_reason=cfg['budget']))
 for n in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/n).touch()
 write(run/'environment.json',dict(python=sys.executable,version=platform.python_version(),numpy=np.__version__,threads=4));write(run/'status.json',dict(status='RUNNING'));inputs=[];results=[];folds=[];error=None
 def checked(p):
  p=Path(p);p=p if p.is_absolute() else ROOT/p;inputs.append(entry(p,'Existing exposed CCAD run artifact','input','internal; no new documents/audit'));return p
 try:
  checked(a.config)
  for name in cfg['parents']:
   parent=ROOT/'runs'/name;raw=[json.loads(x) for x in checked(parent/'metrics.raw.jsonl').read_text().splitlines()];eg=json.loads(checked(parent/'error_grams.json').read_text())['records']
   with np.load(checked(parent/'coefficients.npz')) as b:
    dg=b['source_decoder']@b['source_decoder'].T;supports={m:int(np.count_nonzero(np.linalg.norm(b[m],axis=1))) for m in b.files if m!='source_decoder'}
   methods=sorted(supports);cases={r['case_id']:r for r in raw};pairs=sorted({r['pair'] for r in raw});S={i:(np.asarray(r['source_difference'])[:,None]*dg*np.asarray(r['source_difference'])[None,:])*r['common_dose']**2 for i,r in cases.items()};G={(r['case_id'],r['method']):np.asarray(r['error_gram']) for r in eg};candidates=[m for m in methods if supports[m]<=cfg['support_budget'] and m!='raw']
   assert candidates and len(pairs)>=3
   def scores(ids,m):
    sg=np.mean([S[i] for i in ids],axis=0);gg=np.mean([G[i,m] for i in ids],axis=0);risk=relative_family_risk(sg,gg,cfg['rtol']);one=np.ones(len(sg));den=float(one@sg@one)
    return dict(full_only=float(one@gg@one)/den if den>0 else None,average_parts=float(np.trace(gg)/np.trace(sg)) if np.trace(sg)>0 else None,worst_relative=risk['risk'],risk_status=risk['status'],source_gram=sg.tolist(),error_gram=gg.tolist(),worst_coefficients=risk.get('worst_unit_coefficients'))
   def actual(ids,m):
    rr=[r for r in raw if r['case_id'] in ids and r['method']==m];ops=sorted({r['operator'] for r in rr});by={op:[r for r in rr if r['operator']==op] for op in ops}
    return dict(worst_median_kl_ratio=max(float(np.median([r['normalized_kl_error'] for r in rows if r['normalized_kl_error'] is not None])) for rows in by.values()),worst_median_absolute_kl=max(float(np.median([r['candidate_kl'] for r in rows])) for rows in by.values()))
   for m in methods:
    row=dict(parent=name,method=m,support=supports[m],**scores(list(cases),m),**actual(list(cases),m));results.append(row)
    with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   for pair in pairs:
    train=[i for i,r in cases.items() if r['pair']!=pair];test=[i for i,r in cases.items() if r['pair']==pair];cal={m:scores(train,m) for m in candidates};testmetrics={m:actual(test,m) for m in methods};selection={}
    for selector in cfg['selectors']:
     order=sorted(candidates,key=lambda m:(float('inf') if cal[m][selector] is None else cal[m][selector],m));chosen=order[0];selection[selector]=dict(method=chosen,calibration_score=cal[chosen][selector],**testmetrics[chosen])
    folds.append(dict(parent=name,pair=pair,candidates=candidates,calibration=cal,selected=selection,test_all_methods=testmetrics))
  write(run/'folds.json',dict(folds=folds,scope=cfg['scope']));comparison={}
  for metric in ['worst_median_kl_ratio','worst_median_absolute_kl']:
   comparison[metric]={selector:dict(mean=float(np.mean([f['selected'][selector][metric] for f in folds])),median=float(np.median([f['selected'][selector][metric] for f in folds])),worst_beats_this=sum(f['selected']['worst_relative'][metric]<f['selected'][selector][metric] for f in folds),worst_loses_to_this=sum(f['selected']['worst_relative'][metric]>f['selected'][selector][metric] for f in folds),different_choices=sum(f['selected']['worst_relative']['method']!=f['selected'][selector]['method'] for f in folds)) for selector in cfg['selectors']}
 except Exception:
  error=traceback.format_exc();(run/'stderr.log').write_text(error);comparison={}
 status='FAIL' if error else 'PASS';write(run/'inputs.json',dict(inputs=inputs));summary=dict(status=status,error=error,cells=len(results),folds=len(folds),comparisons=comparison,wall_seconds=time.perf_counter()-start,new_lm_forwards=0,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path=code[0]['path'],generator_script_sha256=code[0]['sha256']);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat()));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));write(run/'stdout.log',summary);print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors)),indent=2));return 0 if not error and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
