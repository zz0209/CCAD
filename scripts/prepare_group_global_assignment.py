"""Complete dictionary PW-MCC/Hungarian baselines from retained natural codes."""
from __future__ import annotations
import os,sys,json,time,argparse,platform,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);cfg=json.loads(ap.parse_args().config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
 started=datetime.now(timezone.utc).isoformat();start=time.perf_counter();cpu=time.process_time();write(run/'config.resolved.json',cfg)
 codes=[]
 for rel in ['scripts/prepare_group_global_assignment.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());codes.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='native.group.global.assignment.v1',run_id=cfg['run_id'],run_parent=cfg['round_id'],purpose=cfg['purpose'],milestone='M4',evidence_level='natural_baseline_fit',started_utc=started,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(codes),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='independent natural mean',threshold_source_split='no threshold; every source/target atom participates',statistics_unit='two complete dictionaries; no population CI',device='cpu',seeds=cfg['seeds'],resource_lease='cpu-heavy resource_manager.run',resource_lease_reason='Dense full-dictionary correlation matrix and exact Hungarian assignment'))
 for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/fn).touch()
 write(run/'status.json',dict(status='RUNNING',updated_utc=started));inputs=[];rows=[];error=None;env={};checks={}
 def checked(p):
  p=Path(p);p=p if p.is_absolute() else ROOT/p;inputs.append(entry(p,'Retained natural SAE code cache','actual input','internal'));return p
 def log(event,**kw):
  r=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),event=event,elapsed_seconds=time.perf_counter()-start,**kw);write(run/'progress.json',r)
  with (run/'stdout.log').open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps(r),flush=True)
 try:
  sys.path.append(cfg['scipy_overlay_dir']);import numpy as np,scipy
  from scipy.sparse import csr_matrix
  from scipy.optimize import linear_sum_assignment
  env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threads=4)
  ref=ROOT/cfg['reference_run'];rcfg=json.loads(checked(ref/'config.resolved.json').read_text());assert json.loads(checked(ref/'status.json').read_text())['status']=='PASS';write(run/'reference_identity.json',dict(reference_run=ref.name,training_run=rcfg['training_run'],checkpoint_step=rcfg['checkpoint_step'],natural_tokens=rcfg['natural_tokens'],reference_config_sha256=sha256(ref/'config.resolved.json')))
  cache={};means={}
  for o in cfg['objectives']:
   for seed in cfg['seeds']:
    for split in ['mean','discovery']:
     p=checked(ref/f'natural_{split}_{o}_seed{seed}.npz');a=np.load(p);Z=csr_matrix((a['values'].astype(np.float64),(a['rows'],a['columns'])),shape=tuple(a['shape']))
     if split=='mean':means[o,seed]=np.asarray(Z.mean(0)).ravel()
     else:cache[o,seed]=Z
  for o in cfg['objectives']:
   for source,target in cfg['seed_pairs']:
    t=time.perf_counter();S=cache[o,source];T=cache[o,target];N=S.shape[0];assert S.shape==T.shape
    sm=np.asarray(S.mean(0)).ravel();tm=np.asarray(T.mean(0)).ravel();sv=np.asarray(S.power(2).mean(0)).ravel()-sm**2;tv=np.asarray(T.power(2).mean(0)).ravel()-tm**2
    cov=(S.T@T/N).toarray();cov-=sm[:,None]*tm[None,:];den=np.sqrt(np.maximum(sv,0)[:,None]*np.maximum(tv,0)[None,:]);corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-14)
    del cov,den
    log('HUNGARIAN_START',objective=o,source=source,target=target,width=len(corr),matrix_bytes=corr.nbytes)
    ii,jj=linear_sum_assignment(-np.abs(corr));assert np.array_equal(ii,np.arange(len(corr)));values=corr[ii,jj];gain=values*np.sqrt(np.maximum(sv,0))/np.sqrt(np.maximum(tv[jj],0)).clip(1e-12)
    intercept=means[o,source]-gain*means[o,target][jj]
    p=run/f'{o}_s{source}_t{target}_global_pw.npz';np.savez_compressed(p,permutation=jj,correlation=values,gain=gain,source_mean=means[o,source],target_mean=means[o,target],intercept=intercept,source_variance=sv,target_variance=tv,source_positive_counts=np.asarray((S>0).sum(0)).ravel(),target_positive_counts=np.asarray((T>0).sum(0)).ravel())
    row=dict(objective=o,source_seed=source,target_seed=target,width=len(values),pw_mcc=float(np.abs(values).mean()),negative_matches=int((values<0).sum()),zero_variance_source=int((sv<=1e-14).sum()),zero_variance_target=int((tv<=1e-14).sum()),wall_seconds=time.perf_counter()-t,path=p.name,sha256=sha256(p))
    rows.append(row)
    with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    log('ASSIGNMENT_COMPLETE',**row);del corr
    if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Natural assignment budget guard; completed maps retained')
  checks=dict(complete_assignments=len(rows),all_rows_and_columns_used_once=True,test_read=False,task_labels_used=False)
 except BaseException as exc:error=repr(exc);(run/'stderr.log').write_text(traceback.format_exc());log('FAIL',error=error)
 status='PASS' if error is None else 'FAIL';summary=dict(status=status,error=error,rows=rows,checks=checks,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/prepare_group_global_assignment.py',generator_script_sha256=codes[0]['sha256'],scope=cfg['scope'])
 write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),error=error));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=v.errors));log('COMPLETE',status=status,contract_ok=v.ok,errors=v.errors)
 return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
