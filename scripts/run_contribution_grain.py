"""Source-defined coarsening versus arbitrary averaging in native contributions."""
import argparse,json,hashlib,platform,time,traceback,sys
from datetime import datetime,timezone
from pathlib import Path
from run_contribution_completion import ROOT,np,entry,aggregate,write,sha256,validate_run_directory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);cfg=json.loads(ap.parse_args().config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg);files=[]
 for rel in ['scripts/run_contribution_grain.py','scripts/run_contribution_completion.py','scripts/run_f4_source_reference_causal.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']+(['scripts/contribution_group_controls.py'] if cfg.get('matched_control') else []):
  p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());files.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=files,aggregate_sha256=aggregate(files),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='contribution.grain.v1',run_id=cfg['run_id'],run_parent=cfg['parent_run'],purpose=cfg['purpose'],milestone='C1-C2-C3',evidence_level='controlled_five_seed_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(files),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='inherited independent means; empirical covariance for iid donor loss',threshold_source_split='source-only contribution neighborhoods before cross-seed fit',statistics_unit='overlapping source-anchor families, five shared seeds and common paired documents',device='cuda:0',seeds=[1,2,3,4,5],resource_lease=cfg.get('resource_lease','cpu-heavy -> gpu-0 resource_manager.run'),resource_lease_reason=cfg['budget']))
 for f in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/f).touch()
 write(run/'status.json',dict(status='RUNNING'));inputs=[];rows=[];checks={};error=None;env={}
 def checked(path):
  p=Path(path);p=p if p.is_absolute() else ROOT/p;inputs.append(entry(p,'Existing controlled CCAD parent','input'));return p
 def load(path):return json.loads(checked(path).read_text())
 def progress(stage,**kw):
  v=dict(stage=stage,seconds=time.perf_counter()-start,**kw);write(run/'progress.json',v);print(json.dumps(v),flush=True)
  if v['seconds']>cfg['budget_seconds']:raise TimeoutError('Unit budget exceeded')
 try:
  import torch
  torch.set_num_threads(cfg.get('cpu_threads',4));torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats();parent=ROOT/'runs'/cfg['parent_run'];assert load(parent/'status.json')['status']=='PASS';load(parent/'config.resolved.json');sel=load(parent/'source_queries.json');queries=sel['queries'];zc={};dd={};gs={};gx={};group={};masks={};freq={};n=None
  def tensor(x):return torch.tensor(x,dtype=torch.float64,device='cuda:0')
  with np.load(checked(parent/'decoders_means.npz')) as dec:
   for s in range(1,6):dd[s]=tensor(dec[f'decoder_{s}'])
  if cfg.get('matched_control'):
   from contribution_group_controls import matched_group
   with np.load(checked(cfg['matched_control']['pca_path'])) as ar:components=tensor(ar['components'])
  for s in range(1,6):
   for split in ['discovery','calibration']:
    with np.load(checked(parent/f'codes_s{s}_{split}.npz')) as ar:z=ar['codes'].astype(float)
    if split=='discovery':freq[s]=tensor((z>0).mean(0))
    n=len(z);zc[s,split]=tensor(z-z.mean(0));gx[s,split]=zc[s,split].T@zc[s,split]/n;gs[s,split]=gx[s,split]*(dd[s]@dd[s].T)
   g=gs[s,'discovery'];diag=torch.sqrt(torch.diag(g));corr=(g/torch.clamp(diag[:,None]*diag[None,:],min=1e-30)).cpu().numpy();eligible=next(x['eligible'] for x in sel['selection'] if x['seed']==s);qq=[]
   for qi,q in enumerate(queries):
    if q['source_seed']!=s:continue
    anchor=q['source_atom'];others=[a for a in eligible if a!=anchor];near=[anchor]+sorted(others,key=lambda a:(-corr[anchor,a],a));random=[anchor]+sorted(others,key=lambda a:hashlib.sha256(f"{cfg['selection_salt']}:{s}:{anchor}:{a}".encode()).hexdigest())
    for size in cfg['group_sizes']:
     for kind,order in [('coherent',near)]+([('random',random)] if size>1 else []):qq.append(dict(query_id=qi,source_seed=s,anchor=anchor,quartile=q['quartile'],size=size,kind=kind,source_atoms=order[:size]))
     if cfg.get('matched_control'):
      match=matched_group(gx[s,'discovery'],dd[s],freq[s],components,eligible,near[:size],s,qi,cfg['matched_control'],run)
      qq.append(dict(query_id=qi,source_seed=s,anchor=anchor,quartile=q['quartile'],size=size,kind='matched_random',**match))
   group[s]=qq;mask=np.zeros((len(dd[s]),len(qq)))
   for j,q in enumerate(qq):mask[q['source_atoms'],j]=q.get('source_weights',1)
   masks[s]=tensor(mask)
   progress('source_groups_ready',source=s,groups=len(qq))
  expected_groups=cfg.get('expected_source_groups',1120)
  write(run/'source_groups.json',dict(groups=group,rule=cfg['group_rule'],frozen_before_cross_seed_fit=True));checks['all_source_groups']=sum(map(len,group.values()))==expected_groups;assert checks['all_source_groups'];progress('source_groups_frozen',groups=expected_groups)
  for target in range(1,6):
   gn=gs[target,'discovery'];gnc=gs[target,'calibration'];active=torch.where(torch.diag(gn)>1e-12)[0];ca=torch.where(torch.diag(gnc)>1e-12)[0];scale=torch.sqrt(torch.diag(gn)[active]);cs=torch.sqrt(torch.diag(gnc)[ca]);norm=gn[active][:,active]/scale[:,None]/scale[None,:];cnorm=gnc[ca][:,ca]/cs[:,None]/cs[None,:];chol=torch.linalg.cholesky(norm);cchol=torch.linalg.cholesky(cnorm);rch=torch.linalg.cholesky(norm+cfg['ridge_fraction']*torch.eye(len(active),dtype=torch.float64,device='cuda:0'))
   for source in range(1,6):
    if target==source:continue
    h0=(zc[target,'discovery'].T@zc[source,'discovery']/n)*(dd[target]@dd[source].T);hc0=(zc[target,'calibration'].T@zc[source,'calibration']/n)*(dd[target]@dd[source].T);mask=masks[source];h=h0@mask;hc=hc0@mask;sv=torch.sum(mask*(gs[source,'discovery']@mask),axis=0);cv=torch.sum(mask*(gs[source,'calibration']@mask),axis=0);assert bool(torch.all(cv>0));nt=len(dd[target]);ng=len(group[source]);nat=torch.zeros((nt,ng),dtype=torch.float64,device='cuda:0');nr=nat.clone();oracle=nat.clone();nat[active]=torch.cholesky_solve(h[active]/scale[:,None],chol)/scale[:,None];nr[active]=torch.cholesky_solve(h[active]/scale[:,None],rch)/scale[:,None];oracle[ca]=torch.cholesky_solve(hc[ca]/cs[:,None],cchol)/cs[:,None]
    bank=dict(native_full=nat,native_ridge_full=nr,calibration_native_oracle=oracle);native64=nat*0;matching=nat*0;random64=nat*0;marginal64=nat*0;diagonal=torch.diag(gn);allactive=active.cpu().numpy();individual=torch.where(diagonal[:,None]>1e-12,h0*h0/torch.clamp(diagonal[:,None],min=1e-30),-torch.inf);assignment=torch.argmax(individual,axis=0)
    for j,q in enumerate(group[source]):
     orders=dict(native64=torch.argsort(torch.abs(nr[:,j])*torch.sqrt(torch.clamp(diagonal,min=0)),descending=True,stable=True)[:64],marginal64=torch.argsort(h[:,j]**2/torch.clamp(diagonal,min=1e-30),descending=True,stable=True)[:64],matching_refit=torch.unique(assignment[q['source_atoms']]),random64=torch.tensor(np.random.default_rng(q['query_id']*10000+q['size']*100+target).choice(allactive,64,replace=False),device='cuda:0'))
     for key,b in [('native64',native64),('marginal64',marginal64),('matching_refit',matching),('random64',random64)]:
      ids=orders[key];g=gn[ids][:,ids];sc=torch.sqrt(torch.diag(g));b[ids,j]=torch.linalg.solve(g/sc[:,None]/sc[None,:]+cfg['ridge_fraction']*torch.eye(len(ids),dtype=torch.float64,device='cuda:0'),h[ids,j]/sc)/sc
    bank.update(native64=native64,marginal64=marginal64,matching_refit=matching,random64=random64)
    np.savez_compressed(run/f'maps_s{source}_t{target}.npz',**{k:v.cpu().numpy() for k,v in bank.items()})
    result={}
    for name,b in bank.items():
     ev=cv-2*torch.sum(b*hc,axis=0)+torch.sum(b*(gnc@b),axis=0);assert float(ev.min())>-1e-7;result[name]=dict(error=torch.clamp(ev,min=0).cpu().numpy(),support=torch.count_nonzero(b,dim=0).cpu().numpy())
    for j,q in enumerate(group[source]):
     values={name:dict(relative_error=float(v['error'][j])/float(cv[j]),absolute_error=float(v['error'][j]),support=int(v['support'][j])) for name,v in result.items()};row=dict(**q,target_seed=target,source_calibration_energy=float(cv[j]),methods=values);rows.append(row)
     with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    # Fixed source basis queries at size1 must reproduce the first unit.
    original=[json.loads(s) for s in (parent/'metrics.raw.jsonl').read_text().splitlines()];lookup={r['query_id']:r for r in original if r['source_seed']==source and r['target_seed']==target}
    for r in rows[-ng:]:
     if r['size']==1:
      for method in ['native_full','native_ridge_full','calibration_native_oracle','native64']:assert np.isclose(r['methods'][method]['relative_error'],lookup[r['query_id']]['methods'][method]['relative_error'],rtol=1e-7,atol=1e-8),(source,target,r['anchor'],method)
    progress('pair_complete',source=source,target=target,rows=len(rows))
  checks.update(complete_panel=len(rows)==expected_groups*4,zero_new_lm_forward=True)
  if 1 in cfg['group_sizes']:checks['singletons_reproduce_parent']=True
  env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cpu_threads=torch.get_num_threads(),gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated(),dtype='float64')
 except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
 status='PASS' if error is None and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(rows),lm_forwards=0,wall_seconds=time.perf_counter()-start,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_contribution_grain.py',generator_script_sha256=files[0]['sha256']);write(run/'environment.json',env);write(run/'inputs.json',dict(inputs=inputs));write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True)
 return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
