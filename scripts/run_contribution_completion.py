"""Compare readout recovery with actual target-decoder contribution completion.

All source queries are selected before cross-seed fitting. Native here means
signed target-decoder scaling, NOT the unweighted MSCC endpoint. Calibration
oracle fits are explicit empirical lower bounds, never deployable estimators.
"""
import argparse,json,os,platform,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_f4_source_reference_causal import ROOT,np
from run_r011s1_raw_hook_asset import entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);cfg=json.loads(ap.parse_args().config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg)
 files=[]
 for rel in ['scripts/run_contribution_completion.py','scripts/run_f4_source_reference_causal.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
  p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());files.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=files,aggregate_sha256=aggregate(files),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='contribution.completion.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],milestone='C1-C2-C3',evidence_level='controlled_five_seed_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(files),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='independent mean split; empirical covariance fits iid donor differences so intercept cancels',threshold_source_split='source-only discovery selection, no target query selection',statistics_unit='source/query families sharing five seeds and paired documents',device='cuda:0',seeds=[1,2,3,4,5],resource_lease='cpu-heavy -> gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
 for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
 write(run/'status.json',dict(status='RUNNING'));inputs=[];env={};rows=[];checks={};error=None
 def checked(path,expected=None):
  p=Path(path);p=p if p.is_absolute() else ROOT/p;r=entry(p,'Existing CCAD paired assets / locked sparsify weights','input')
  if expected and r['sha256']!=expected:raise ValueError('Input changed: '+str(p))
  inputs.append(r);return p
 def load(path):return json.loads(checked(path).read_text(encoding='utf-8-sig'))
 def progress(stage,**kw):
  v=dict(stage=stage,seconds=time.perf_counter()-start,**kw);write(run/'progress.json',v);print(json.dumps(v),flush=True)
  if v['seconds']>cfg['budget_seconds']:raise TimeoutError('Fixed unit compute budget exceeded')
 try:
  import torch,transformers
  from sparsify import SparseCoder
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
  asset=load(cfg['asset_config']);rm=load(cfg['raw_manifest']);tm=load(asset['token_manifest_path']);load('.aris/compute/local-r006b1-env-spec.json')
  paired=[json.loads(s) for s in checked(ROOT/'runs'/asset['paired_corpus_run']/'artifacts/documents.jsonl').read_text().splitlines() if s];train=load(cfg['training_documents'])['documents']
  checks['training_disjoint']=all(not({x[k] for x in paired}&{x[k] for x in train}) for k in ['document_id','text_sha256']);assert checks['training_disjoint']
  raw={};indices={};token={}
  for split in ['mean','discovery','calibration']:
   spec=next(x for x in rm['splits'] if x['split']==split);raw[split]=np.memmap(checked(spec['path'],spec['sha256']),dtype='<f4',mode='r',shape=tuple(spec['shape']))
   if split!='mean':
    indices[split]=np.linspace(0,len(raw[split])-1,cfg['positions'],dtype=int);ts=tm['outputs'][split];token[split]=np.memmap(checked(ROOT/'runs'/asset['paired_corpus_run']/ts['path'],ts['sha256']),dtype='<u2',mode='r')[indices[split]]
  means={};codes={};dec={};query=[];selection=[]
  for cp in sorted(cfg['checkpoints'],key=lambda x:x['seed']):
   seed=cp['seed'];weight=checked(Path(cp['path'])/'sae.safetensors',cp['sha256']);sae=SparseCoder.load_from_disk(weight.parent,device='cuda:0').eval();dec[seed]=sae.W_dec.detach().cpu().numpy().astype(float)
   def encode(h):
    a=np.zeros((len(h),len(dec[seed])),dtype=np.float64)
    with torch.no_grad():
     for j in range(0,len(h),512):
      e=sae.encode(torch.tensor(np.array(h[j:j+512]),device='cuda:0',dtype=torch.float32));ii=e.top_indices.cpu().numpy();vv=e.top_acts.cpu().numpy();np.add.at(a,(np.arange(j,j+len(ii))[:,None],ii),vv)
    return a
   total=np.zeros(len(dec[seed]))
   for j in range(0,len(raw['mean']),512):total+=encode(raw['mean'][j:j+512]).sum(0)
   means[seed]=total/len(raw['mean'])
   for split in ['discovery','calibration']:
    codes[seed,split]=encode(raw[split][indices[split]])
    np.savez_compressed(run/f'codes_s{seed}_{split}.npz',codes=codes[seed,split].astype(np.float32),row_indices=indices[split],tokens=token[split])
   z=codes[seed,'discovery'];hits=np.count_nonzero(z,axis=0);var=np.var(z,axis=0)*np.sum(dec[seed]**2,axis=1);eligible=np.flatnonzero((hits>=cfg['minimum_hits'])&(hits<=cfg['maximum_frequency']*len(z))&(var>1e-12));rank=eligible[np.argsort(var[eligible],kind='stable')];chosen=[]
   for quartile,ids in enumerate(np.array_split(rank,4)):
    order=sorted(ids.tolist(),key=lambda a:__import__('hashlib').sha256(f"{cfg['selection_salt']}:{seed}:{a}".encode()).hexdigest());take=order[:cfg['queries_per_seed']//4];chosen.extend(take)
    for atom in take:query.append(dict(source_seed=seed,source_atom=atom,quartile=quartile,discovery_hits=int(hits[atom]),discovery_energy=float(var[atom]),top_discovery_rows=np.argsort(-z[:,atom],kind='stable')[:16].tolist()))
   assert len(chosen)==cfg['queries_per_seed'];selection.append(dict(seed=seed,eligible=eligible.tolist(),selected=chosen));del sae;progress('source_encoded_and_selected',seed=seed,eligible=len(eligible))
  write(run/'source_queries.json',dict(queries=query,selection=selection,rule=cfg['source_selection'],selected_before_cross_seed_fit=True));np.savez_compressed(run/'decoders_means.npz',**{f'decoder_{s}':v for s,v in dec.items()},**{f'mean_{s}':v for s,v in means.items()})
  checks['source_only_queries_frozen']=len(query)==5*cfg['queries_per_seed']
  def tensor(x):return torch.tensor(np.asarray(x),dtype=torch.float64,device='cuda:0')
  # All covariance fits are exactly the empirical iid-donor objective / 2.
  zc={(s,k):tensor(z-z.mean(0)) for (s,k),z in codes.items()};dd={s:tensor(d) for s,d in dec.items()};query_by_seed={s:[(i,q['source_atom']) for i,q in enumerate(query) if q['source_seed']==s] for s in dec};n=cfg['positions'];rawc={k:tensor(np.asarray(raw[k][indices[k]],dtype=float)-np.mean(raw[k][indices[k]],axis=0,dtype=float)) for k in indices};rawgram=rawc['discovery'].T@rawc['discovery']/n;rawscale=torch.sqrt(torch.diag(rawgram));rawnorm=rawgram/rawscale[:,None]/rawscale[None,:];rawchol=torch.linalg.cholesky(rawnorm+cfg['ridge_fraction']*torch.eye(len(rawscale),device='cuda:0',dtype=torch.float64))
  for target in sorted(dec):
   xt=zc[target,'discovery'];xc=zc[target,'calibration'];dt=dd[target];gx=xt.T@xt/n;gc=xc.T@xc/n;dg=dt@dt.T;gn=gx*dg;gnc=gc*dg;active=torch.where(torch.diag(gn)>1e-12)[0];ca=torch.where(torch.diag(gnc)>1e-12)[0];scale=torch.sqrt(torch.diag(gn)[active]);cs=torch.sqrt(torch.diag(gnc)[ca]);norm=gn[active][:,active]/scale[:,None]/scale[None,:];cnorm=gnc[ca][:,ca]/cs[:,None]/cs[None,:]
   # Unregularized complete dictionary solution: record failures, never
   # substitute a ridge residual for an exact empirical lower bound.
   chol=torch.linalg.cholesky(norm);cchol=torch.linalg.cholesky(cnorm);rch=torch.linalg.cholesky(norm+cfg['ridge_fraction']*torch.eye(len(active),device='cuda:0',dtype=torch.float64));xscale=torch.sqrt(torch.diag(gx)[active]);xn=gx[active][:,active]/xscale[:,None]/xscale[None,:];xch=torch.linalg.cholesky(xn+cfg['ridge_fraction']*torch.eye(len(active),device='cuda:0',dtype=torch.float64))
   progress('target_kernel_factorized',target=target,active=len(active),calibration_active=len(ca))
   for source in sorted(dec):
    items=query_by_seed[source];ids=[a for _,a in items];ys=zc[source,'discovery'][:,ids];yc=zc[source,'calibration'][:,ids];ds=dd[source][ids];cos=dt@ds.T;h=(xt.T@ys/n)*cos;hc=(xc.T@yc/n)*cos;sv=torch.mean(ys*ys,axis=0)*torch.sum(ds*ds,axis=1);cv=torch.mean(yc*yc,axis=0)*torch.sum(ds*ds,axis=1);assert bool(torch.all(cv>0))
    nat=torch.zeros((len(dt),len(ids)),dtype=torch.float64,device='cuda:0');nr=nat.clone();reg=nat.clone();oracle=nat.clone();nat[active]=torch.cholesky_solve(h[active]/scale[:,None],chol)/scale[:,None];nr[active]=torch.cholesky_solve(h[active]/scale[:,None],rch)/scale[:,None];oracle[ca]=torch.cholesky_solve(hc[ca]/cs[:,None],cchol)/cs[:,None];reg[active]=torch.cholesky_solve((xt[:,active].T@ys/n)/xscale[:,None],xch)/xscale[:,None];rawbeta=torch.cholesky_solve((rawc['discovery'].T@ys/n)/rawscale[:,None],rawchol)/rawscale[:,None]
    bank=dict(native_full=nat,native_ridge_full=nr,readout_full=reg,raw=rawbeta,calibration_native_oracle=oracle);support_bank={}
    for cap in cfg['supports']:
     b=nat*0
     for j in range(len(ids)):
      order=torch.argsort(torch.abs(nr[:,j])*torch.sqrt(torch.clamp(torch.diag(gn),min=0)),descending=True,stable=True)[:cap];g=gn[order][:,order];sc=torch.sqrt(torch.diag(g));v=torch.linalg.solve(g/sc[:,None]/sc[None,:]+cfg['ridge_fraction']*torch.eye(cap,dtype=torch.float64,device='cuda:0'),h[order,j]/sc)/sc;b[order,j]=v;support_bank[cap,j]=order
     bank[f'native{cap}']=b
    reg64=nat*0;best=nat*0;geo=nat*0;bestids=[];geoids=[]
    for j in range(len(ids)):
     order=support_bank[64,j];g=gx[order][:,order];sc=torch.sqrt(torch.diag(g));reg64[order,j]=torch.linalg.solve(g/sc[:,None]/sc[None,:]+cfg['ridge_fraction']*torch.eye(64,dtype=torch.float64,device='cuda:0'),(xt[:,order].T@ys[:,j]/n)/sc)/sc
     score=torch.where(torch.diag(gn)>1e-12,h[:,j]**2/torch.clamp(torch.diag(gn),min=1e-30),-torch.inf);a=int(torch.argmax(score));b=int(torch.argmax(torch.abs(cos[:,j])/torch.linalg.norm(dt,dim=1)/torch.linalg.norm(ds[j])));best[a,j]=h[a,j]/gn[a,a];geo[b,j]=h[b,j]/gn[b,b] if gn[b,b]>1e-12 else 0;bestids.append(a);geoids.append(b)
    bank.update(readout_native64_support=reg64,best_native_atom=best,geometric_native_atom=geo)
    np.savez_compressed(run/f'maps_s{source}_t{target}.npz',source_atoms=np.array(ids),**{k:v.detach().cpu().numpy() for k,v in bank.items()})
    for j,(qi,atom) in enumerate(items):
     values={}
     for name,beta in bank.items():
      b=beta[:,j]
      if name.startswith('readout') or name=='raw':
       design=rawc['calibration'] if name=='raw' else xc;res=yc[:,j]-design@b;err=float(torch.mean(res*res)*torch.sum(ds[j]*ds[j]));orth=0.
      else:
       err=float(cv[j]-2*b@hc[:,j]+b@gnc@b);projected=b*cos[:,j]/torch.sum(ds[j]*ds[j]);res=yc[:,j]-xc@projected;parallel=float(torch.mean(res*res)*torch.sum(ds[j]*ds[j]));orth=err-parallel
       assert err>=-1e-8 and orth>=-1e-8,(source,target,atom,name,err,orth)
      values[name]=dict(relative_error=max(0.,err)/float(cv[j]),absolute_error=max(0.,err),orthogonal_error=max(0.,orth),support=int(torch.count_nonzero(b)))
     train_floor=float(sv[j]-nat[:,j]@h[:,j]);normal_res=float(torch.linalg.norm(gn@nat[:,j]-h[:,j])/torch.clamp(torch.linalg.norm(h[:,j]),min=1e-30));assert normal_res<1e-7
     if source==target:assert values['native_full']['relative_error']<1e-8
     row=dict(query_id=qi,source_seed=source,target_seed=target,source_atom=atom,quartile=query[qi]['quartile'],discovery_hits=query[qi]['discovery_hits'],calibration_hits=int(np.count_nonzero(codes[source,'calibration'][:,atom])),source_calibration_energy=float(cv[j]),native_discovery_oracle_floor=max(0.,train_floor)/float(sv[j]),normal_equation_relative_residual=normal_res,best_native_atom=bestids[j],geometric_native_atom=geoids[j],methods=values)
     rows.append(row)
     with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    progress('pair_complete',source=source,target=target,rows=len(rows))
   del xt,xc,gx,gc,gn,gnc,chol,cchol,rch,xch
  checks.update(complete_panel=len(rows)==25*cfg['queries_per_seed'],self_identity=True,normal_equations=True,zero_new_lm_forward=True)
  env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated(),dtype='float64 kernels and solves; original SAE float32')
 except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
 status='PASS' if error is None and checks and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(rows),lm_forwards=0,wall_seconds=time.perf_counter()-start,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'))
 write(run/'environment.json',env);write(run/'inputs.json',dict(inputs=inputs));write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True)
 return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
