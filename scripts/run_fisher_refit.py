"""Fit frozen-support operation-weighted maps on exposed natural inputs."""
import argparse,json,os,platform,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
import numpy as np
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from ccad.operation_weighted_fit import fit_operation_weighted
from f4_probability_endpoints import log_prob

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg);code=[]
 for rel in ['scripts/run_fisher_refit.py','scripts/run_r011s1_raw_hook_asset.py','scripts/f4_probability_endpoints.py','src/ccad/artifacts.py','src/ccad/activation_contract.py','src/ccad/operation_weighted_fit.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='operation.weighted.refit.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],milestone='C1-C3',evidence_level='source_behavior_supervised_development_fit',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original independent mean; donor differences cancel means',threshold_source_split='fixed before fit; no sweep',statistics_unit=cfg['scope'],device='cuda:0',seeds=[1,3],resource_lease='gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
 for n in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/n).touch()
 write(run/'status.json',dict(status='RUNNING'));inputs=[];checks={};env={};error=None;calls=0;jvps=0;diag={}
 def checked(p,expected=None):
  p=Path(p);p=p if p.is_absolute() else ROOT/p;rec=entry(p,'Existing CCAD artifacts / cached model','input','internal; source-supervised development');inputs.append(rec)
  if expected:assert rec['sha256']==expected,str(p)
  return p
 def load(p):return json.loads(checked(p).read_text(encoding='utf-8-sig'))
 try:
  checked(a.config);load('.aris/compute/local-r006b1-env-spec.json');asset=load(cfg['asset_config']);extra=load(cfg['target_asset_config']);asset['saes']+=extra['saes'];parent=ROOT/cfg['base_map_parent'];load(parent/'inputs.json')
  with np.load(checked(parent/'coefficients.npz')) as b:direction=np.array(b['source_decoder']);beta={m:np.array(b[m]) for m in b.files if m!='source_decoder'}
  b0=beta[cfg['fit_spec']['support_method']];support=np.flatnonzero(np.linalg.norm(b0,axis=1)>0);assert len(support)<=cfg['support_budget'];ops={k:np.array(v,float) for k,v in cfg['operators'].items()};dg=direction@direction.T
  cases=[];reference=[];saved=[]
  for panel in cfg['fit_panels']:
   cc=json.loads(checked(panel['inputs'],panel['sha256']).read_text())['cases'];rr=[json.loads(s) for s in checked(Path(panel['source_parent'])/'metrics.raw.jsonl').read_text().splitlines()];refs={r['case_id']:r for r in rr};probs=np.load(checked(Path(panel['source_parent'])/'probabilities.npz'));saved.append(probs)
   for i,c in enumerate(cc):cases.append(dict(c,panel=panel['inputs'],local_case_id=i));reference.append((refs[i],probs,i))
  assert len({c['document_id'] for c in cases})==len(cases)==48
  import torch,transformers
  from sparsify import SparseCoder
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
  tok=transformers.AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True);model=transformers.AutoModelForCausalLM.from_pretrained(asset['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.config.use_cache=False
  for p in model.parameters():p.requires_grad_(False)
  saes={}
  for seed in [cfg['source_seed'],cfg['target_seed']]:
   item=next(r for r in asset['saes'] if r['seed']==seed);weight=checked(Path(item['path'])/'sae.safetensors',item['sha256']);saes[seed]=SparseCoder.load_from_disk(weight.parent,device='cuda:0').eval()
  assert np.array_equal(saes[cfg['source_seed']].W_dec.detach().cpu().numpy()[cfg['source_atoms']].astype(float),direction)
  module=model.get_submodule(asset['hook_module_path']);contract=HookPointContract(asset['hook_module_path'],5,'resid_post',768);dt=torch.tensor(direction,device='cuda:0',dtype=torch.float32);hooks=[];codes={seed:[] for seed in saes};batches=[]
  for c in cases:
   tokens=torch.tensor([c['token_ids']],device='cuda:0');batches.append(tokens);obs={}
   def capture(m,inp,out):obs['h']=extract_primary_hook_tensor(out,contract)[0,-1].detach()
   handle=module.register_forward_hook(capture)
   try:
    with torch.no_grad():model(tokens,use_cache=False)
   finally:handle.remove()
   calls+=1;h=obs['h'];hooks.append(h.cpu().numpy())
   with torch.no_grad():
    for seed,sae in saes.items():
     encoded=sae.encode(h[None]);z=np.zeros(3072);z[encoded.top_indices[0].cpu().numpy()]=encoded.top_acts[0].cpu().numpy();codes[seed].append(z)
  x=[];y=[];doses=[];fisher=[];maxprob=0.
  for i,(c,tokens) in enumerate(zip(cases,batches)):
   truth=codes[cfg['source_seed']][i^1][cfg['source_atoms']]-codes[cfg['source_seed']][i][cfg['source_atoms']];dx=codes[cfg['target_seed']][i^1]-codes[cfg['target_seed']][i];deltas={op:(truth*theta)@direction for op,theta in ops.items()};dose=min(1.,cfg['max_hook_fraction']*np.linalg.norm(hooks[i])/max(max(np.linalg.norm(d) for d in deltas.values()),1e-30));ref,probs,oldi=reference[i];assert np.array_equal(truth,ref['source_difference']) and np.isclose(dose,ref['common_dose'],rtol=1e-7,atol=1e-10)
   x.append(dx[support]);y.append(truth);doses.append(dose);point={}
   for op,theta in ops.items():
    delta=torch.tensor(dose*deltas[op],device='cuda:0',dtype=torch.float32)
    def func(w):
     nonlocal calls
     def hook(m,inp,out):
      h=extract_primary_hook_tensor(out,contract);changed=h.clone();changed[0,-1]+=delta+w@dt;return replace_primary_hook_tensor(out,changed,contract)
     handle=module.register_forward_hook(hook)
     try:lg=model(tokens,use_cache=False).logits[0,-1].float()
     finally:handle.remove()
     calls+=1;return lg
    js=[];oldlg=None
    for k in range(2):
     lg,j=torch.autograd.functional.jvp(func,torch.zeros(2,device='cuda:0'),torch.eye(2,device='cuda:0')[k],create_graph=False,strict=True);jvps+=1;lg=lg.detach().cpu().numpy();assert oldlg is None or np.array_equal(lg,oldlg);oldlg=lg;js.append(j.detach().cpu().numpy())
    J=np.stack(js,axis=1).astype(float);p=np.exp(log_prob(lg[None])[0]);maxprob=max(maxprob,float(np.max(abs(p-probs[f'source_{op}_{oldi}']))));mu=p@J;F=J.T@(p[:,None]*J)-np.outer(mu,mu);point[op]=(F+F.T)/2
    with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(dict(case_id=i,panel=c['panel'],local_case_id=oldi,operator=op,source_difference=truth.tolist(),common_dose=float(dose),fisher=point[op].tolist()))+'\n')
   fisher.append(point)
   if (i+1)%8==0:print(json.dumps(dict(stage='FITTING_SOURCE_GEOMETRY',cases=i+1,calls=calls,seconds=time.perf_counter()-start)),flush=True)
   if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Fitting wall budget')
  for p in saved:p.close()
  checks.update(source_probability_match=maxprob<1e-5,calls=calls==432,jvps=jvps==384)
  assert all(checks.values()),checks
  x=np.array(x);y=np.array(y);constant=np.mean([F for point in fisher for F in point.values()],axis=0);grams={}
  for method in ['euclidean_refit','constant_fisher_refit','pointwise_fisher_refit']:
   q=[]
   for i,point in enumerate(fisher):
    qi=np.zeros((2,2))
    for op,theta in ops.items():
     F=dg if method=='euclidean_refit' else constant if method=='constant_fisher_refit' else point[op];qi+=doses[i]**2*(theta[:,None]*F*theta[None,:])/len(ops)
    q.append(qi)
   q=np.array(q);grams[method]=q;fitted,info=fit_operation_weighted(x,y,q,b0[support],cfg['fit_spec']['ridge_fraction']);beta[method]=np.zeros_like(b0);beta[method][support]=fitted;diag[method]=info;assert info['fitted_objective']<=info['prior_objective']*(1+1e-10)+1e-12 and info['normal_equation_relative_error']<1e-9
  np.savez_compressed(run/'fit_arrays.npz',x=x,y=y,support=support,prior=b0[support],**grams);np.savez_compressed(run/'coefficients.npz',source_decoder=direction,**beta);write(run/'fit_metadata.json',dict(diagnostics=diag,source_probability_max_abs_error=maxprob,parameters=len(support)*2,independent_document_pairs=len(cases)//2,support=support.tolist(),same_prior_support=True,fit_case_ids=cases,original_method_names=[m for m in beta if not m.endswith('_refit')],coefficient_freeze_utc=datetime.now(timezone.utc).isoformat()));checks['fit_decreases_regularized_objective']=True
  env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated())
 except Exception:
  error=traceback.format_exc();(run/'stderr.log').write_text(error)
 status='FAIL' if error or not checks or not all(checks.values()) else 'PASS';write(run/'environment.json',env);write(run/'inputs.json',dict(inputs=inputs));summary=dict(status=status,error=error,checks=checks,fit_diagnostics=diag,wall_seconds=time.perf_counter()-start,new_lm_forward_calls=calls,jvp_calls=jvps,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'));write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat()));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));write(run/'stdout.log',summary);print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors)),indent=2));return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
