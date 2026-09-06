"""Source-point Fisher selection pilot; reuses exposed, fixed candidate maps."""
import argparse,json,os,platform,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
import numpy as np
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from f4_probability_endpoints import log_prob

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg);code=[]
 for rel in ['scripts/run_pointwise_fisher.py','scripts/run_r011s1_raw_hook_asset.py','scripts/f4_probability_endpoints.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='pointwise.fisher.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],milestone='C1-C3',evidence_level='exposed_method_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original independent means; donor differences fixed',threshold_source_split='fixed config before Fisher',statistics_unit=cfg['scope'],device='cuda:0',seeds=[1,2,3],resource_lease='gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
 for n in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/n).touch()
 write(run/'status.json',dict(status='RUNNING'));inputs=[];checks={};folds=[];env={};error=None;calls=0;jvps=0;comparisons={}
 def checked(p):
  p=Path(p);p=p if p.is_absolute() else ROOT/p;inputs.append(entry(p,'Existing CCAD artifacts / cached model','input','internal development; no new data'));return p
 def load(p):return json.loads(checked(p).read_text(encoding='utf-8-sig'))
 def raw(p):return [json.loads(s) for s in checked(p/'metrics.raw.jsonl').read_text().splitlines()]
 try:
  checked(a.config);load('.aris/compute/local-r006b1-env-spec.json');parent=ROOT/'runs'/cfg['source_parent'];pc=load(parent/'config.resolved.json');asset=load(pc['asset_config']);source_rows=raw(parent);cases=load(parent/'authored_inputs.json')['cases'];case_row={r['case_id']:r for r in source_rows};ops={k:np.array(v,float) for k,v in pc['operators'].items()}
  load(parent/'inputs.json')
  with np.load(checked(parent/'coefficients.npz')) as b:direction=np.array(b['source_decoder'])
  saved=np.load(checked(parent/'probabilities.npz'));dg=direction@direction.T
  import torch,transformers
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
  tok=transformers.AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True)
  model=transformers.AutoModelForCausalLM.from_pretrained(asset['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.config.use_cache=False
  for p in model.parameters():p.requires_grad_(False)
  env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name())
  module=model.get_submodule(asset['hook_module_path']);contract=HookPointContract(asset['hook_module_path'],5,'resid_post',768);dt=torch.tensor(direction,device='cuda:0',dtype=torch.float32);fish={};jac={};witness=[];maxdiff=0.
  for i,c in enumerate(cases):
   tokens=torch.tensor([c['token_ids'] if 'token_ids' in c else tok.encode(c['text'],add_special_tokens=False)],device='cuda:0');row=case_row[i]
   for op,theta in ops.items():
    delta=torch.tensor(row['common_dose']*(np.array(row['source_difference'])*theta)@direction,device='cuda:0',dtype=torch.float32)
    def func(w):
     nonlocal calls
     def hook(m,inp,out):
      h=extract_primary_hook_tensor(out,contract);changed=h.clone();changed[0,-1]+=delta+w@dt
      return replace_primary_hook_tensor(out,changed,contract)
     handle=module.register_forward_hook(hook)
     try:out=model(tokens,use_cache=False).logits[0,-1].float()
     finally:handle.remove()
     calls+=1;return out
    js=[];previous=None
    for k in range(2):
     logits,j=torch.autograd.functional.jvp(func,torch.zeros(2,device='cuda:0'),torch.eye(2,device='cuda:0')[k],create_graph=False,strict=True);jvps+=1
     logits=logits.detach().cpu().numpy();js.append(j.detach().cpu().numpy());assert previous is None or np.array_equal(previous,logits);previous=logits
    J=np.stack(js,axis=1).astype(float);p=np.exp(log_prob(logits[None])[0]);maxdiff=max(maxdiff,float(np.max(abs(p-saved[f'source_{op}_{i}']))));mu=p@J;F=J.T@(p[:,None]*J)-np.outer(mu,mu);F=(F+F.T)/2;assert np.isfinite(F).all() and np.linalg.eigvalsh(F).min()>-1e-10
    fish[i,op]=F;jac[f'j_{i}_{op}']=J.astype(np.float32)
    if (i,op) in [(0,'field_component'),(1,'difference')]:
     for k in range(2):
      steps=[]
      for eps in cfg.get('finite_difference_epsilons',[cfg['finite_difference_epsilon']]):
       v=torch.eye(2,device='cuda:0')[k]*eps
       with torch.no_grad():fd=((func(v)-func(-v))/(2*eps)).cpu().numpy().astype(float)
       relative=float(np.linalg.norm(fd-J[:,k])/max(np.linalg.norm(J[:,k]),1e-30));steps.append(dict(epsilon=eps,relative_l2_error=relative,jvp_l2_norm=float(np.linalg.norm(J[:,k]))))
      witness.append(dict(case_id=i,operator=op,coordinate=k,steps=steps));write(run/'witness.json',dict(finite_difference=witness,probability_max_abs_error=maxdiff))
      assert sum(s['relative_l2_error']<cfg['finite_difference_relative_tolerance'] for s in steps)>=cfg.get('finite_difference_required_passes',1),witness[-1]
    with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(dict(case_id=i,operator=op,fisher=F.tolist(),eigenvalues=np.linalg.eigvalsh(F).tolist()))+'\n')
   print(json.dumps(dict(stage='SOURCE_FISHER',case=i,forward_calls=calls,seconds=time.perf_counter()-start)),flush=True)
   if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Probe wall budget')
  saved.close();checks.update(source_probability_match=maxdiff<=cfg['probability_match_atol'],finite_difference=len(witness)==4,expected_jvps=jvps==96,expected_calls=calls==96+8*len(cfg.get('finite_difference_epsilons',[cfg['finite_difference_epsilon']])));write(run/'witness.json',dict(finite_difference=witness,probability_max_abs_error=maxdiff));assert all(checks.values()),checks
  np.savez_compressed(run/'logit_jacobians.npz',**jac);write(run/'fisher.json',dict(records=[dict(case_id=i,operator=op,fisher=F.tolist()) for (i,op),F in fish.items()],definition='Full-vocabulary logit Fisher at source intervention; no Jacobian averaging'))
  env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated());del model;torch.cuda.empty_cache()
  for name in cfg['parents']:
   pr=ROOT/'runs'/name;rr=raw(pr);conf=load(pr/'config.resolved.json');assert conf['operators']==pc['operators']
   with np.load(checked(pr/'coefficients.npz')) as b:
    assert np.array_equal(direction,b['source_decoder']);supports={m:int(np.count_nonzero(np.linalg.norm(b[m],axis=1))) for m in b.files if m!='source_decoder'}
   for r in rr:
    ref=case_row[r['case_id']];assert r['text']==ref['text'] and r['common_dose']==ref['common_dose'] and r['source_difference']==ref['source_difference']
   methods=sorted(supports);candidates=[m for m in methods if supports[m]<=cfg['support_budget'] and m!='raw'];pairs=sorted({r['pair'] for r in rr});tv={}
   with np.load(checked(pr/'probabilities.npz')) as probs:
    for r in rr:
     i,op,m=r['case_id'],r['operator'],r['method'];tv[i,op,m]=float(.5*np.sum(abs(probs[f'source_{op}_{i}'].astype(float)-probs[f'{m}_{op}_{i}'].astype(float))))
   def endpoint(rows,metric):
    return max(float(np.median([r[metric] for r in rows if r['operator']==op and r[metric] is not None])) for op in ops)
   for pair in pairs:
    calrows=[r for r in rr if r['pair']!=pair];testrows=[r for r in rr if r['pair']==pair];ids=sorted({r['case_id'] for r in calrows});constant=np.mean([fish[i,op] for i in ids for op in ops],axis=0);cal={};actual={}
    for m in methods:
     proxy=[]
     for r in calrows:
      if r['method']!=m:continue
      i,op=r['case_id'],r['operator'];e=r['common_dose']*(np.array(r['predicted_difference'])-np.array(r['source_difference']))*ops[op];den=r['source_kl']
      proxy.append(dict(operator=op,euclidean_matched=r['vector_squared_error']/r['source_delta_energy'] if r['source_delta_energy']>0 else None,fisher_constant=float(.5*e@constant@e)/den if den>1e-12 else None,fisher_pointwise=float(.5*e@fish[i,op]@e)/den if den>1e-12 else None))
     cal[m]={s:endpoint(proxy,s) for s in cfg['selectors']};tr=[dict(r,tv=tv[r['case_id'],r['operator'],m]) for r in testrows if r['method']==m];actual[m]=dict(worst_median_kl_ratio=endpoint(tr,'normalized_kl_error'),worst_median_absolute_kl=endpoint(tr,'candidate_kl'),worst_median_tv=endpoint(tr,'tv'))
    selected={}
    for s in cfg['selectors']:
     chosen=min(candidates,key=lambda m:(cal[m][s],m));selected[s]=dict(method=chosen,calibration_score=cal[chosen][s],**actual[chosen])
    folds.append(dict(parent=name,pair=pair,candidates=candidates,calibration=cal,selected=selected,test_all_methods=actual))
   print(json.dumps(dict(stage='SELECTION',parent=name,folds=len(folds),seconds=time.perf_counter()-start)),flush=True)
  write(run/'folds.json',dict(folds=folds,scope=cfg['scope']))
  for metric in ['worst_median_kl_ratio','worst_median_absolute_kl','worst_median_tv']:
   comparisons[metric]={s:dict(mean=float(np.mean([f['selected'][s][metric] for f in folds])),median=float(np.median([f['selected'][s][metric] for f in folds])),pointwise_wins=sum(f['selected']['fisher_pointwise'][metric]<f['selected'][s][metric] for f in folds),pointwise_losses=sum(f['selected']['fisher_pointwise'][metric]>f['selected'][s][metric] for f in folds),different_choices=sum(f['selected']['fisher_pointwise']['method']!=f['selected'][s]['method'] for f in folds)) for s in cfg['selectors']}
  checks['all_folds']=len(folds)==60
 except Exception:
  error=traceback.format_exc();(run/'stderr.log').write_text(error)
 status='FAIL' if error or not checks or not all(checks.values()) else 'PASS';write(run/'environment.json',env);write(run/'inputs.json',dict(inputs=inputs));summary=dict(status=status,error=error,checks=checks,folds=len(folds),comparisons=comparisons,wall_seconds=time.perf_counter()-start,new_lm_forward_calls=calls,jvp_calls=jvps,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'));write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat()));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));write(run/'stdout.log',summary);print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors)),indent=2));return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
