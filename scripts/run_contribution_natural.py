"""Frozen contribution maps on fresh natural donor interventions, all seeds."""
import argparse,json,platform,time,traceback,sys
from pathlib import Path
from datetime import datetime,timezone
from run_contribution_completion import ROOT,np,entry,aggregate,write,sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from f4_probability_endpoints import log_prob

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);cfg=json.loads(ap.parse_args().config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg);files=[]
 for rel in ['scripts/run_contribution_natural.py','scripts/run_contribution_completion.py','scripts/run_f4_source_reference_causal.py','scripts/run_r011s1_raw_hook_asset.py','scripts/f4_probability_endpoints.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']:
  p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());files.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=files,aggregate_sha256=aggregate(files),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='contribution.natural.v1',run_id=cfg['run_id'],run_parent=cfg['parent_run'],purpose=cfg['purpose'],milestone='C1-C2-C3',evidence_level='fresh_natural_frozen_contribution_operations',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(files),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original independent mean; cancels in donor differences',threshold_source_split='frozen source-only anchor activity and same-token cross-document donor rule',statistics_unit='document and source-query families sharing five seeds; target directions dependent',device='cuda:0',seeds=[1,2,3,4,5],resource_lease='cpu-heavy -> gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
 for f in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/f).touch()
 write(run/'status.json',dict(status='RUNNING'));inputs=[];rows=[];checks={};error=None;env={};forwards=0
 def checked(path,digest=None):
  p=Path(path);p=p if p.is_absolute() else ROOT/p
  if digest and sha256(p)!=digest:raise ValueError('Frozen identity changed '+str(p))
  inputs.append(entry(p,'Existing CCAD frozen map or fresh independent corpus','input'));return p
 def load(p):return json.loads(checked(p).read_text())
 def progress(stage,**kw):
  v=dict(stage=stage,seconds=time.perf_counter()-start,**kw);write(run/'progress.json',v);print(json.dumps(v),flush=True)
  if v['seconds']>cfg['budget_seconds']:raise TimeoutError('Bounded real experiment budget exceeded')
 try:
  import torch,transformers
  from sparsify import SparseCoder
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
  parent=ROOT/'runs'/cfg['parent_run'];grain=ROOT/'runs'/cfg['grain_run'];corpus=ROOT/'runs'/cfg['corpus_run'];assert load(parent/'status.json')['status']=='PASS' and load(grain/'status.json')['status']=='PASS' and load(corpus/'status.json')['status']=='PASS'
  for spec in cfg['frozen_inputs']:checked(spec['path'],spec['sha256'])
  pc=load(parent/'config.resolved.json');asset=load(pc['asset_config']);allq=load(parent/'source_queries.json')['queries'];group=load(grain/'source_groups.json')['groups'];selected=cfg['query_ids'];assert len(selected)==40
  # Load and bind all fitted maps before new activations or behavior is read.
  maps={};dec={};saes={}
  for s in range(1,6):
   for t in range(1,6):
    if s==t:continue
    for name,path in [('fine',parent),('grain',grain)]:
     with np.load(checked(path/f'maps_s{s}_t{t}.npz')) as ar:maps[name,s,t]={k:np.array(ar[k]) for k in ar.files}
   cp=next(x for x in pc['checkpoints'] if x['seed']==s);weight=checked(Path(cp['path'])/'sae.safetensors',cp['sha256']);saes[s]=SparseCoder.load_from_disk(weight.parent,device='cuda:0').eval();dec[s]=saes[s].W_dec.detach().cpu().numpy().astype(float)
  checks['maps_loaded_before_fresh_activations']=True
  tm=load(corpus/'artifacts/token_manifest.json')['outputs']['calibration'];tokens=np.fromfile(checked(corpus/tm['path'],tm['sha256']),dtype='<u2').reshape(-1,128).astype(np.int64);seq=load(corpus/'artifacts/sequence_records.json')['sequences'];seq=[r for r in seq if r['split']=='calibration'];single={r['sequence_index']:r['document_ids'][0] for r in seq if len(r['document_ids'])==1};tok=transformers.AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True)
  model=transformers.AutoModelForCausalLM.from_pretrained(asset['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.config.use_cache=False;module=model.get_submodule(asset['hook_module_path']);contract=HookPointContract(asset['hook_module_path'],5,'resid_post',768)
  def forward(ids,positions=None,edits=None,capture=False):
   nonlocal forwards
   captured=[];batch=torch.tensor(ids,device='cuda:0');pos=None if positions is None else torch.tensor(positions,device='cuda:0')
   def hook(m,i,out):
    h=extract_primary_hook_tensor(out,contract)
    if capture:captured.append(h.detach().cpu().numpy())
    if edits is None:return out
    new=h.clone();new[torch.arange(len(batch),device='cuda:0'),pos]+=torch.tensor(edits,dtype=h.dtype,device='cuda:0');return replace_primary_hook_tensor(out,new,contract)
   handle=module.register_forward_hook(hook)
   try:
    with torch.no_grad():
     hidden=model.gpt_neox(batch,use_cache=False).last_hidden_state;logits=None if positions is None else model.get_output_embeddings()(hidden[torch.arange(len(batch),device='cuda:0'),pos]).float().cpu().numpy()
   finally:handle.remove()
   forwards+=len(batch);return logits,captured[0] if capture else None
  # Verify selected-position evaluation agrees with the public full LM path.
  l,_=forward(tokens[:1],[100])
  with torch.no_grad():public=model(torch.tensor(tokens[:1],device='cuda:0'),use_cache=False).logits[0,100].cpu().numpy()
  forwards+=1;checks['selected_logits_match_full_forward']=bool(np.allclose(l[0],public,rtol=1e-5,atol=1e-5));assert checks['selected_logits_match_full_forward']
  raw=np.empty((len(tokens),128,768),dtype=np.float32)
  for i in range(0,len(tokens),cfg['batch_size']):_,raw[i:i+cfg['batch_size']]=forward(tokens[i:i+cfg['batch_size']],capture=True)
  np.save(run/'raw_hook.npy',raw);np.save(run/'tokens.npy',tokens);progress('fresh_hook_capture',sequences=len(tokens),single_document_sequences=len(single))
  flat=raw.reshape(-1,768);pool=np.array([q*128+p for q in single for p in range(8,127) if tokens[q,p]!=tok.eos_token_id and tokens[q,p+1]!=tok.eos_token_id]);pt=tokens.ravel()[pool];docs=np.array([single[int(p)//128] for p in pool]);anchor_values={}
  def encode(s,arr):
   ans=np.zeros((len(arr),3072),dtype=np.float32)
   with torch.no_grad():
    for i in range(0,len(arr),512):
     e=saes[s].encode(torch.tensor(np.array(arr[i:i+512]),device='cuda:0',dtype=torch.float32));ix=e.top_indices.cpu().numpy();v=e.top_acts.cpu().numpy();np.add.at(ans,(np.arange(i,i+len(ix))[:,None],ix),v)
   return ans
  cases=[];missing=[]
  for s in range(1,6):
   qids=[i for i in selected if allq[i]['source_seed']==s];zz=encode(s,flat[pool]);anchor_values[s]=zz[:,[allq[i]['source_atom'] for i in qids]]
   for ji,qi in enumerate(qids):
    q=allq[qi];a=anchor_values[s][:,ji];used=set();chosen=[]
    for ri in np.argsort(-a,kind='stable'):
     if a[ri]<=0:break
     if docs[ri] in used:continue
     donors=np.flatnonzero((pt==pt[ri])&(docs!=docs[ri])&(a<a[ri]))
     if not len(donors):continue
     di=int(donors[np.argmin(a[donors])]);used.add(docs[ri]);chosen.append(dict(case_id=len(cases),query_id=qi,source_seed=s,anchor=q['source_atom'],quartile=q['quartile'],recipient=int(pool[ri]),donor=int(pool[di]),recipient_document=str(docs[ri]),donor_document=str(docs[di]),source_anchor_recipient=float(a[ri]),source_anchor_donor=float(a[di]),token_id=int(pt[ri]),token=tok.decode([int(pt[ri])]),recipient_text=tok.decode(tokens[int(pool[ri])//128,:int(pool[ri])%128+1]),donor_text=tok.decode(tokens[int(pool[di])//128,:int(pool[di])%128+1])))
     cases.append(chosen[-1])
     if len(chosen)==cfg['pairs_per_query']:break
    if len(chosen)<cfg['pairs_per_query']:missing.append(dict(query_id=qi,found=len(chosen),requested=cfg['pairs_per_query']))
   del zz;progress('source_cases_selected',source=s,cases=len(cases))
  write(run/'source_cases.json',dict(cases=cases,missing=missing,rule=cfg['source_rule'],target_behavior_used=False));assert cases
  chosen_rows=sorted({c[k] for c in cases for k in ['recipient','donor']});loc={x:i for i,x in enumerate(chosen_rows)};codes={s:encode(s,flat[chosen_rows]).astype(float) for s in range(1,6)};np.savez_compressed(run/'selected_codes.npz',row_indices=chosen_rows,**{f'seed{s}':v for s,v in codes.items()})
  saes.clear()
  # All geometry and operational choices are saved before intervention outcomes.
  plans=[];edits=[]
  for case in cases:
   s=case['source_seed'];qi=case['query_id'];ri=case['recipient'];di=case['donor'];zs=codes[s][loc[di]]-codes[s][loc[ri]];groups=[(j,q) for j,q in enumerate(group[str(s)]) if q['query_id']==qi and q['size'] in cfg['group_sizes'] and (q['kind']=='coherent' or q['size'] in cfg['random_source_sizes'])]
   for gj,gq in groups:
    truth=zs[gq['source_atoms']]@dec[s][gq['source_atoms']];dose=min(1.,cfg['max_hook_fraction']*float(np.linalg.norm(flat[ri]))/max(float(np.linalg.norm(truth)),1e-30));source=truth*dose;key=len(plans);plan=dict(case_id=case['case_id'],query_id=qi,source_seed=s,anchor=case['anchor'],recipient=ri,donor=di,size=gq['size'],kind=gq['kind'],source_atoms=gq['source_atoms'],dose=dose,source_energy=float(source@source),methods=[])
    source_edit_index=len(edits);edits.append(source)
    for t in range(1,6):
     if t==s:continue
     dz=codes[t][loc[di]]-codes[t][loc[ri]];fineids=maps['fine',s,t]['source_atoms'];fj=int(np.where(fineids==case['anchor'])[0][0]);bank={name:(dz*maps['grain',s,t][name][:,gj])@dec[t] for name in cfg['native_methods']}
     if gq['size']==1:
      bank.update(best_native_atom=(dz*maps['fine',s,t]['best_native_atom'][:,fj])@dec[t],readout_same64=(dz@maps['fine',s,t]['readout_native64_support'][:,fj])*dec[s][case['anchor']],raw=((flat[di].astype(float)-flat[ri])@maps['fine',s,t]['raw'][:,fj])*dec[s][case['anchor']])
     for method,vector in bank.items():
      edit=vector*dose;record=dict(target_seed=t,method=method,edit_index=len(edits),vector_squared_error=float(np.sum((edit-source)**2)),candidate_energy=float(edit@edit));edits.append(edit);plan['methods'].append(record)
    plan['source_edit_index']=source_edit_index;plans.append(plan)
  write(run/'operation_plan.json',dict(plans=plans,scope='Frozen before all edited LM outputs; source-only dose, no candidate renormalization'));np.save(run/'edits.npy',np.array(edits));checks['all_operations_frozen_before_behavior']=True
  # Stream full-vocabulary scores per source operation to bound disk and RAM.
  for pi,plan in enumerate(plans):
   ri=plan['recipient'];sequence=tokens[ri//128];pos=ri%128;targettoken=int(sequence[pos+1]);source=edits[plan['source_edit_index']];base,_=forward(sequence[None],[pos]);ref,_=forward(sequence[None],[pos],[source]);lb=log_prob(base)[0];lr=log_prob(ref)[0];p=np.exp(lr);den=max(0.,float(np.sum(p*(lr-lb))));snll=float(lb[targettoken]-lr[targettoken]);allm=plan['methods']
   if pi==0:
    noop,_=forward(sequence[None],[pos],[np.zeros(768)]);checks['noop_exact']=bool(np.array_equal(noop,base));assert checks['noop_exact']
   for off in range(0,len(allm),cfg['batch_size']):
    sub=allm[off:off+cfg['batch_size']];lg,_=forward(np.tile(sequence,(len(sub),1)),[pos]*len(sub),[edits[m['edit_index']] for m in sub]);ll=log_prob(lg)
    for m,lc in zip(sub,ll):
     kl=max(0.,float(np.sum(p*(lr-lc))));cnll=float(lb[targettoken]-lc[targettoken]);r=dict(**{k:v for k,v in plan.items() if k!='methods'},**m,source_kl=den,candidate_kl=kl,relative_kl=kl/den if den>1e-12 else None,source_nll_change=snll,candidate_nll_change=cnll,nll_change_squared_error=(snll-cnll)**2,observed_next_token=targettoken,source_observed_probability=float(p[targettoken]),candidate_observed_probability=float(np.exp(lc[targettoken])));rows.append(r)
     with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   if pi%10==0 or pi==len(plans)-1:progress('natural_operations',complete=pi+1,total=len(plans),rows=len(rows),lm_sequences=forwards)
  checks.update(all_plans_evaluated=len(rows)==sum(len(p['methods']) for p in plans),no_refit=True,finite=all(np.isfinite(r['candidate_kl']) for r in rows));env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated())
 except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
 status='PASS' if error is None and checks and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(rows),lm_sequences=forwards,wall_seconds=time.perf_counter()-start,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope=cfg['scope']);write(run/'environment.json',env);write(run/'inputs.json',dict(inputs=inputs));write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True)
 return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
