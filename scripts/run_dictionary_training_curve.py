"""Controlled dictionary_learning objectives on one exact natural token stream."""
from __future__ import annotations
import os,sys,json,time,platform,argparse,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);cfg=json.loads(ap.parse_args().config.read_text())
 run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);bulk=Path(cfg['bulk_output_dir']);bulk.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter();cpu=time.process_time();started=datetime.now(timezone.utc).isoformat();write(run/'config.resolved.json',cfg)
 codes=[]
 for rel in ['scripts/run_dictionary_training_curve.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());codes.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='dictionary.training.v1',run_id=cfg['run_id'],run_parent=cfg.get('round_id','FINAL_THREE_R19'),purpose=cfg['purpose'],milestone='M4',evidence_level='controlled_second_family_training',started_utc=started,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(codes),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='validation for quality only; trainer first-batch median',threshold_source_split='unchanged trainer EMA on training activations',statistics_unit='initialization seed within objective; fixed validation document stream',device='cuda:0',seeds=cfg['seeds'],resource_lease='gpu-0 resource_manager.run',resource_lease_reason='Ten controlled SAEs share one base-model activation batch; two CPU threads; bounded intermittent checkpoint writes'))
 for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/fn).touch()
 write(run/'status.json',dict(status='RUNNING',updated_utc=started));inputs=[];env={};checks={};snaps=[];error=None;rows=[];trained=0;timings={}
 def log(event,**kw):
  value=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),event=event,**kw)
  with (run/'stdout.log').open('a') as f:f.write(json.dumps(value)+'\n')
  print(json.dumps(value),flush=True)
 try:
  import torch,numpy as np,transformers
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
  sys.path.append(cfg['dictionary_source_dir']);sys.path.append(cfg['dictionary_overlay_dir'])
  from dictionary_learning.trainers.top_k import TopKTrainer
  from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer
  for rel in ['dictionary_learning/trainers/top_k.py','dictionary_learning/trainers/matryoshka_batch_top_k.py','dictionary_learning/trainers/trainer.py','dictionary_learning/dictionary.py','LICENSE']:
   p=Path(cfg['dictionary_source_dir'])/rel;inputs.append(entry(p,'dictionary_learning '+cfg['dictionary_commit'],'unchanged trainer source','MIT'))
  manifest_path=ROOT/cfg['token_manifest'];token_manifest=json.loads(manifest_path.read_text());inputs.append(entry(manifest_path,'retained FineWeb documents','token manifest','ODC-By-1.0'))
  data={}
  for split in ['train','validation']:
   info=token_manifest['outputs'][split];p=Path(info['path']);assert sha256(p)==info['sha256'];inputs.append(entry(p,'GPT-2 tokenized natural corpus',split,'ODC-By-1.0'));data[split]=np.memmap(p,dtype='<u2',mode='r').reshape(-1,128)
  for name in ['config.json','tokenizer.json','model.safetensors']:inputs.append(entry(Path(cfg['model_local_dir'])/name,cfg['model_id']+' '+cfg['model_revision'],'base model','MIT'))
  model=transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.requires_grad_(False);model.config.use_cache=False
  module=model.get_submodule(cfg['hook_module_path']);d=model.config.hidden_size;bs=cfg['batch_size_sequences'];steps=cfg['steps'];assert len(data['train'])==bs*steps
  env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),threads=2,precision='float32; matmul precision high',hook=cfg['hook_module_path'],hidden_size=d,dictionary_commit=cfg['dictionary_commit'])
  trainers={}
  for objective in cfg['objectives']:
   for seed in cfg['seeds']:
    kw=dict(steps=steps,activation_dim=d,dict_size=cfg['dict_size'],k=cfg['k'],layer=cfg['layer'],lm_name=cfg['model_id'],lr=cfg['learning_rate'],warmup_steps=cfg['warmup_steps'],decay_start=cfg['decay_start'],threshold_start_step=cfg['threshold_start_step'],seed=seed,device='cuda:0')
    trainer=TopKTrainer(**kw) if objective=='topk' else MatryoshkaBatchTopKTrainer(**kw,group_fractions=cfg['group_fractions'])
    trainers[(objective,seed)]=trainer
  write(run/'trainer_configs.json',dict(rows=[dict(objective=o,seed=s,config=t.config) for (o,s),t in trainers.items()]))
  class StopAtHook(Exception):pass
  def capture(ids,stop=True):
   observed={}
   def hook(m,i,out):
    observed['x']=(out[0] if isinstance(out,tuple) else out).detach()
    if stop:raise StopAtHook()
   h=module.register_forward_hook(hook)
   try:
    with torch.no_grad():
     try:result=model(ids,use_cache=False,output_hidden_states=not stop)
     except StopAtHook:result=None
   finally:h.remove()
   if not stop:checks['hook_oracle_max_error']=float((observed['x']-result.hidden_states[cfg['layer']+1]).abs().max());assert checks['hook_oracle_max_error']==0
   return observed['x']
  first=torch.as_tensor(data['train'][:bs].astype('int64'),device='cuda:0');complete=capture(first,stop=False);early=capture(first)
  checks['truncated_forward_max_error']=float((complete-early).abs().max());assert checks['truncated_forward_max_error']==0
  del complete,early
  valid_ids=torch.as_tensor(data['validation'].astype('int64'))
  def save(step,exact=False):
   dest=bulk/f'step_{step}';dest.mkdir(exist_ok=True)
   for (o,s),tr in trainers.items():
    path=dest/f'{o}_seed{s}.pt';torch.save(tr.ae.state_dict(),path)
    snaps.append(dict(step=step,objective=o,seed=s,path=str(path),sha256=sha256(path),tokens=step*bs*128))
   if exact:
    p=dest/'training_state.pt'
    torch.save(dict(completed_steps=step,trainers={f'{o}_{s}':dict(optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),num_tokens_since_fired=t.num_tokens_since_fired) for (o,s),t in trainers.items()},torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),p)
    write(run/'exact_state.json',dict(path=str(p),sha256=sha256(p),completed_steps=step,model_weights_directory=str(dest)))
   write(run/'checkpoints.json',dict(checkpoints=snaps))
  @torch.no_grad()
  def quality(step):
   # Per-sequence CE is retained; full fixed validation is evaluated for every SAE.
   totals={key:dict(sse=0.,sum=torch.zeros(d,device='cuda:0',dtype=torch.float64),sq=0.,n=0,l0=0,counts=torch.zeros(cfg['dict_size'],device='cuda:0',dtype=torch.int64),clean=0.,zero=0.,recon=0.,weight=0,sequence_ce=[]) for key in trainers}
   for off in range(0,len(valid_ids),cfg['eval_batch_size_sequences']):
    ids=valid_ids[off:off+cfg['eval_batch_size_sequences']].to('cuda:0');hidden=capture(ids);x=hidden.flatten(0,1);weight=len(ids)*127
    def forward(replacement=None):
     def hfn(m,i,out):return (replacement,)+out[1:] if isinstance(out,tuple) else replacement
     handle=module.register_forward_hook(hfn) if replacement is not None else None
     try:
      result=model(ids,use_cache=False);loss=torch.nn.functional.cross_entropy(result.logits[:,:-1].reshape(-1,result.logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(len(ids),-1).mean(1);return loss
     finally:
      if handle is not None:handle.remove()
    clean=forward();zero=forward(torch.zeros_like(hidden))
    for key,tr in trainers.items():
     z=tr.ae.encode(x);recon=tr.ae.decode(z);rc=forward(recon.reshape_as(hidden));a=totals[key]
     a['sse']+=float((x-recon).square().sum());a['sum']+=x.double().sum(0);a['sq']+=float(x.double().square().sum());a['n']+=len(x);a['l0']+=int((z!=0).sum());a['counts']+=(z!=0).sum(0);a['clean']+=float(clean.mean())*weight;a['zero']+=float(zero.mean())*weight;a['recon']+=float(rc.mean())*weight;a['weight']+=weight
     a['sequence_ce']+=list(zip(clean.cpu().tolist(),rc.cpu().tolist(),zero.cpu().tolist()))
   for (o,s),a in totals.items():
    tr=trainers[(o,s)];ce={k:a[k]/a['weight'] for k in ['clean','zero','recon']};variance=a['sq']-float(a['sum'].square().sum())/a['n'];W=tr.ae.decoder.weight.T if o=='topk' else tr.ae.W_dec
    q=dict(step=step,tokens=step*bs*128,objective=o,seed=s,fve=1-a['sse']/variance,ce=ce,ce_recovered=1-(ce['recon']-ce['clean'])/(ce['zero']-ce['clean']),l0=a['l0']/a['n'],alive=int((a['counts']>0).sum()),dead=int((a['counts']==0).sum()),decoder_norm_max_error=float((W.norm(dim=1)-1).abs().max()),threshold=float(tr.ae.threshold),sequence_ce=a['sequence_ce'],validation_sequences=len(valid_ids))
    if not all(np.isfinite(q[k]) for k in ['fve','ce_recovered','l0']):raise ValueError('Nonfinite quality')
    rows.append(q)
    with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(q)+'\n')
    log('QUALITY',**{k:v for k,v in q.items() if k!='sequence_ce'})
  ts=time.perf_counter();last=ts;losses={};quality_seconds=0.;save_seconds=0.
  for step in range(steps):
   ids=torch.as_tensor(data['train'][step*bs:(step+1)*bs].astype('int64'),device='cuda:0');x=capture(ids).flatten(0,1)
   for (o,s),tr in trainers.items():
    value=tr.update(step,x)
    if not np.isfinite(value):raise ValueError(f'Nonfinite loss {o}/{s}/{step}')
    losses[f'{o}_{s}']=value
   trained=step+1
   if trained%64==0:
    torch.cuda.synchronize();now=time.perf_counter();elapsed=now-ts;per=(now-last)/64;last=now
    p=dict(completed_steps=trained,tokens=trained*bs*128,total_steps=steps,elapsed_seconds=elapsed,latest_seconds_per_step=per,projected_training_seconds=per*steps,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),losses=losses)
    write(run/'progress.json',p);log('TRAINING',**p)
    if trained==64:write(run/'throughput_pilot.json',dict(**p,includes_first_optimizer_allocation_and_initialization=True))
   stop=time.perf_counter()-ts>cfg['budget_seconds']
   if trained in cfg['checkpoint_steps'] or stop:
    t=time.perf_counter();save(trained,exact=trained==steps or stop);save_seconds+=time.perf_counter()-t
    t=time.perf_counter();quality(trained);quality_seconds+=time.perf_counter()-t;last=time.perf_counter()
   if stop and trained<steps:raise TimeoutError('Measured wall guard; inference and optimizer states retained')
  timings=dict(training_loop_wall_seconds=time.perf_counter()-ts,quality_seconds=quality_seconds,checkpoint_save_seconds=save_seconds)
  checks.update(exact_steps=trained==steps,all_quality_rows=len(rows)==len(cfg['checkpoint_steps'])*len(trainers))
  env['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
 except BaseException as exc:
  error=repr(exc);(run/'stderr.log').write_text(traceback.format_exc());log('FAIL',error=error)
 status='PASS' if error is None else ('CUT' if 'TimeoutError' in error else 'FAIL')
 summary=dict(status=status,error=error,checks=checks,completed_steps=trained,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_dictionary_training_curve.py',generator_script_sha256=codes[0]['sha256'],wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,timings=timings,scope=cfg['scope'])
 write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),error=error));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=v.errors));log('COMPLETE',summary=summary,contract_ok=v.ok,errors=v.errors)
 return 0 if status=='PASS' and v.ok else 1

if __name__=='__main__':raise SystemExit(main())
