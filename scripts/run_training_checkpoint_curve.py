"""Fixed-stream training checkpoints using the unchanged Sparsify trainer."""
import argparse,json,os,sys,time,traceback,platform
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',WANDB_DISABLED='true',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r006b_topk_capacity import ROOT,MemmapTokens,attach_model_trace,attach_loss_trace,evaluate,state_hash,canonicalize_sparsify_multiseed_state,save_sparsify_exact_state,HookPointContract
from run_r011s1_raw_hook_asset import entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.training_progress import completed_step_callback
from ccad.checkpointing import load_sparsify_exact_state

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg);inputs=[];code=[];rows=[];snaps=[];checks={};error=None;env={}
 for rel in ['scripts/run_training_checkpoint_curve.py','scripts/run_r006b_topk_capacity.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/checkpointing.py','src/ccad/training_progress.py','src/ccad/artifacts.py','src/ccad/activation_contract.py','src/ccad/sae_quality.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='training.curve.v1',run_id=cfg['run_id'],run_parent=cfg.get('run_parent','R011-NR1'),purpose=cfg['purpose'],milestone='C3',evidence_level=cfg.get('evidence_level','same_stream_learning_curve_development'),started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='fixed validation mean for quality only',threshold_source_split='checkpoint steps fixed before training',statistics_unit='init seeds and fixed validation sequences, checkpoints dependent',device=cfg['device'],seeds=cfg['init_seeds'],resource_lease='gpu-0 resource_manager.run',resource_lease_reason='GPU training/inference; four host threads; bounded checkpoint IO'))
 for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
 write(run/'status.json',dict(status='RUNNING'))
 def checked(path,digest=None):
  p=Path(path);p=p if p.is_absolute() else ROOT/p;r=entry(p,'existing locked training assets','input')
  if digest:assert r['sha256']==digest
  inputs.append(r);return p
 try:
  checked(args.config);checked('.aris/compute/local-r006b1-env-spec.json');checked(cfg['environment_lock']);checked(cfg['token_manifest_path'])
  train=checked(cfg['train_token_path'],cfg['train_token_sha256']);valid=checked(cfg['validation_token_path'],cfg['validation_token_sha256']);docs=json.loads(checked(cfg['document_records']).read_text())['documents']
  checks['document_disjoint']=all(not({r[k] for r in docs if r['split']=='train'}&{r[k] for r in docs if r['split']=='validation'}) for k in ['document_id','text_sha256']);assert checks['document_disjoint']
  checked(Path(cfg['sparsify_source_dir'])/'sparsify/trainer.py');checked(Path(cfg['sparsify_source_dir'])/'sparsify/sparse_coder.py')
  model_dir=Path(cfg['model_local_dir'])
  for fn in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
   checked(model_dir/fn)
  weight=model_dir/'model.safetensors'
  if weight.exists():checked(weight,cfg.get('model_weights_sha256'))
  else:checked(model_dir/'pytorch_model.bin',cfg.get('model_weights_sha256'))
  if cfg.get('compatible_tokenizer_dir'):
   checks['tokenizer_identity']=all(sha256(model_dir/fn)==sha256(Path(cfg['compatible_tokenizer_dir'])/fn) for fn in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']);assert checks['tokenizer_identity']
  if cfg.get('cuda_allocator_conf'):os.environ['PYTORCH_CUDA_ALLOC_CONF']=cfg['cuda_allocator_conf']
  import torch,transformers
  from sparsify import SaeConfig,TrainConfig,Trainer,SparseCoder
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');torch.cuda.reset_peak_memory_stats();device=torch.device(cfg['device'])
  model=transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(device);model.config.use_cache=False
  contract=HookPointContract(cfg['hook_module_path'],cfg['layer_index'],'resid_post',model.config.hidden_size)
  assert cfg['hook_oracle_hidden_state_index']==cfg['layer_index']+1
  env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),matmul_precision=torch.get_float32_matmul_precision(),hidden_size=model.config.hidden_size,layer_index=cfg['layer_index'],cuda_allocator_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
  def seed_for(name):
   return int(name.rsplit('/seed',1)[1]) if '/seed' in name else cfg['init_seeds'][0]
  td=MemmapTokens(train,cfg['context_length'],cfg['max_train_sequences']);vd=MemmapTokens(valid,cfg['context_length'],cfg['max_validation_sequences']);expected=len(td)//cfg['batch_size_sequences'];assert expected==max(cfg['checkpoint_steps'])
  tc=TrainConfig(sae=SaeConfig(activation='topk',num_latents=cfg['num_latents'],k=cfg['k'],normalize_decoder=True),batch_size=cfg['batch_size_sequences'],optimizer='adam',lr=cfg['learning_rate'],lr_warmup_steps=cfg['lr_warmup_steps'],hookpoints=[cfg['hookpoint']],init_seeds=cfg['init_seeds'],save_every=expected+2,dead_feature_threshold=cfg['dead_feature_threshold'],auxk_alpha=cfg['auxk_alpha'],exclude_tokens=cfg['exclude_token_ids'],save_best=False,log_to_wandb=False,run_name=cfg['run_id'],save_dir=cfg['bulk_output_dir']+'/upstream_unused')
  trainer=Trainer(tc,td,model);canonicalize_sparsify_multiseed_state(trainer);assert len(trainer.optimizers)==1 and trainer.cfg.grad_acc_steps==1
  bulk=Path(cfg['bulk_output_dir']);bulk.mkdir(parents=True,exist_ok=False)
  prior_trace=dict(traces={},input_hashes=[])
  if cfg.get('resume_final_run'):
   if cfg.get('resume_run'):raise ValueError('Choose either interruption recovery or a new training phase')
   parent=ROOT/cfg['resume_final_run'];resume=json.loads(checked(parent/'exact_final.json').read_text());directory=Path(resume['path'])
   previous=json.loads(checked(parent/'training_trace.json').read_text())
   prior_trace=dict(traces=previous['traces'],input_hashes=previous['input_hashes'],updates=previous['updates'])
   for p in directory.rglob('*'):
    if p.is_file():checked(p)
   load_sparsify_exact_state(trainer,directory,expected_data_cursor_examples=prior_trace['updates']*cfg['batch_size_sequences'])
   assert trainer.global_step==prior_trace['updates']<expected
   snaps=json.loads(checked(parent/'checkpoints.json').read_text())['checkpoints']
   checks['continuation_initial_weights_equal']=all(state_hash(sae.state_dict())==next(s['state_hash'] for s in snaps if s['step']==trainer.global_step and s['seed']==seed_for(name)) for name,sae in trainer.saes.items())
   assert checks['continuation_initial_weights_equal']
   # The previous phase finished at zero LR. This is a declared continuation
   # with a new schedule, not exact replay of the old optimization trajectory.
   phase_lr=float(cfg['continuation_learning_rate']);phase_warmup=int(cfg['continuation_warmup_steps'])
   assert phase_lr>0 and 0<=phase_warmup<expected-trainer.global_step
   for opt in trainer.optimizers:
    for group in opt.param_groups:group['lr']=phase_lr;group['initial_lr']=phase_lr
   from transformers import get_linear_schedule_with_warmup
   trainer.lr_schedulers=[get_linear_schedule_with_warmup(trainer.optimizers[0],phase_warmup,expected-trainer.global_step)]
   write(run/'continuation_phase.json',dict(parent=cfg['resume_final_run'],start_step=trainer.global_step,total_steps=expected,additional_steps=expected-trainer.global_step,
        learning_rate=phase_lr,warmup_steps=phase_warmup,optimizer_moments_retained=True,sae_weights_counters_rng_restored=True,
        scope='Declared fresh-data training continuation with restarted LR schedule; not uninterrupted original training.'))
  if cfg.get('resume_run'):
   parent=ROOT/cfg['resume_run'];resume=json.loads(checked(parent/'resumable_state.json').read_text());directory=Path(resume['path'])
   prior_trace=json.loads(checked(parent/'partial_training_trace.json').read_text())
   for p in directory.rglob('*'):
    if p.is_file():checked(p)
   load_sparsify_exact_state(trainer,directory,expected_data_cursor_examples=prior_trace['updates']*cfg['batch_size_sequences'])
   assert trainer.global_step==prior_trace['updates']
   if (parent/'checkpoints.json').exists():snaps=json.loads(checked(parent/'checkpoints.json').read_text())['checkpoints']
  # Avoid a redundant final upstream copy; exact final continuation state is saved below.
  trainer.save=lambda:None
  attach_model_trace(model);traces,handles=attach_loss_trace(trainer);model.ccad_input_hashes=[];updates=trainer.global_step;checkpoint_seconds=0.
  def checkpoint_hook(opt,args,kwargs):
   nonlocal updates,checkpoint_seconds
   updates+=1
   if not cfg.get('resource_guard') and time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Training unit wall budget exceeded')
   if updates not in cfg['checkpoint_steps']:return
   t=time.perf_counter()
   for name,sae in trainer.saes.items():
    seed=seed_for(name);dest=bulk/f'step_{updates}'/f'seed_{seed}';sae.save_to_disk(dest)
    snaps.append(dict(step=updates,seed=seed,sae_name=name,train_sequences=updates*cfg['batch_size_sequences'],packed_train_tokens=updates*cfg['batch_size_sequences']*cfg['context_length'],path=str(dest),sha256=sha256(dest/'sae.safetensors'),state_hash=state_hash(sae.state_dict()),learning_rate_used=float(opt.param_groups[cfg['init_seeds'].index(seed)]['lr']),captured_after='optimizer update, before scheduler/dead-counter update; inference weights only',seconds_since_start=time.perf_counter()-start))
   checkpoint_seconds+=time.perf_counter()-t;write(run/'checkpoints.json',dict(checkpoints=snaps));write(run/'progress.json',dict(stage='CHECKPOINT',updates=updates,tokens=updates*cfg['batch_size_sequences']*cfg['context_length'],elapsed=time.perf_counter()-start));print(json.dumps(dict(stage='CHECKPOINT',updates=updates,tokens=updates*cfg['batch_size_sequences']*cfg['context_length'],elapsed=time.perf_counter()-start)),flush=True)
  def combined_trace():
   return dict(traces={name:prior_trace['traces'].get(name,[])+value for name,value in traces.items()},input_hashes=prior_trace['input_hashes']+list(model.ccad_input_hashes),updates=updates)
  last_progress=[updates,time.perf_counter()];slow_windows=[0]
  def after_complete_step():
   nonlocal checkpoint_seconds
   assert trainer.global_step==updates
   if updates%64 and updates!=expected:return
   elapsed=time.perf_counter()-start;seconds_per_update=(time.perf_counter()-last_progress[1])/max(updates-last_progress[0],1);last_progress[:]=[updates,time.perf_counter()]
   free,total=torch.cuda.mem_get_info();stats=torch.cuda.memory_stats();progress=dict(stage='TRAINING',updates=updates,tokens=updates*cfg['batch_size_sequences']*cfg['context_length'],elapsed=elapsed,seconds_per_update=seconds_per_update,cuda_free_bytes=free,cuda_total_bytes=total,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),inactive_split_bytes=stats.get('inactive_split_bytes.all.current'))
   if cfg.get('cache_release_threshold_bytes') and torch.cuda.memory_reserved()>cfg['cache_release_threshold_bytes']:
    torch.cuda.empty_cache();progress.update(cache_released=True,cuda_reserved_after_release_bytes=torch.cuda.memory_reserved(),cuda_allocated_after_release_bytes=torch.cuda.memory_allocated())
   write(run/'progress.json',progress)
   with (run/'resource_trace.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
   slow_windows[0]=slow_windows[0]+1 if seconds_per_update>cfg.get('max_seconds_per_update',float('inf')) else 0
   stop=updates<expected and (elapsed>cfg.get('graceful_budget_seconds',cfg['budget_seconds']-90) or slow_windows[0]>=3)
   if updates in cfg.get('exact_checkpoint_steps',[]) or stop:
    t=time.perf_counter();directory=bulk/f'exact_step_{updates}';exact=save_sparsify_exact_state(trainer,directory,data_cursor_examples=updates*cfg['batch_size_sequences']);checkpoint_seconds+=time.perf_counter()-t
    write(run/'partial_training_trace.json',combined_trace());write(run/'resumable_state.json',dict(path=str(directory),metadata=exact,complete_update=True));print(json.dumps(dict(stage='RESUMABLE_CHECKPOINT',updates=updates,elapsed=time.perf_counter()-start)),flush=True)
   if stop:raise TimeoutError('Resource guard saved exact continuation before stopping')
  handle=trainer.optimizers[0].register_step_post_hook(checkpoint_hook);ts=time.perf_counter()
  if cfg.get('resource_guard'):
   with completed_step_callback(after_complete_step):trainer.fit()
  else:trainer.fit()
  torch.cuda.synchronize();train_seconds=time.perf_counter()-ts;handle.remove()
  for h in handles:h.remove()
  combined=combined_trace();traces=combined['traces'];training_hashes=combined['input_hashes'];write(run/'training_trace.json',dict(traces=traces,input_hashes=training_hashes,train_seconds=train_seconds,checkpoint_save_seconds=checkpoint_seconds,updates=updates,resume_run=cfg.get('resume_run')))
  checks.update(updates_exact=updates==expected==trainer.global_step,one_base_forward_per_step=len(training_hashes)==expected,trace_complete=all(len(v)==expected for v in traces.values()),all_checkpoints=len(snaps)==len(cfg['init_seeds'])*len(cfg['checkpoint_steps']))
  exact=save_sparsify_exact_state(trainer,bulk/'exact_final',data_cursor_examples=len(td));write(run/'exact_final.json',dict(path=str(bulk/'exact_final'),metadata=exact))
  for name,sae in trainer.saes.items():
   seed=seed_for(name);snap=next(s for s in snaps if s['step']==expected and s['seed']==seed);assert state_hash(sae.state_dict())==snap['state_hash']
  for snap in snaps:
   sae=SparseCoder.load_from_disk(snap['path'],device=cfg['device']).eval();assert state_hash(sae.state_dict())==snap['state_hash'];q=evaluate(model,sae,model.get_submodule(cfg['hook_module_path']),contract,cfg['hook_oracle_hidden_state_index'],vd,cfg['eval_batch_size_sequences'],device,torch)
   row=dict(**snap,quality=q,decoder_norm_max_error=float((sae.W_dec.norm(dim=1)-1).abs().max()));rows.append(row)
   with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(dict(stage='QUALITY',step=snap['step'],seed=snap['seed'],fve=q['fve'],ce_recovered=q['ce_recovered'])),flush=True);del sae
  checks.update(quality_complete=len(rows)==len(snaps),quality_oracle=all(r['quality']['hook_oracle_max_error']==0 and r['quality']['capture_logit_max_error']==0 for r in rows),same_validation_baselines=max(r['quality']['ce']['clean'] for r in rows)==min(r['quality']['ce']['clean'] for r in rows),finite=all(__import__('math').isfinite(r['quality']['fve']) and __import__('math').isfinite(r['quality']['ce_recovered']) for r in rows))
  env.update(peak_vram_bytes=torch.cuda.max_memory_allocated())
 except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
 write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);status='PASS' if error is None and checks and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(rows),wall_seconds=time.perf_counter()-start,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope=cfg['scope_limit']);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));write(run/'stdout.log',summary);v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True);return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
