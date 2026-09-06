"""Two-seed fixed-stream training checkpoints using the unchanged Sparsify trainer."""
import argparse,json,os,sys,time,traceback,platform
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',WANDB_DISABLED='true',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r006b_topk_capacity import ROOT,MemmapTokens,attach_model_trace,attach_loss_trace,evaluate,state_hash,canonicalize_sparsify_multiseed_state,save_sparsify_exact_state,HookPointContract
from run_r011s1_raw_hook_asset import entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg);inputs=[];code=[];rows=[];snaps=[];checks={};error=None;env={}
 for rel in ['scripts/run_training_checkpoint_curve.py','scripts/run_r006b_topk_capacity.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/checkpointing.py','src/ccad/artifacts.py','src/ccad/activation_contract.py','src/ccad/sae_quality.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='training.curve.v1',run_id=cfg['run_id'],run_parent='R011-NR1',purpose=cfg['purpose'],milestone='C3',evidence_level='two_seed_same_stream_learning_curve_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='fixed validation mean for quality only',threshold_source_split='checkpoint steps fixed before training',statistics_unit='two init seeds and fixed validation sequences, checkpoints dependent',device=cfg['device'],seeds=cfg['init_seeds'],resource_lease='gpu-0 resource_manager.run',resource_lease_reason='GPU training/inference; four host threads; bounded checkpoint IO'))
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
  import torch,transformers
  from sparsify import SaeConfig,TrainConfig,Trainer,SparseCoder
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');torch.cuda.reset_peak_memory_stats();device=torch.device(cfg['device'])
  model=transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(device);model.config.use_cache=False
  td=MemmapTokens(train,cfg['context_length'],cfg['max_train_sequences']);vd=MemmapTokens(valid,cfg['context_length'],cfg['max_validation_sequences']);expected=len(td)//cfg['batch_size_sequences'];assert expected==max(cfg['checkpoint_steps'])
  tc=TrainConfig(sae=SaeConfig(activation='topk',num_latents=cfg['num_latents'],k=cfg['k'],normalize_decoder=True),batch_size=cfg['batch_size_sequences'],optimizer='adam',lr=cfg['learning_rate'],lr_warmup_steps=cfg['lr_warmup_steps'],hookpoints=[cfg['hookpoint']],init_seeds=cfg['init_seeds'],save_every=expected+2,dead_feature_threshold=cfg['dead_feature_threshold'],auxk_alpha=cfg['auxk_alpha'],exclude_tokens=cfg['exclude_token_ids'],save_best=False,log_to_wandb=False,run_name=cfg['run_id'],save_dir=cfg['bulk_output_dir']+'/upstream_unused')
  trainer=Trainer(tc,td,model);canonicalize_sparsify_multiseed_state(trainer);assert len(trainer.optimizers)==1 and trainer.cfg.grad_acc_steps==1
  bulk=Path(cfg['bulk_output_dir']);bulk.mkdir(parents=True,exist_ok=False)
  # Avoid a redundant final upstream copy; exact final continuation state is saved below.
  trainer.save=lambda:None
  attach_model_trace(model);traces,handles=attach_loss_trace(trainer);model.ccad_input_hashes=[];updates=0;checkpoint_seconds=0.
  def checkpoint_hook(opt,args,kwargs):
   nonlocal updates,checkpoint_seconds
   updates+=1
   if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Training unit wall budget exceeded')
   if updates not in cfg['checkpoint_steps']:return
   t=time.perf_counter()
   for name,sae in trainer.saes.items():
    seed=int(name.rsplit('seed',1)[1]);dest=bulk/f'step_{updates}'/f'seed_{seed}';sae.save_to_disk(dest)
    snaps.append(dict(step=updates,seed=seed,sae_name=name,train_sequences=updates*cfg['batch_size_sequences'],packed_train_tokens=updates*cfg['batch_size_sequences']*cfg['context_length'],path=str(dest),sha256=sha256(dest/'sae.safetensors'),state_hash=state_hash(sae.state_dict()),learning_rate_used=float(opt.param_groups[cfg['init_seeds'].index(seed)]['lr']),captured_after='optimizer update, before scheduler/dead-counter update; inference weights only',seconds_since_start=time.perf_counter()-start))
   checkpoint_seconds+=time.perf_counter()-t;write(run/'checkpoints.json',dict(checkpoints=snaps));print(json.dumps(dict(stage='CHECKPOINT',updates=updates,tokens=updates*512,elapsed=time.perf_counter()-start)),flush=True)
  handle=trainer.optimizers[0].register_step_post_hook(checkpoint_hook);ts=time.perf_counter();trainer.fit();torch.cuda.synchronize();train_seconds=time.perf_counter()-ts;handle.remove()
  for h in handles:h.remove()
  training_hashes=list(model.ccad_input_hashes);write(run/'training_trace.json',dict(traces=traces,input_hashes=training_hashes,train_seconds=train_seconds,checkpoint_save_seconds=checkpoint_seconds,updates=updates))
  checks.update(updates_exact=updates==expected==trainer.global_step,one_base_forward_per_step=len(training_hashes)==expected,trace_complete=all(len(v)==expected for v in traces.values()),all_checkpoints=len(snaps)==len(cfg['init_seeds'])*len(cfg['checkpoint_steps']))
  exact=save_sparsify_exact_state(trainer,bulk/'exact_final',data_cursor_examples=len(td));write(run/'exact_final.json',dict(path=str(bulk/'exact_final'),metadata=exact))
  for name,sae in trainer.saes.items():
   seed=int(name.rsplit('seed',1)[1]);snap=next(s for s in snaps if s['step']==expected and s['seed']==seed);assert state_hash(sae.state_dict())==snap['state_hash']
  contract=HookPointContract(cfg['hook_module_path'],5,'resid_post',768)
  for snap in snaps:
   sae=SparseCoder.load_from_disk(snap['path'],device=cfg['device']).eval();assert state_hash(sae.state_dict())==snap['state_hash'];q=evaluate(model,sae,model.get_submodule(cfg['hook_module_path']),contract,cfg['hook_oracle_hidden_state_index'],vd,cfg['eval_batch_size_sequences'],device,torch)
   row=dict(**snap,quality=q,decoder_norm_max_error=float((sae.W_dec.norm(dim=1)-1).abs().max()));rows.append(row)
   with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(dict(stage='QUALITY',step=snap['step'],seed=snap['seed'],fve=q['fve'],ce_recovered=q['ce_recovered'])),flush=True);del sae
  checks.update(quality_complete=len(rows)==len(snaps),quality_oracle=all(r['quality']['hook_oracle_max_error']==0 and r['quality']['capture_logit_max_error']==0 for r in rows),same_validation_baselines=max(r['quality']['ce']['clean'] for r in rows)==min(r['quality']['ce']['clean'] for r in rows),finite=all(__import__('math').isfinite(r['quality']['fve']) and __import__('math').isfinite(r['quality']['ce_recovered']) for r in rows))
  env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated())
 except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
 write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);status='PASS' if error is None and checks and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(rows),wall_seconds=time.perf_counter()-start,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope=cfg['scope_limit']);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));write(run/'stdout.log',summary);v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True);return 0 if status=='PASS' and v.ok else 1
if __name__=='__main__':raise SystemExit(main())
