"""Frozen-candidate choice on original CausalGym examples, using source endpoints.

Development driver: original rows, Pair/Batch alignment, labels and score remain
unchanged. It evaluates declared tasks/sites, not the original whole-layer sweep.
"""
from __future__ import annotations
import os,sys,json,time,platform,argparse,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.causalgym_interface import CausalGymModel
from ccad.native_group_selection import source_members,candidate_groups,support_matching_score,group_semantic_ot,source_path_prediction


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);cfg=json.loads(ap.parse_args().config.read_text())
 run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);started=datetime.now(timezone.utc).isoformat();start=time.perf_counter();cpu=time.process_time();write(run/'config.resolved.json',cfg)
 codes=[]
 for rel in ['scripts/run_causalgym_group_selector.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/native_group_selection.py','src/ccad/causalgym_interface.py','src/ccad/semantic_context_matching.py','src/ccad/nip_baselines.py','src/ccad/artifacts.py']:
  p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());codes.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
 write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
 write(run/'manifest.json',dict(schema_version='causalgym.group.selector.v1',run_id=cfg['run_id'],run_parent='FINAL_THREE_R19',purpose=cfg['purpose'],milestone='M4',evidence_level='controlled_selector_development',started_utc=started,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(codes),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='independent natural mean split',threshold_source_split='fixed config and original train/dev; test unavailable',statistics_unit='source/target SAE identities, task, prompt-connected component; descriptive pilot',device='cuda:0',seeds=cfg['seeds'],resource_lease='gpu-0 resource_manager.run',resource_lease_reason='Real original-benchmark source and candidate interventions; bounded natural activations'))
 for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/fn).touch()
 write(run/'status.json',dict(status='RUNNING',updated_utc=started));inputs=[];env={};checks={};rows=[];choices=[];error=None;model_api=None
 def checked(path,source='CCAD existing pinned asset',license='internal'):
  p=Path(path);p=p if p.is_absolute() else ROOT/p;inputs.append(entry(p,source,'actual input',license));return p
 def log(event,**kw):
  msg=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),event=event,elapsed_seconds=time.perf_counter()-start,**kw);write(run/'progress.json',msg)
  with (run/'stdout.log').open('a') as f:f.write(json.dumps(msg)+'\n')
  print(json.dumps(msg),flush=True)
 def record(row):
  rows.append(row)
  with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 try:
  import torch,numpy as np,transformers
  sys.path.append(cfg['dictionary_source_dir']);sys.path.append(cfg['dictionary_overlay_dir']);sys.path.append(cfg['scipy_overlay_dir'])
  from dictionary_learning.trainers.top_k import AutoEncoderTopK
  from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
  from ccad.semantic_context_matching import top_distributions,retrieve,euclidean_cost
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
  trainrun=ROOT/cfg['training_run'];traincfg=json.loads(checked(trainrun/'config.resolved.json').read_text());status=json.loads(checked(trainrun/'status.json').read_text());assert status['status']=='PASS',status
  snapshots=json.loads(checked(trainrun/'checkpoints.json').read_text())['checkpoints'];saes={};decoders={}
  for objective in cfg['objectives']:
   for seed in cfg['seeds']:
    snap=next(r for r in snapshots if r['step']==cfg['checkpoint_step'] and r['objective']==objective and r['seed']==seed);p=checked(snap['path']);assert sha256(p)==snap['sha256']
    state=torch.load(p,map_location='cuda:0',weights_only=True)
    if objective=='topk':ae=AutoEncoderTopK(1024,traincfg['dict_size'],traincfg['k'])
    else:ae=MatryoshkaBatchTopKSAE(1024,traincfg['dict_size'],traincfg['k'],state['group_sizes'].cpu().tolist())
    ae=ae.to('cuda:0');ae.load_state_dict(state);ae.eval();ae.requires_grad_(False);saes[(objective,seed)]=ae;decoders[(objective,seed)]=ae.decoder.weight.T if objective=='topk' else ae.W_dec
  for fn in ['config.json','tokenizer.json','model.safetensors']:checked(Path(traincfg['model_local_dir'])/fn,traincfg['model_id']+' '+traincfg['model_revision'],'MIT')
  tokenizer=transformers.AutoTokenizer.from_pretrained(traincfg['model_local_dir'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
  model=transformers.AutoModelForCausalLM.from_pretrained(traincfg['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.requires_grad_(False);model.config.use_cache=False
  module=model.get_submodule(traincfg['hook_module_path']);model_api=CausalGymModel(model,tokenizer,module,checked(cfg['official_data_module'],'Original CausalGym Pair/Batch','private reference source'),traincfg['layer'])
  env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),threads=2,training_run=trainrun.name,checkpoint_step=cfg['checkpoint_step'])
  # Paired-reference sampling spans the original non-audit document splits.
  pm=json.loads(checked(cfg['paired_manifest']).read_text());natural={};Z={};means={}
  class StopAtHook(Exception):pass
  def hidden_for(ids):
   cache={}
   def hook(m,i,out):cache['x']=(out[0] if isinstance(out,tuple) else out).detach();raise StopAtHook()
   handle=module.register_forward_hook(hook)
   try:
    with torch.no_grad():
     try:model(ids,use_cache=False)
     except StopAtHook:pass
   finally:handle.remove()
   return cache['x']
  for split,count in cfg['natural_tokens'].items():
   info=pm['outputs'][split];p=checked(info['path'],'Original non-audit paired prefix','ODC-By-1.0');assert sha256(p)==info['sha256'];tokens=np.fromfile(p,dtype='<u2').reshape(-1,128)
   eligible=np.flatnonzero(tokens.reshape(-1)!=tokenizer.eos_token_id);selected=eligible[np.linspace(0,len(eligible)-1,count,dtype=int)];parts=[]
   for off in range(0,len(tokens),cfg['batch_size']):
    local=selected[(selected>=off*128)&(selected<(off+cfg['batch_size'])*128)]-off*128
    if len(local):
     h=hidden_for(torch.as_tensor(tokens[off:off+cfg['batch_size']].astype('int64'),device='cuda:0')).flatten(0,1)
     parts.append(h[torch.as_tensor(local,device='cuda:0')])
   x=torch.cat(parts);assert len(x)==count;natural[split]=x
   np.savez_compressed(run/f'natural_{split}_states.npz',hidden=x.cpu().numpy(),packed_positions=selected)
   for key,ae in saes.items():
    with torch.no_grad():z=torch.cat([ae.encode(x[j:j+512]) for j in range(0,len(x),512)])
    Z[(split,*key)]=z
    if split=='mean':means[key]=z.mean(0)
    nz=z.nonzero();values=z[nz[:,0],nz[:,1]]
    np.savez_compressed(run/f'natural_{split}_{key[0]}_seed{key[1]}.npz',rows=nz[:,0].cpu().numpy().astype(np.int32),columns=nz[:,1].cpu().numpy().astype(np.uint16),values=values.cpu().numpy(),shape=np.array(z.shape))
   log('NATURAL_REFERENCE',split=split,tokens=count,saes=len(saes),cuda_bytes=torch.cuda.memory_allocated())
  # JSON parsing reads original train/dev records; only declared tasks enter
  # computation. test.json is never loaded. Do not conflate reading and use.
  original={name:json.loads(checked(Path(cfg['dataset_dir'])/(name+'.json'),'aryaman/causalgym '+cfg['dataset_revision'],'MIT').read_text()) for name in ['train','dev']}
  panel=[]
  for task in cfg['tasks']:
   fit=[dict(r,original_index=i,official_split='train',split='fit') for i,r in enumerate(original['train']) if r['task']==task][:cfg['train_rows_per_task']]
   dev=[dict(r,original_index=i,official_split='dev',split='calibration' if j<cfg['calibration_rows'] else 'held_dev') for j,(i,r) in enumerate((i,r) for i,r in enumerate(original['dev']) if r['task']==task)]
   panel+=fit+dev
  for i,r in enumerate(panel):
   r['row_id']=i;r['pair_key']=hashlib.sha256(json.dumps(sorted([''.join(r['base']),''.join(r['src'])])).encode()).hexdigest()
  write(run/'panel.json',dict(rows=panel,scope='Original records/order/labels; no deduplication or reciprocal augmentation. First declared training prefix, then original dev with fixed calibration prefix. Test untouched.'))
  # Fixed source group budgets and positions; source query never uses target.
  for task in cfg['tasks']:
   taskrows=[r for r in panel if r['task']==task];nfit=sum(r['split']=='fit' for r in taskrows);ncal=cfg['calibration_rows'];N=len(taskrows)
   if cfg['positions'][task]=='last':region=len(taskrows[0]['base'])-1
   else:region=max(j for j,(a,b) in enumerate(zip(taskrows[0]['base'],taskrows[0]['src'])) if a!=b)
   if any((max(j for j,(a,b) in enumerate(zip(r['base'],r['src'])) if a!=b) if cfg['positions'][task]=='changed' else len(r['base'])-1)!=region for r in taskrows):raise ValueError('Original task has variable selected region index')
   batches=[];base_h=[];donor_h=[];base_margin=[];raw_margin=[];base_grad=[];base_lp=[];raw_lp=[]
   for off in range(0,N,cfg['batch_size']):
    batch=model_api.batch(taskrows[off:off+cfg['batch_size']]);bp,sp=model_api.positions(batch,region);ids=torch.arange(len(batch.pairs),device='cuda:0');b=model_api.forward(batch,oracle=off==0);d=model_api.forward(batch,donor=True)
    bh=b['hidden'][ids,bp];dh=d['hidden'][ids,sp];raw=model_api.forward(batch,positions=bp,delta=dh-bh);grad=model_api.forward(batch,positions=bp,gradient=True)
    assert torch.equal(b['margin'],grad['margin']);batches.append((off,batch,bp));base_h.append(bh);donor_h.append(dh);base_margin.append(b['margin']);raw_margin.append(raw['margin']);base_grad.append(grad['gradient']);base_lp.append(b['log_probs']);raw_lp.append(raw['log_probs'])
   bh=torch.cat(base_h);dh=torch.cat(donor_h);bm=torch.cat(base_margin);rawm=torch.cat(raw_margin);gbase=torch.cat(base_grad);blp=torch.cat(base_lp);rlp=torch.cat(raw_lp)
   signs=torch.tensor([1. if r['src_type']==sorted([r['base_type'],r['src_type']])[1] else -1. for r in taskrows],device='cuda:0')
   codes_task={}
   with torch.no_grad():
    for key,ae in saes.items():codes_task[key]=(ae.encode(bh),ae.encode(dh))
   for objective in cfg['objectives']:
    for source_seed,target_seed in cfg['seed_pairs']:
     query=f'{task}_{objective}_s{source_seed}_t{target_seed}';sk=(objective,source_seed);tk=(objective,target_seed);zs0,zs1=codes_task[sk];zt0,zt1=codes_task[tk];dzs=zs1-zs0;dzt=zt1-zt0;Ds=decoders[sk];Dt=decoders[tk]
     members,contrast=source_members(dzs[:nfit],signs[:nfit],cfg['source_members'],decoder=Ds,gradients=gbase[:nfit],mode=cfg.get('source_mode','contrast'));q=dzs[:,members]@Ds[members]
     sm=[];gs=[];sl=[]
     for off,batch,bp in batches:
      r=model_api.forward(batch,positions=bp,delta=q[off:off+len(batch.pairs)],gradient=True);sm.append(r['margin']);gs.append(r['gradient']);sl.append(r['log_probs'])
     sm=torch.cat(sm);gs=torch.cat(gs);sl=torch.cat(sl)
     groups,fit_details,reader,cos,corr=candidate_groups(Z[('discovery',*sk)],Z[('discovery',*tk)],(means[sk],means[tk]),Ds,Dt,members,dzs[:nfit],dzt[:nfit],gs[:nfit],budget=cfg['target_members'],pool_size=cfg['fit_pool'])
     # All candidate weights and scores are fixed before held-dev candidate execution.
     family=sorted(groups);weights=torch.stack([groups[name] for name in family]);np.savez_compressed(run/(query+'_groups.npz'),weights=weights.cpu().numpy(),source_members=members.cpu().numpy(),source_contrast=contrast.cpu().numpy(),candidate_names=np.array(family),reader_support=reader['support'].cpu().numpy(),reader_matrix=reader['reader'].cpu().numpy())
     write(run/(query+'_fit.json'),fit_details)
     sc=Z[('calibration',*sk)][:,members]-means[sk][members];desired_nat=sc@Ds[members];tc=Z[('calibration',*tk)]-means[tk];cal=slice(nfit,nfit+ncal);held=slice(nfit+ncal,N)
     predictions={};selection_scores={name:{} for name in ['cosine','pw_mcc','semantic_ot_group','natural_mse','base_linear','anchored_base_linear','source_endpoint','source_path','finite_budget','finite_margin']};candidate_stats={};cal_actual={};deltas={}
     natref=natural['calibration'].cpu().numpy();source_activity=Z[('calibration',*sk)][:,members].sum(1).cpu().numpy()
     tscore=time.perf_counter()
     for name,w in groups.items():
      qt=(dzt*w)@Dt;error_vector=qt-q;deltas[name]=qt
      pred=dict(base_linear=bm+(gbase*qt).sum(1),anchored_base_linear=sm+(gbase*error_vector).sum(1),source_endpoint=sm+(gs*error_vector).sum(1),source_path=source_path_prediction(bm,sm,gbase,gs,q,qt))
      predictions[name]=pred
      for selector in pred:selection_scores[selector][name]=float(torch.sigmoid(pred[selector][cal]).mean())
      selection_scores['cosine'][name]=support_matching_score(cos,w)
      selection_scores['pw_mcc'][name]=support_matching_score(corr.abs(),w)
      nm=float((((tc*w)@Dt-desired_nat).square().sum(1)).mean());selection_scores['natural_mse'][name]=-nm
      distance,diag=group_semantic_ot(source_activity,(Z[('calibration',*tk)]@w).cpu().numpy(),natref)
      selection_scores['semantic_ot_group'][name]=-distance
      candidate_stats[name]=dict(members=int((w>1e-7).sum()),l1=float(w.sum()),min_weight=float(w.min()),max_weight=float(w.max()),natural_mse=nm,semantic_ot=diag,mean_delta_norm=float(qt.norm(dim=1).mean()),relative_realization_l2=float(error_vector.square().sum()/q.square().sum().clamp_min(1e-10)))
      if not (bool((w>=0).all()) and bool((w<=1).all())):raise ValueError('Non-native participation weight')
     score_seconds=time.perf_counter()-tscore
     # Candidate calibration outputs are used only by the explicit finite budget
     # selector, and for diagnostic validation of the other predictors.
     per_candidate=max(1,min(ncal,cfg['finite_budget_total_sequences']//len(family)))
     selected_cal_indices=np.arange(nfit,nfit+per_candidate)
     for name in family:
      values=[]
      for off in range(nfit,nfit+ncal,cfg['batch_size']):
       batch=model_api.batch(taskrows[off:min(off+cfg['batch_size'],nfit+ncal)]);bp,_=model_api.positions(batch,region);v=model_api.forward(batch,positions=bp,delta=deltas[name][off:off+len(batch.pairs)]);values+=v['margin'].cpu().tolist()
      cal_actual[name]=np.asarray(values);selection_scores['finite_budget'][name]=float(np.mean(cal_actual[name][:per_candidate]>0))
      selection_scores['finite_margin'][name]=float(np.mean(1/(1+np.exp(-cal_actual[name][:per_candidate]))))
     selected={selector:max(family,key=lambda n:(scores[n],-family.index(n))) for selector,scores in selection_scores.items()}
     choice=dict(query=query,task=task,objective=objective,source_seed=source_seed,target_seed=target_seed,source_members=members.cpu().tolist(),region=region,selected=selected,scores=selection_scores,candidate_stats=candidate_stats,finite_budget_sequences_per_candidate=per_candidate,finite_budget_total_sequences=per_candidate*len(family),calibration_rows=ncal,score_computation_seconds=score_seconds,source_calibration_forward_sequences=ncal,source_calibration_backward_sequences=ncal,source_train_forward_sequences=nfit,source_train_backward_sequences=nfit,written_before_held_candidate_execution_utc=datetime.now(timezone.utc).isoformat())
     choices.append(choice);write(run/'selection_choices.json',dict(choices=choices))
     # Held-dev effects are materialized only after selector choices were saved.
     actual={name:[] for name in family+['ridge_reader','full_target','source','raw']}
     for name in family+['ridge_reader','full_target']:
      qt=deltas[name] if name in deltas else dzt[:,reader['support']]@reader['reader'] if name=='ridge_reader' else dzt@Dt
      for off in range(nfit+ncal,N,cfg['batch_size']):
       current=taskrows[off:off+cfg['batch_size']];batch=model_api.batch(current);bp,_=model_api.positions(batch,region);v=model_api.forward(batch,positions=bp,delta=qt[off:off+len(current)])
       kl=(sl[off:off+len(current)].exp()*(sl[off:off+len(current)]-v['log_probs'])).sum(1)
       for j,r in enumerate(current):
        i=off+j;actual[name].append(float(v['margin'][j]));record(dict(query=query,task=task,objective=objective,source_seed=source_seed,target_seed=target_seed,method=name,row_id=r['row_id'],official_index=r['original_index'],official_split=r['official_split'],split='held_dev',pair_key=r['pair_key'],base_correct=bool(bm[i]<0),source_margin=float(sm[i]),source_iia=bool(sm[i]>0),base_margin=float(bm[i]),margin=float(v['margin'][j]),iia=bool(v['margin'][j]>0),log_odds_ratio=float(v['margin'][j]-bm[i]),source_kl=float(kl[j]),source_delta_norm=float(q[i].norm()),target_delta_norm=float(qt[i].norm()),predictions={k:float(x[i]) for k,x in predictions[name].items()} if name in predictions else {},members=candidate_stats[name]['members'] if name in candidate_stats else None))
     for name,margin in [('source',sm),('raw',rawm)]:
      for i in range(nfit+ncal,N):
       r=taskrows[i];actual[name].append(float(margin[i]));record(dict(query=query,task=task,objective=objective,source_seed=source_seed,target_seed=target_seed,method=name,row_id=r['row_id'],official_index=r['original_index'],official_split='dev',split='held_dev',pair_key=r['pair_key'],base_correct=bool(bm[i]<0),base_margin=float(bm[i]),source_margin=float(sm[i]),source_iia=bool(sm[i]>0),margin=float(margin[i]),iia=bool(margin[i]>0),log_odds_ratio=float(margin[i]-bm[i])))
     held_summary={name:dict(iia=float(np.mean(np.asarray(values)>0)),margin=float(np.mean(values))) for name,values in actual.items()}
     choice['held_summary']=held_summary;choice['selector_held_iia']={s:held_summary[name]['iia'] for s,name in selected.items()};choice['source_held_iia']=held_summary['source']['iia'];choice['oracle_native_held_iia']=max(held_summary[n]['iia'] for n in family)
     write(run/'selection_choices.json',dict(choices=choices));np.savez_compressed(run/(query+'_calibration.npz'),names=np.array(family),actual_margin=np.stack([cal_actual[n] for n in family]),source_margin=sm[cal].cpu().numpy(),predicted_endpoint=np.stack([predictions[n]['source_endpoint'][cal].cpu().numpy() for n in family]),predicted_base=np.stack([predictions[n]['base_linear'][cal].cpu().numpy() for n in family]),base_margin=bm[cal].cpu().numpy())
     log('QUERY_COMPLETE',query=query,source_iia=choice['source_held_iia'],oracle_native_iia=choice['oracle_native_held_iia'],selector_iia=choice['selector_held_iia'])
     if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Development guard; completed query artifacts retained')
  checks.update(official_hook=model_api.checks,completed_queries=len(choices),all_native_weights_feasible=True,test_read=False)
  env.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(),forward_sequences=model_api.sequences,forward_tokens=model_api.tokens,backward_sequences=model_api.backward_sequences)
 except BaseException as exc:
  error=repr(exc);(run/'stderr.log').write_text(traceback.format_exc());log('FAIL',error=error)
 status='PASS' if error is None else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(rows),completed_queries=len(choices),wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_causalgym_group_selector.py',generator_script_sha256=codes[0]['sha256'],scope=cfg['scope'])
 write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),error=error));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=v.errors));log('COMPLETE',summary=summary,contract_ok=v.ok,errors=v.errors)
 return 0 if status=='PASS' and v.ok else 1

if __name__=='__main__':raise SystemExit(main())
