"""Natural-reference SemanticOT retrieval for frozen external source members."""
import argparse
import gc
import json
import os
import platform
import runpy
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.semantic_context_matching import top_distributions,retrieve,euclidean_cost


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    started=time.perf_counter();cpu_start=time.process_time();inputs=[];code=[];rows=[];checks={};error=None
    write(run/'config.resolved.json',cfg)
    for rel in ['scripts/prepare_semantic_context_matches.py','src/ccad/semantic_context_matching.py',
                'src/ccad/nip_baselines.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
        p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
        code.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha256(p),snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='natural.semantic.retrieval.v1',run_id=cfg['run_id'],run_parent='FINAL_FIVE_R14',
          purpose=cfg['purpose'],milestone='external_native_closest_method',evidence_level='controlled_development',
          started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),
          config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,
          audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='Uncentered empirical activation-weighted reference distributions; no fitted centering',
          threshold_source_split='Fixed source IDs from development; natural reference only for matching',
          statistics_unit='reference contexts and source members; not independent task or seed replications',device='cpu',
          seeds=[1,2],resource_lease='cpu-heavy',resource_lease_reason=cfg['budget']))
    for name in ['metrics.raw.jsonl','stdout.log','stderr.log']:(run/name).touch()
    write(run/'status.json',dict(status='RUNNING'))
    def checked(path,digest=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        record=entry(p,'Pinned existing public asset','actual input')
        if digest and record['sha256']!=digest:raise ValueError('Identity changed: '+str(p))
        inputs.append(record);write(run/'inputs.json',dict(inputs=inputs));return p
    def progress(stage,**values):
        if time.perf_counter()-started>cfg['budget_seconds']:raise TimeoutError('Natural-reference matching budget exhausted')
        msg=dict(stage=stage,written_at_utc=datetime.now(timezone.utc).isoformat(),elapsed=time.perf_counter()-started,**values)
        write(run/'progress.json',msg);print(json.dumps(msg),flush=True)
    try:
        import torch,transformers,psutil
        from safetensors import safe_open
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
              torch=torch.__version__,transformers=transformers.__version__,device='cpu',threads=4,
              initial_available_ram=psutil.virtual_memory().available))
        checked(args.config)
        parent=ROOT/cfg['source_run'];parent_cfg=json.loads(checked(parent/'config.resolved.json').read_text())
        if json.loads(checked(parent/'status.json').read_text())['status']!='PASS':raise ValueError('Source producer did not complete')
        fits=json.loads(checked(parent/'fit_diagnostics.json').read_text())['rows']
        source_ids=sorted({member for row in fits if 'fits' in row for member in row['source_members']})
        path=checked(cfg['reference_tokens'],cfg['reference_tokens_sha256'])
        tokens=np.memmap(path,dtype='<u2',mode='r').reshape(-1,cfg['context_length'])[:cfg['reference_sequences']].astype(np.int64)
        cache=ROOT/cfg['reference_cache_run'] if cfg.get('reference_cache_run') else None
        if cache is None:
            modelpath=Path(parent_cfg['model_local_dir'])
            checked(modelpath/'config.json');checked(modelpath/'model.safetensors')
            model=transformers.AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
            for p in model.parameters():p.requires_grad_(False)
            module=model.gpt_neox.layers[parent_cfg['layer']]
            class Captured(Exception):pass
            captured=[]
            stop_after_hook=True
            def capture(mod,args,output):
                value=output[0] if isinstance(output,tuple) else output
                captured.append(value.detach().cpu().numpy().copy())
                if stop_after_hook:raise Captured()
            handle=module.register_forward_hook(capture)
            hidden=[]
            try:
                for offset in range(0,len(tokens),cfg['batch_size']):
                    batch=torch.as_tensor(tokens[offset:offset+cfg['batch_size']])
                    captured.clear()
                    with torch.no_grad():
                        try:model.gpt_neox(input_ids=batch,use_cache=False)
                        except Captured:pass
                    if len(captured)!=1:raise ValueError('Reference hook did not fire once')
                    current=captured[0]
                    if offset==0:
                        stop_after_hook=False;captured.clear()
                        with torch.no_grad():model.gpt_neox(input_ids=batch,use_cache=False)
                        checks['early_stop_matches_full_model']=bool(np.array_equal(current,captured[0]));stop_after_hook=True
                        if not checks['early_stop_matches_full_model']:raise ValueError('Reference prefix capture differs from full model')
                    hidden.append(current)
                    progress('REFERENCE_CAPTURE',sequences=min(offset+len(batch),len(tokens)),total_sequences=len(tokens),rss=psutil.Process().memory_info().rss)
            finally:handle.remove()
            reference=np.concatenate(hidden).reshape(-1,model.config.hidden_size)
            token_ids=tokens.reshape(-1);eligible=~np.isin(token_ids,cfg['exclude_token_ids'])
            reference=reference[eligible];flat_positions=np.flatnonzero(eligible)
            np.savez_compressed(run/'reference_states.npz',hidden=reference,tokens=token_ids[eligible],packed_positions=flat_positions)
            del hidden,model;gc.collect()
        else:
            cache_cfg=json.loads(checked(cache/'config.resolved.json').read_text())
            for key in ['source_run','reference_tokens_sha256','reference_sequences','context_length','exclude_token_ids']:
                if cfg[key]!=cache_cfg[key]:raise ValueError('Reference cache configuration differs: '+key)
            prior=json.loads(checked(cache/'metrics.summary.json').read_text())
            if not prior['checks'].get('early_stop_matches_full_model'):raise ValueError('Reference full-model witness unavailable')
            reference=np.load(checked(cache/'reference_states.npz'))['hidden']
            checks['early_stop_matches_full_model']=True
            progress('REFERENCE_CACHE_REUSED',reference_run=cfg['reference_cache_run'],tokens=len(reference),prior_run_status=prior['status'])
        kernel_path=checked(Path(parent_cfg['sparsify_source'])/'sparsify/fused_encoder.py')
        checked(Path(parent_cfg['sparsify_source'])/'sparsify/sparse_coder.py')
        kernel=runpy.run_path(str(kernel_path))['fused_encoder']
        distributions={};centroids={};counts={}
        for seed in [1,2]:
            path=Path(parent_cfg['sae_root'])/f'seed_{seed}'
            saecfg=json.loads(checked(path/'cfg.json').read_text());checked(path/'sae.safetensors')
            if cache is None:
                with safe_open(path/'sae.safetensors',framework='pt',device='cpu') as f:
                    w=f.get_tensor('encoder.weight');b=f.get_tensor('encoder.bias');bd=f.get_tensor('b_dec')
                all_indices=[];all_values=[]
                with torch.no_grad():
                    for offset in range(0,len(reference),cfg['encode_batch_tokens']):
                        x=torch.as_tensor(reference[offset:offset+cfg['encode_batch_tokens']])-bd
                        act,ix,_=kernel(x,w,b,saecfg['k'],saecfg['activation'])
                        all_indices.append(ix.numpy().astype(np.uint16));all_values.append(act.numpy())
                indices=np.concatenate(all_indices);activations=np.concatenate(all_values)
                np.savez_compressed(run/f'seed{seed}_reference_codes.npz',indices=indices,activations=activations)
                width=w.shape[0]
                del w,b,bd,all_indices,all_values
            else:
                cached=np.load(checked(cache/f'seed{seed}_reference_codes.npz'))
                indices=cached['indices'];activations=cached['activations'];width=8192
                if len(indices)!=len(reference):raise ValueError('Cached codes and states differ in token count')
            distributions[seed],centroids[seed],counts[seed]=top_distributions(indices,activations,width,cfg['top_k'],reference)
            np.savez_compressed(run/f'seed{seed}_context_distributions.npz',centroids=centroids[seed],positive_counts=counts[seed])
            del indices,activations
            progress('REFERENCE_CODES_COMPLETE',seed=seed,active=int((counts[seed]>0).sum()),eligible=int((counts[seed]>=cfg['minimum_activations']).sum()))
        rng=np.random.default_rng(cfg['distance_scale_seed'])
        sample=rng.choice(len(reference),size=min(256,len(reference)),replace=False)
        cost_sample=euclidean_cost(reference[sample].astype(float),reference[sample].astype(float))
        scale=float(np.median(cost_sample[np.triu_indices(len(sample),1)]))
        regularization=cfg['sinkhorn_scale_fraction']*scale
        def record(row):
            rows.append(dict(run_id=cfg['run_id'],metric_version='semantic_context_topk_v1',**row))
            with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(rows[-1])+'\n')
            progress('FEATURE_MATCH_COMPLETE',completed=len(rows),total=len(source_ids),source_member=row['source_member'],match_status=row['status'])
        matches=retrieve(source_ids,distributions[1],distributions[2],centroids[1],centroids[2],counts[1],counts[2],reference,
                         candidate_count=cfg['centroid_candidates'],minimum_count=cfg['minimum_activations'],
                         regularization=regularization,tolerance=cfg['sinkhorn_tolerance'],max_iterations=cfg['sinkhorn_max_iterations'],progress=record,dual_fallback=cfg.get('dual_fallback',False))
        write(run/'matches.json',dict(source_seed=1,target_seed=2,source_ids=source_ids,rows=matches,
              reference_tokens=len(reference),top_k=cfg['top_k'],regularization=regularization,reference_distance_scale=scale,reference_cache_run=cfg.get('reference_cache_run'),
              ground_cost='Euclidean in same-token layer3 residual space; one global regularization scale',
              unmatched_policy='Keep source member as unmatched; downstream missing contribution is zero, with coverage recorded',scope=cfg['scope']))
        checks['all_source_members_recorded']=len(matches)==len(source_ids)
        checks['matches_have_reference_support']=all(r['reference_activations']>=cfg['minimum_activations'] for r in matches if r['status']=='MATCHED')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    status='PASS' if error is None and all(checks.values()) else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,rows=len(rows),wall_seconds=time.perf_counter()-started,
                 process_cpu_seconds=time.process_time()-cpu_start,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'))
    write(run/'inputs.json',dict(inputs=inputs));write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary)
    write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    validation=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=validation.ok,errors=list(validation.errors)))
    print(json.dumps(dict(**summary,contract_ok=validation.ok)),flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__':raise SystemExit(main())
