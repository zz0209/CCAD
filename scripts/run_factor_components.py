"""Factor-difference SAE material and compact source-component interventions.

All factors and lexical splits are development data. No language-model endpoint
is used to choose the source direction or membership. Source and target edits
are additive at the same typed token position; decoder bias cancels in pairs.
"""
import argparse,json,os,platform,sys,time,traceback,shutil
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from run_functional_material import logprob
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();write(run/'config.resolved.json',cfg);inputs=[];checks={};metrics=[];env={};error=None;forwards=0
    code=[]
    for rel in ['scripts/run_factor_components.py','scripts/factor_transfer.py','scripts/run_functional_material.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/factor_correspondence.py','src/ccad/nip_baselines.py','src/ccad/proposal.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']:
        p=ROOT/rel;q=run/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='factor.components.v1',run_id=cfg['run_id'],run_parent='SEVEN_R2',purpose=cfg['purpose'],milestone='functional-components',evidence_level='controlled_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='paired differences cancel fixed mean',threshold_source_split='source lexical discovery blocks; no target or LM endpoints',statistics_unit='lexical blocks; reciprocal directions and syntax dependent',device='cuda:0',seeds=cfg.get('seeds',[]),resource_lease='gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
    for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/fn).touch()
    write(run/'status.json',dict(status='RUNNING'))
    def checked(path):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;inputs.append(entry(p,'existing pinned CCAD assets','input'));return p
    def record(row):
        line=json.dumps(row)
        with (run/'metrics.raw.jsonl').open('a') as f:f.write(line+'\n')
        metrics.append(row)
    def progress(stage,**kw):
        row=dict(stage=stage,elapsed=time.perf_counter()-start,sequence_forwards=forwards,**kw);write(run/'progress.json',row);print(json.dumps(row),flush=True)
    try:
        import torch,transformers
        from sparsify import SparseCoder
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');torch.cuda.reset_peak_memory_stats()
        parent=ROOT/cfg['resume_parent'] if cfg.get('resume_parent') else None
        if parent is not None:
            oldcfg=json.loads(checked(parent/'config.resolved.json').read_text())
            for key in ['model_local_dir','hook_module_path','layer_index','batch_size','discovery_blocks','member_budgets','checkpoints','transfer']:assert oldcfg[key]==cfg[key],key
            checked(parent/'code_hashes.json');prefix=checked(parent/'metrics.raw.jsonl');shutil.copy2(prefix,run/'metrics.raw.jsonl');metrics.extend(json.loads(line) for line in prefix.read_text().splitlines());inherited=[]
            for p in sorted(parent.iterdir()):
                if p.suffix=='.npz' or p.name.endswith('_components.json') or (p.name.startswith('transfer_') and p.suffix=='.json'):
                    checked(p);shutil.copy2(p,run/p.name);inherited.append(dict(path=p.name,parent_path=str(p),sha256=sha256(p),bytes=p.stat().st_size))
            write(run/'inherited_outputs.json',dict(parent_run=str(parent),prefix_rows=len(metrics),prefix_bytes=prefix.stat().st_size,prefix_sha256=sha256(prefix),files=inherited,numerical_change='OT max iteration ceiling only; same epsilon/tolerance/objective. Completed parent outputs retained byte-for-byte.'))
        checked(args.config);source=checked(Path(cfg['material_run'])/'crossed_inputs.json');panel=json.loads(source.read_text());rows=panel['rows'];pairs=panel['pairs']
        if cfg.get('transfer'):
            checked('.aris/compute/local-f4-sparse-env-spec.json');checked('D:/CCAD_Storage/environments/f4_sparse_overlay_v1/sklearn/linear_model/_coordinate_descent.py')
        tr=json.loads(checked(Path(cfg['material_run'])/'1b_token_rows.json').read_text());labels=tr['label_ids']
        modeldir=Path(cfg['model_local_dir'])
        for fn in ['config.json','tokenizer.json','model.safetensors']:checked(modeldir/fn)
        tokenizer=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().cuda();model.config.use_cache=False
        contract=HookPointContract(cfg['hook_module_path'],cfg['layer_index'],'resid_post',model.config.hidden_size);module=model.get_submodule(cfg['hook_module_path']);bs=cfg['batch_size']
        positions=np.array([[r['region_ends'][1],r['region_ends'][3]] for r in tr['rows']]);h=np.empty((len(rows),2,model.config.hidden_size),np.float32)
        def forward(ids,delta=None,slot=None,capture=False):
            nonlocal forwards
            if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Component run budget')
            texts=[rows[i]['text'] for i in ids];enc=tokenizer(texts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');ix=torch.arange(len(ids),device='cuda');last=enc.attention_mask.sum(1)-1;cap=[]
            def hook(mod,args,out):
                hidden=extract_primary_hook_tensor(out,contract)
                if capture:cap.append(hidden[ix[:,None],torch.as_tensor(positions[ids],device='cuda')].detach().cpu().numpy().copy())
                if delta is not None:
                    hidden=hidden.clone();p=torch.as_tensor(positions[ids,slot],device='cuda');hidden[ix,p]+=torch.as_tensor(delta,device='cuda');return replace_primary_hook_tensor(out,hidden,contract)
                return out
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    hidden=model.gpt_neox(**enc,use_cache=False).last_hidden_state;lp=logprob(model.get_output_embeddings()(hidden[ix,last]).float().cpu().numpy())
            finally:handle.remove()
            forwards+=len(ids);return lp,cap[0] if cap else None
        if parent is None:
            baseline=[]
            for off in range(0,len(rows),bs):
                ids=np.arange(off,min(off+bs,len(rows)));lp,cap=forward(ids,capture=True);baseline.append(lp);h[ids]=cap
            baseline=np.concatenate(baseline);np.savez_compressed(run/'raw_cache.npz',hidden=h,positions=positions,baseline_logprobs=baseline.astype(np.float32))
        else:
            inherited_raw=np.load(run/'raw_cache.npz');h=inherited_raw['hidden'];positions=inherited_raw['positions'];baseline=inherited_raw['baseline_logprobs'].astype(np.float64)
        # Current precision and batching are captured afresh, rather than assuming
        # an older material cache is numerically identical to this run.
        noop,cap=forward(np.arange(bs),np.zeros((bs,h.shape[-1]),np.float32),np.zeros(bs,dtype=int),capture=True)
        if parent is None:checks['noop_exact']=bool(np.array_equal(noop,baseline[:bs]))
        else:
            checks['resume_raw_hidden_exact']=bool(np.array_equal(cap,h[:bs]));checks['resume_baseline_float32_exact']=bool(np.array_equal(noop.astype(np.float32),baseline[:bs].astype(np.float32)))
            fresh=[noop]
            for off in range(bs,len(rows),bs):fresh.append(forward(np.arange(off,min(off+bs,len(rows))))[0])
            fresh=np.concatenate(fresh);checks['resume_all_baselines_exact_at_saved_precision']=bool(np.array_equal(fresh.astype(np.float32),baseline.astype(np.float32)));baseline=fresh
        progress('RAW_CAPTURED')
        pair_cache={}
        for factor,slot in [('subject',0),('distractor',1)]:
            pp=[p for p in pairs if p['factor']==factor];ri=np.array([p['base'] for p in pp]);di=np.array([p['donor'] for p in pp]);dh=h[di,slot]-h[ri,slot]
            train=np.array([p['block'] in cfg['discovery_blocks'] for p in pp]);sign=np.array([1 if rows[p['donor']][factor+'_number'] else -1 for p in pp]);pair_cache[factor]=(pp,ri,di,dh,train,sign,slot)
        def measure(factor,name,deltas,identity,reference=None,teacher_deltas=None,persist=True):
            pp,ri,di,dh,train,sign,slot=pair_cache[factor];logs=[]
            for off in range(0,len(pp),bs):
                ix=np.arange(off,min(off+bs,len(pp)));lp,_=forward(ri[ix],deltas[ix],np.full(len(ix),slot));logs.append(lp)
                for j,k in enumerate(ix) if persist else []:
                    margin=float(lp[j,labels[1]]-lp[j,labels[0]]);base=baseline[ri[k]];expected=bool(rows[di[k]]['subject_number']);ref=reference[k] if reference is not None else None
                    record(dict(kind='intervention',factor=factor,method=name,**identity,pair_id=pp[k]['id'],block=pp[k]['block'],template=pp[k]['template'],split='discovery' if train[k] else 'held_lexical_development',base=int(ri[k]),donor=int(di[k]),plural_margin=margin,margin_change=margin-float(base[labels[1]]-base[labels[0]]),donor_label_correct=bool((margin>0)==expected),delta_norm=float(np.linalg.norm(deltas[k])),raw_delta_norm=float(np.linalg.norm(dh[k])),relative_vector_sqerror=float(np.sum((deltas[k]-dh[k])**2)/max(1e-12,np.sum(dh[k]**2))),teacher_vector_sqerror=float(np.sum((deltas[k]-teacher_deltas[k])**2)/max(1e-12,np.sum(teacher_deltas[k]**2))) if teacher_deltas is not None else None,kl_to_reference=max(0.,float(np.sum(np.exp(ref)*(ref-lp[j])))) if ref is not None else None))
            return np.concatenate(logs)
        raw_refs={f:measure(f,'raw_donor',p[3],dict(checkpoint='raw')) for f,p in pair_cache.items()} if parent is None else {}
        transfer_cache={}
        for spec in cfg['checkpoints']:
            if parent is not None and spec.get('transfer_seed') is None:continue
            path=Path(spec['path']);checked(path/'sae.safetensors');checked(path/'cfg.json');sae=SparseCoder.load_from_disk(path,device='cuda').eval()
            with torch.no_grad():
                d=sae.W_dec.detach().cpu().numpy()
                if parent is None:
                    x=torch.as_tensor(h.reshape(-1,h.shape[-1]),device='cuda');acts,inds,_=sae.encode(x);z=torch.zeros((len(x),sae.num_latents),device='cuda').scatter_(1,inds,acts);z=z.cpu().numpy().reshape(len(h),2,-1)
                else:
                    inherited_codes=np.load(run/(spec['checkpoint']+'_components.npz'));z=np.stack([inherited_codes['subject_z'],inherited_codes['distractor_z']],axis=1)
            if cfg.get('transfer') and spec.get('transfer_seed') is not None:transfer_cache[spec['transfer_seed']]=(z,d)
            if parent is not None:
                del sae;continue
            identity={k:v for k,v in spec.items() if k!='path'};support_record={};component_arrays={}
            for factor,(pp,ri,di,dh,train,sign,slot) in pair_cache.items():
                dz=z[di,slot]-z[ri,slot];dy=dz@d;u=(dy[train]*sign[train,None]).mean(0);u=u/max(1e-12,np.linalg.norm(u));a=d@u
                score=np.mean(dz[train]*sign[train,None],axis=0)*a;order=np.argsort(-np.abs(score),kind='stable');full_c=dz@a
                support_record[factor]=dict(source_direction='unit mean oriented SAE difference on discovery blocks',scores=score[order[:64]].tolist(),members=order[:64].tolist(),source_coefficient=a[order[:64]].tolist(),direction_norm=float(np.linalg.norm(u)))
                component_arrays[factor+'_u']=u;component_arrays[factor+'_z']=z[:,slot];component_arrays[factor+'_a']=a
                measure(factor,'full_sae_delta',dy,identity,raw_refs[factor]);measure(factor,'projected_full',full_c[:,None]*u,identity,raw_refs[factor])
                for budget in cfg['member_budgets']:
                    support=order[:budget];compact_c=dz[:,support]@a[support]
                    measure(factor,'native_'+str(budget),dz[:,support]@d[support],identity,raw_refs[factor]);measure(factor,'projected_'+str(budget),compact_c[:,None]*u,identity,raw_refs[factor])
            stem=spec['checkpoint'];write(run/(stem+'_components.json'),dict(checkpoint=identity,discovery_blocks=cfg['discovery_blocks'],components=support_record));np.savez_compressed(run/(stem+'_components.npz'),**component_arrays)
            del sae;progress('CHECKPOINT_COMPONENTS_COMPLETE',checkpoint=stem)
        if cfg.get('transfer'):
            from factor_transfer import run_transfer
            assert set(transfer_cache)==set(cfg['transfer']['seeds']);run_transfer(cfg,run,pair_cache,transfer_cache,measure,progress)
            directions={(r['source_seed'],r['target_seed'],r['factor']) for r in metrics if r.get('checkpoint')=='transfer' and r['method']=='full_code_ridge'}
            checks['all_transfer_directions']=len(directions)==len(transfer_cache)*(len(transfer_cache)-1)*len(pair_cache)
            if cfg.get('expected_raw_rows') is not None:checks['expected_rows']=len(metrics)==cfg['expected_raw_rows']
            checks['unique_observations']=len(metrics)==len({(r['checkpoint'],r['method'],r['factor'],r['pair_id'],r.get('source_seed'),r.get('target_seed')) for r in metrics})
        checks['all_checkpoints']=len(cfg['checkpoints'])>0;checks['finite']=all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v,float))
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated(),matmul_precision=torch.get_float32_matmul_precision(),sae_encoding='float32 encode; matches fixed validation evaluator',cpu_threads=4)
        if cfg.get('transfer'):
            import scipy,sklearn
            env.update(scipy=scipy.__version__,sklearn=sklearn.__version__)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL';summary=dict(status=status,error=error,checks=checks,rows=len(metrics),wall_seconds=time.perf_counter()-start,sequence_forwards=forwards,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_factor_components.py',generator_script_sha256=sha256(run/'source_snapshot/scripts/run_factor_components.py'))
    write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env);write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));write(run/'stdout.log',summary);v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True);return 0 if status=='PASS' and v.ok else 1


if __name__=='__main__':raise SystemExit(main())
