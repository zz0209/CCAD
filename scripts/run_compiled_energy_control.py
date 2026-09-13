"""Retained-development norm/orientation exchange and optional train-only scale fitting."""
from __future__ import annotations
import argparse, json, os, platform, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.artifacts import sha256, validate_run_directory
from ccad.causalgym_interface import CausalGymModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    cfg=json.loads(parser.parse_args().config.read_text())
    run=ROOT/'runs'/cfg['run_id']; run.mkdir(exist_ok=False)
    started=datetime.now(timezone.utc).isoformat(); timer=time.perf_counter(); cpu=time.process_time()
    write(run/'config.resolved.json',cfg)
    codes=[]
    for rel in ['scripts/run_compiled_energy_control.py','scripts/run_r011s1_raw_hook_asset.py',
                'src/ccad/causalgym_interface.py','src/ccad/finite_native_group.py','src/ccad/artifacts.py']:
        p=ROOT/rel; q=run/'source_snapshot'/rel; q.parent.mkdir(parents=True,exist_ok=True); q.write_bytes(p.read_bytes())
        codes.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=codes,aggregate_sha256=aggregate(codes),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='compiled.energy.control.v1',run_id=cfg['run_id'],
        run_parent=cfg['round_id'],purpose=cfg['purpose'],milestone='M4',evidence_level=cfg['evidence_level'],
        started_utc=started,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(codes),source_snapshot_required=True,audit_opened=False,
        candidate_family_frozen=True,mean_constants_source_split='retained parent operator',
        threshold_source_split='prespecified replay and norm tolerances; no scientific threshold',
        statistics_unit='paired development rows and six fixed queries; one shared SAE seed edge',
        device='cuda:0',seeds=[1,2],resource_lease='gpu-0 resource_manager.run',
        resource_lease_reason='Bounded actual language-model norm/orientation interventions'))
    for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
    write(run/'status.json',dict(status='RUNNING',updated_utc=started))
    inputs=[]; rows=[]; results=[]; checks={}; env={}; error=None; api=None

    def checked(path,source='CCAD retained input',license='internal'):
        p=Path(path); p=p if p.is_absolute() else ROOT/p
        inputs.append(entry(p,source,'actual input',license)); return p

    def log(event,**kw):
        item=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),event=event,
                  elapsed_seconds=time.perf_counter()-timer,**kw)
        write(run/'progress.json',item)
        with (run/'stdout.log').open('a') as stream:stream.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)

    def record(row):
        rows.append(row)
        with (run/'metrics.raw.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

    try:
        import torch, numpy as np, transformers
        parent=ROOT/cfg['parent_run']; cp=checked(parent/'config.resolved.json')
        assert sha256(cp)==cfg['parent_config_sha256']
        pc=json.loads(cp.read_text()); assert json.loads(checked(parent/'status.json').read_text())['status']=='PASS'
        for key in ['dictionary_source_dir','dictionary_overlay_dir','scipy_overlay_dir']:sys.path.append(pc[key])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('high')
        tr=ROOT/pc['training_run']; tc=json.loads(checked(tr/'config.resolved.json').read_text())
        assert json.loads(checked(tr/'status.json').read_text())['status']=='PASS'
        snaps=json.loads(checked(tr/'checkpoints.json').read_text())['checkpoints']; saes={}; decoders={}
        for objective in pc['objectives']:
            for seed in pc['seeds']:
                snap=next(s for s in snaps if s['step']==pc['checkpoint_step'] and s['objective']==objective and s['seed']==seed)
                p=checked(snap['path']); assert sha256(p)==snap['sha256']
                state=torch.load(p,map_location='cuda:0',weights_only=True)
                ae=(AutoEncoderTopK(1024,tc['dict_size'],tc['k']) if objective=='topk' else
                    MatryoshkaBatchTopKSAE(1024,tc['dict_size'],tc['k'],state['group_sizes'].cpu().tolist()))
                ae=ae.to('cuda:0'); ae.load_state_dict(state); ae.eval(); ae.requires_grad_(False)
                saes[objective,seed]=ae; decoders[objective,seed]=ae.decoder.weight.T if objective=='topk' else ae.W_dec
        for fn in ['config.json','tokenizer.json','model.safetensors']:
            checked(Path(tc['model_local_dir'])/fn,tc['model_id']+' '+tc['model_revision'],'MIT')
        tokenizer=transformers.AutoTokenizer.from_pretrained(tc['model_local_dir'],local_files_only=True)
        tokenizer.pad_token=tokenizer.eos_token
        model=transformers.AutoModelForCausalLM.from_pretrained(tc['model_local_dir'],local_files_only=True,
            dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0')
        model.requires_grad_(False); model.config.use_cache=False
        api=CausalGymModel(model,tokenizer,model.get_submodule(tc['hook_module_path']),
            checked(pc['official_data_module'],'Original CausalGym Pair/Batch','private reference source'),
            tc.get('layer',tc.get('layer_index')))
        env=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
            transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),threads=2)
        panel=json.loads(checked(parent/'panel.json').read_text())['rows']
        assert all(r['official_split'] in ['train','dev'] for r in panel)
        choices={c['query']:c for c in json.loads(checked(parent/'selection_choices.json').read_text())['choices']}
        old={}
        for line in checked(parent/'metrics.raw.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['method'] in ['source','compiled_shared_axis','compiled_functional_axis']:
                old[r['query'],r['method'],r['row_id']]=r
        cached_task=None; codes_task={}; batch_size=cfg['batch_size']
        assert batch_size==pc['batch_size']
        for query in cfg['queries']:
            c=choices[query]; task=c['task']; region=c['region']
            taskrows=[r for r in panel if r['task']==task]; nfit=pc['train_rows_per_task']; ncal=pc['calibration_rows']
            assert all(r['split']=='held_dev' for r in taskrows[nfit+ncal:])
            if task!=cached_task:
                base=[]; donor=[]
                for off in range(0,len(taskrows),batch_size):
                    batch=api.batch(taskrows[off:off+batch_size]); bp,sp=api.positions(batch,region)
                    ids=torch.arange(len(batch.pairs),device='cuda:0')
                    base.append(api.forward(batch,oracle=off==0)['hidden'][ids,bp])
                    donor.append(api.forward(batch,donor=True)['hidden'][ids,sp])
                bh=torch.cat(base); dh=torch.cat(donor)
                with torch.no_grad():codes_task={key:(ae.encode(bh),ae.encode(dh)) for key,ae in saes.items()}
                cached_task=task
            zs0,zs1=codes_task[c['objective'],c['source_seed']]; zt0,zt1=codes_task[c['objective'],c['target_seed']]
            Ds=decoders[c['objective'],c['source_seed']]; Dt=decoders[c['objective'],c['target_seed']]
            ops=[]
            for suffix in ['_compiled_axis_operator.npz','_compiled_functional_axis_operator.npz']:
                with np.load(checked(parent/(query+suffix)),allow_pickle=False) as data:
                    ops.append({k:torch.as_tensor(v,device='cuda:0') for k,v in data.items()})
            g,f=ops
            for key in ['source_axis','target_reader','positive_negative_members']:assert torch.equal(g[key],f[key])
            with np.load(checked(parent/(query+'_axis_source.npz')),allow_pickle=False) as data:
                U=torch.as_tensor(data['basis'],device='cuda:0'); assert torch.equal(U,g['source_axis'])
            q=(((zs1-zs0)@Ds)@U)@U.T
            alpha=(zt1-zt0)[nfit:]@g['target_reader']; sign=(alpha<0).long(); magnitude=alpha.abs()
            gv=g['realized_unit_directions']; fv=f['realized_unit_directions']; gn=gv.norm(dim=1); fn=fv.norm(dim=1)
            assert bool((gn>0).all() and (fn>0).all())
            one=torch.ones_like(gn)
            variants={'geometric_original':(g,one),'geometric_learned_norm':(g,fn/gn),
                'functional_original_norm':(f,gn/fn),'functional_learned_norm':(f,one)}
            if cfg.get('norm_fits'):
                from ccad.finite_native_group import fit_compiled
                reference=[]
                for off in range(0,nfit,batch_size):
                    batch=api.batch(taskrows[off:off+batch_size]); bp,_=api.positions(batch,region)
                    reference.append(api.forward(batch,positions=bp,delta=q[off:off+len(batch.pairs)])['log_probs'])
                reference=torch.cat(reference)
                alpha_train=(zt1-zt0)[:nfit]@g['target_reader']
                unit=torch.cat([U.T,-U.T])
                for name,fit_config in cfg['norm_fits'].items():
                    # One fixed ray per sign makes the existing nonnegative
                    # fitter optimize exactly two amplitudes. No learned full
                    # writer weights or held outcomes initialize these fits.
                    scale,fit=fit_compiled(api,taskrows[:nfit],region,alpha_train,
                        torch.ones((2,1),device='cuda:0'),unit,reference,
                        decoder=gv[:,None,:],**fit_config)
                    variants[name]=(g,scale[:,0])
                    fit.update(query=query,parameters=2,initial_scales=[1.,1.],
                        independently_fitted=True,scales=scale[:,0].cpu().tolist(),
                        fixed_geometric_directions=True,held_outcomes_used=False)
                    write(run/(query+'_'+name+'_fit.json'),fit)
                    log('NORM_FIT_COMPLETE',query=query,method=name,scales=fit['scales'],
                        wall_seconds=fit['wall_seconds'],final_train=fit['trace'][-1])
            deltas={}; writes={}; diagnostics={}
            for name,(op,scale) in variants.items():
                rays=op['realized_unit_directions']*scale[:,None]
                weights=op['positive_negative_code_weights']*scale[:,None]
                assert bool((weights>=0).all())
                qt=magnitude[:,None]*rays[sign]; deltas[name]=qt
                update=magnitude[:,None]*weights[sign]; members=op['positive_negative_members'][sign]
                replay=torch.einsum('nb,nbd->nd',update,Dt[members])
                replay_error=float((replay-qt).abs().max())
                assert torch.allclose(replay,qt,rtol=1e-5,atol=1e-5),(query,name,replay_error)
                np.savez_compressed(run/(query+'_'+name+'_write.npz'),
                    row_ids=np.array([r['row_id'] for r in taskrows[nfit:]]),members=members.cpu().numpy(),
                    code_increment=update.cpu().numpy(),realized_unit_directions=rays.cpu().numpy())
                diagnostics[name]=dict(sign_scales=scale.cpu().tolist(),native_replay_max_abs=replay_error,
                    max_changed_members=int((update!=0).sum(1).max()),unit_norms=rays.norm(dim=1).cpu().tolist())
            for a,b in [('geometric_original','functional_original_norm'),('geometric_learned_norm','functional_learned_norm')]:
                assert torch.allclose(deltas[a].norm(dim=1),deltas[b].norm(dim=1),rtol=cfg['norm_match_rtol'],atol=1e-6)
            replay_max={name:0. for name in ['source','compiled_shared_axis','compiled_functional_axis']}
            method_map={'geometric_original':'compiled_shared_axis','functional_learned_norm':'compiled_functional_axis'}
            for off in range(nfit+ncal,len(taskrows),batch_size):
                current=taskrows[off:off+batch_size]; batch=api.batch(current); bp,_=api.positions(batch,region)
                teacher=api.forward(batch,positions=bp,delta=q[off:off+len(current)])
                for j,r in enumerate(current):
                    previous=old[query,'source',r['row_id']]
                    replay_max['source']=max(replay_max['source'],abs(float(teacher['margin'][j])-previous['margin']))
                    assert bool(teacher['margin'][j]>0)==previous['iia']
                for name,qt in deltas.items():
                    edit=qt[off-nfit:off-nfit+len(current)]
                    out=api.forward(batch,positions=bp,delta=edit)
                    kl=(teacher['log_probs'].exp()*(teacher['log_probs']-out['log_probs'])).sum(1)
                    for j,r in enumerate(current):
                        margin=float(out['margin'][j]); previous_method=method_map.get(name)
                        if previous_method:
                            previous=old[query,previous_method,r['row_id']]
                            replay_max[previous_method]=max(replay_max[previous_method],abs(margin-previous['margin']))
                            assert (margin>0)==previous['iia']
                        record(dict(query=query,task=task,objective=c['objective'],source_seed=c['source_seed'],
                            target_seed=c['target_seed'],method=name,row_id=r['row_id'],official_index=r['original_index'],
                            official_split=r['official_split'],split=r['split'],pair_key=r['pair_key'],margin=margin,
                            iia=margin>0,source_kl=float(kl[j]),donor_ce=-float(out['log_probs'][j,batch.src_labels[j]]),
                            source_delta_norm=float(q[off+j].norm()),target_delta_norm=float(edit[j].norm())))
            assert max(replay_max.values())<=cfg['margin_replay_tolerance'],(query,replay_max)
            summary={}
            for name in variants:
                values=[r for r in rows if r['query']==query and r['method']==name]
                summary[name]={key:float(np.mean([r[key] for r in values])) for key in ['iia','margin','source_kl','donor_ce','target_delta_norm']}
            result=dict(query=query,task=task,objective=c['objective'],held_rows=len(taskrows)-nfit-ncal,
                orientation_cosine=torch.nn.functional.cosine_similarity(gv,fv,dim=1).cpu().tolist(),
                diagnostics=diagnostics,replay_margin_max_abs=replay_max,summary=summary)
            results.append(result); write(run/'query_results.json',dict(queries=results,scope=cfg['scope']))
            log('QUERY_COMPLETE',query=query,summary=summary,replay_margin_max_abs=replay_max)
            if time.perf_counter()-timer>cfg['budget_seconds']:raise TimeoutError('Bounded development control exceeded budget')
        checks.update(completed_queries=len(results),official_hook=api.checks,test_read=False,
                      norm_match=True,nonnegative_native_writes=True,unchanged_original_IIA=True,
                      independent_norm_fits=list(cfg.get('norm_fits',{})))
    except BaseException as exc:
        error=repr(exc); (run/'stderr.log').write_text(traceback.format_exc()); log('FAIL',error=error)
    if api is not None:
        env.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(),forward_sequences=api.sequences,
                   forward_tokens=api.tokens,backward_sequences=api.backward_sequences)
    status='PASS' if error is None else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,rows=len(rows),completed_queries=len(results),
        wall_seconds=time.perf_counter()-timer,process_cpu_seconds=time.process_time()-cpu,
        metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path=codes[0]['path'],
        generator_script_sha256=codes[0]['sha256'],scope=cfg['scope'])
    write(run/'inputs.json',dict(inputs=inputs)); write(run/'environment.json',env); write(run/'metrics.summary.json',summary)
    write(run/'status.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),error=error))
    validation=validate_run_directory(run)
    write(run/'contract_validation.json',dict(ok=validation.ok,errors=validation.errors))
    log('COMPLETE',summary=summary,contract_ok=validation.ok,errors=validation.errors)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__':raise SystemExit(main())
