"""Fit existing signed component maps to finite full-vocabulary intervention KL."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
from composition_runtime import CompositionRun,ROOT,write,np,HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from run_component_correspondence import measure
from ccad.component_correspondence import predefined_masks,component_update
from ccad.finite_output_fit import DirectLayerNormHead,fit_finite_map


BASE_METHODS=['aggregate_ols','dense_half','matched_atoms','aggregate_full','aggregate_raw']
FIT_METHODS=[('finite_ols','aggregate_ols','matrix'),('finite_dense','dense_half','matrix'),('finite_atoms','matched_atoms','diagonal'),('finite_full','aggregate_full','matrix'),('finite_raw','aggregate_raw','matrix'),('finite_scalar','aggregate_ols','scalar')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_finite_output_correspondence.py',['scripts/run_component_correspondence.py','src/ccad/finite_output_fit.py','src/ccad/component_correspondence.py','scripts/run_pair_complete_correspondence.py','src/ccad/complete_support.py','src/ccad/pair_complete_correspondence.py','src/ccad/factor_correspondence.py','src/ccad/nip_baselines.py']);error=None
    manifest=json.loads((work.run/'manifest.json').read_text());manifest.update(milestone='finite output fitting of reusable source components',candidate_family_frozen=bool(work.cfg.get('frozen_finite_fit_run')),mean_constants_source_split='Uncentered absolute source-code map; no mean fit. Stronger source full-distribution supervision declared.',statistics_unit='Five shared controlled SAE seeds and authored lexical/cue pairs');write(work.run/'manifest.json',manifest)
    try:
        work.load();cfg=work.cfg;torch=work.torch
        def budget_check():
            if time.perf_counter()-work.start>cfg['budget_seconds']:raise TimeoutError('Finite-output run measured wall budget exhausted')
        if cfg.get('freeze_json'):
            freeze=json.loads(work.checked(ROOT/cfg['freeze_json']).read_text())
            for r in freeze['files']:assert hashlib.sha256(work.checked(ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
            assert hashlib.sha256(json.dumps([r['text'] for r in work.rows],ensure_ascii=False).encode()).hexdigest()==freeze['evaluation_texts_sha256']
            work.checks['freeze_and_new_text_identity']=True
        dev=ROOT/cfg['development_material_run'];panel=json.loads(work.checked(dev/'panel.json').read_text())
        dev_h=np.load(work.checked(dev/'raw_cache.npz'))['layer15'][:,0].astype(np.float64)
        fit_ids=np.array([i for i,r in enumerate(panel['rows']) if r['block'] in cfg['fit_blocks']]);cal_ids=np.array([i for i in range(len(panel['rows'])) if i not in set(fit_ids)])
        donors={f:np.array([r[f] for r in panel['pairs']]) for f in cfg['source_budgets']}
        for f,ds in donors.items():
            assert set(ds[fit_ids]).issubset(set(fit_ids)) and set(ds[cal_ids]).issubset(set(cal_ids))
        assets={};allfits=[];validations=[];geometry=[];masks_record=[];expected_rows=0
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];a=np.load(work.checked(dev/f'seed{seed}_codes.npz'));b=np.load(work.checked(ROOT/cfg['evaluation_code_run']/f'seed{seed}_codes.npz'))
            assets[seed]=dict(dev=a['number_z'].astype(np.float64),evaluation=b['number_z'].astype(np.float64))
        if cfg.get('finite_calibration_material_run') and not cfg.get('frozen_finite_fit_run'):
            calibration_root=ROOT/cfg['finite_calibration_material_run']
            cp=json.loads(work.checked(calibration_root/'panel.json').read_text())
            selected=np.array([i for i,r in enumerate(cp['rows']) if r['block'] in cfg['finite_calibration_blocks']])
            inverse=np.full(len(cp['rows']),-1);inverse[selected]=np.arange(len(selected))
            original_n=len(dev_h)
            for f in donors:
                cd=inverse[np.array([r[f] for r in cp['pairs']])[selected]]
                assert (cd>=0).all(),'Calibration lexical selection splits a reciprocal pair'
                donors[f]=np.concatenate([donors[f],original_n+cd])
            ch=np.load(work.checked(calibration_root/'raw_cache.npz'))['layer15'][selected,0].astype(np.float64)
            dev_h=np.concatenate([dev_h,ch]);fit_ids=np.arange(original_n);cal_ids=np.arange(original_n,len(dev_h))
            for seed,asset in assets.items():
                cc=np.load(work.checked(ROOT/cfg['finite_calibration_code_run']/f'seed{seed}_codes.npz'))['number_z'][selected].astype(np.float64)
                asset['dev']=np.concatenate([asset['dev'],cc])
            write(work.run/'fit_calibration_panel.json',dict(fit_material=cfg['development_material_run'],fit_original_row_ids=list(range(original_n)),calibration_material=cfg['finite_calibration_material_run'],calibration_original_row_ids=selected.tolist(),merged_fit_ids=fit_ids.tolist(),merged_calibration_ids=cal_ids.tolist(),calibration_texts=[cp['rows'][i]['text'] for i in selected],scope='All original geometry-training rows fit finite KL. Separate already-exposed role contexts select the iterate. Neither panel is fresh confirmation.'))
        for source_seed,target_seed in cfg['seed_pairs']:
            for factor,budget in cfg['source_budgets'].items():
                origin=ROOT/(cfg.get('frozen_finite_fit_run')or cfg['initial_map_run']);old=np.load(work.checked(origin/f'maps_s{source_seed}_t{target_seed}_{factor}.npz'))
                support=old['source_members'];decoder=old['source_decoder'];source=assets[source_seed];target=assets[target_seed]
                zs=source['dev'][:,support];ze=source['evaluation'][:,support]
                methods={name:dict(members=old[name+'_members'],weights=old[name+'_weights'],input_kind='raw' if name=='aggregate_raw' else 'codes') for name in BASE_METHODS}
                if cfg.get('frozen_finite_fit_run'):
                    meta=json.loads(work.checked(origin/f'maps_s{source_seed}_t{target_seed}_{factor}.json').read_text())
                    for name,_,_ in FIT_METHODS:methods[name]=dict(members=old[name+'_members'],weights=old[name+'_weights'],input_kind=meta['input_kinds'][name])
                else:
                    # Preserve the actual production head and float32 update
                    # rounding; avoid a numerically different GEMM rearrangement.
                    to=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float64)
                    head=DirectLayerNormHead(to(dev_h),to(decoder),work.model.gpt_neox.final_layer_norm,work.model.get_output_embeddings())
                    # Match the production/cache batch shape. GPU GEMM output
                    # and gradients can differ when the same first rows are
                    # evaluated in a different batch shape.
                    ids_np=np.arange(min(cfg['batch_size'],work.n));ids=torch.as_tensor(ids_np,device='cuda')
                    witness=DirectLayerNormHead(to(work.h[:,0]),to(decoder),work.model.gpt_neox.final_layer_norm,work.model.get_output_embeddings())
                    c=to(-ze[ids_np]);c.requires_grad_(True);lp_fit=witness.logprobs(c,ids)
                    delta=np.zeros_like(work.h,dtype=np.float64);delta[:,0]=(-ze)@decoder
                    lp_actual=work.evaluate(delta,ids_np)
                    # Check the actual fitting objective through a complete
                    # Transformer, rather than two differently rounded ways
                    # of computing a selected-logit difference.
                    teacher=to(np.exp(work.base[ids_np]))
                    ga=torch.autograd.grad(-(teacher*lp_fit).sum(),c)[0].detach().cpu().numpy()
                    full_c=c.detach().clone().requires_grad_(True)
                    enc=work.tokenizer([work.rows[i]['text'] for i in ids_np],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                    ix=torch.arange(len(ids_np),device='cuda');last=enc.attention_mask.sum(1)-1
                    contract=HookPointContract(f"gpt_neox.layers.{cfg['sae_layer']}",cfg['sae_layer'],'resid_post',work.model.config.hidden_size)
                    def full_hook(mod,args,out):
                        h=extract_primary_hook_tensor(out,contract).clone()
                        h[ix,torch.as_tensor(work.positions[ids_np,0],device='cuda')]+=(full_c@to(decoder)).to(h.dtype)
                        return replace_primary_hook_tensor(out,h,contract)
                    handle=work.model.get_submodule(f"gpt_neox.layers.{cfg['sae_layer']}").register_forward_hook(full_hook)
                    try:
                        hidden=work.model.gpt_neox(**enc,use_cache=False).last_hidden_state[ix,last]
                        full_lp=torch.log_softmax(work.model.get_output_embeddings()(hidden).double(),dim=-1)
                        gb=torch.autograd.grad(-(teacher*full_lp).sum(),full_c)[0].detach().cpu().numpy()
                    finally:handle.remove()
                    work.forwards+=len(ids_np)
                    lp_error=float(np.max(np.abs(lp_fit.detach().cpu().numpy()-lp_actual)));gradient_error=float(np.linalg.norm(ga-gb)/max(np.linalg.norm(gb),1e-12))
                    full_lp_error=float((lp_fit.detach()-full_lp.detach()).abs().max())
                    checks=dict(source_seed=source_seed,factor=factor,batch_size=len(ids_np),actual_head_logprob_max_error=lp_error,actual_full_model_logprob_max_error=full_lp_error,actual_full_model_gradient_relative_error=gradient_error,logprob_tolerance=1e-4,gradient_relative_tolerance=1e-4,scope='Same full-vocabulary distribution loss and production batch shape on original head versus complete-Transformer hooked source-code gradient; double source update rounded before original float32 head.')
                    validations.append(checks);write(work.run/'direct_head_validation.json',dict(rows=validations))
                    assert lp_error<1e-4 and full_lp_error<1e-4 and gradient_error<1e-4,checks
                    del c,lp_fit,teacher,full_c,full_lp,hidden,enc,witness,ga,gb
                    masks=[('whole',np.ones(budget))];rng=np.random.default_rng(12000+100*source_seed+(factor=='time'))
                    for i in range(2):
                        mask=np.zeros(budget);mask[rng.choice(budget,budget//2,replace=False)]=1;masks.append((f'fit_random_half_{i}',mask))
                    for name,parent,mode in FIT_METHODS:
                        m=methods[parent];x=dev_h if m['input_kind']=='raw' else target['dev'];start=time.perf_counter()
                        weights,diag=fit_finite_map(head,to(x[:,m['members']]),to(zs),to(m['weights']),torch.as_tensor(donors[factor],device='cuda'),[(n,to(a)) for n,a in masks],torch.as_tensor(fit_ids,device='cuda'),torch.as_tensor(cal_ids,device='cuda'),cfg,mode=mode,budget_check=budget_check,progress=lambda **kw:work.progress('FINITE_FIT_CHECKPOINT',source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,**kw))
                        methods[name]=dict(members=m['members'].copy(),weights=weights,input_kind=m['input_kind']);allfits.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,parent_method=parent,**diag))
                        np.savez_compressed(work.run/f'fit_s{source_seed}_t{target_seed}_{factor}_{name}.npz',members=m['members'],weights=weights,source_members=support,source_decoder=decoder)
                        write(work.run/'finite_fit_diagnostics.json',dict(rows=allfits));work.progress('FINITE_METHOD_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,seconds=time.perf_counter()-start)
                    del head
                arrays=dict(source_members=support,source_decoder=decoder)
                for name,m in methods.items():arrays.update({name+'_members':m['members'],name+'_weights':m['weights']})
                np.savez_compressed(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.npz',**arrays)
                write(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.json',dict(source_seed=source_seed,target_seed=target_seed,factor=factor,input_kinds={n:m['input_kind'] for n,m in methods.items()},fit_rows=0 if cfg.get('frozen_finite_fit_run') else len(zs),scope='Finite-KL source allocation, not a target-native or semantic operation'))
                predictions={name:(work.h[:,0] if m['input_kind']=='raw' else target['evaluation'])[:,m['members']]@m['weights'] for name,m in methods.items()}
                np.savez_compressed(work.run/f'predictions_s{source_seed}_t{target_seed}_{factor}.npz',source=ze,**predictions)
                for mask_name,mask in predefined_masks(budget,source_seed*100+(factor=='time')):
                    masks_record.append(dict(source_seed=source_seed,factor=factor,name=mask_name,source_members=support.tolist(),scales=mask.tolist()))
                    for consumer in cfg['consumers']:
                        change=lambda z:z[work.donors[factor]]-z if consumer=='contrast' else -z
                        for dose in cfg.get('mask_doses',{}).get(mask_name,cfg['doses']):
                            teacher=dose*component_update(change(ze),decoder,mask)
                            reference=measure(work,'source',factor,mask_name,consumer,dose,teacher,None,source_seed,target_seed)
                            for name,pred in predictions.items():
                                vectors=dose*component_update(change(pred),decoder,mask)
                                measure(work,name,factor,mask_name,consumer,dose,vectors,reference,source_seed,target_seed)
                                for role in ['temporal','quoted']:
                                    ids=np.array([i for i in work.evaluation_ids if work.rows[i].get('cue_role','temporal')==role])
                                    if len(ids):geometry.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,mask=mask_name,consumer=consumer,dose=dose,role=role,method=name,n=len(ids),error_sse=float(np.sum((vectors[ids]-teacher[ids])**2)),teacher_energy=float(np.sum(teacher[ids]**2))))
                            expected_rows+=len(work.evaluation_ids)*(1+len(predictions))
                work.progress('FINITE_FACTOR_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor)
        write(work.run/'geometry.json',dict(rows=geometry));write(work.run/'source_masks.json',dict(rows=masks_record));write(work.run/'finite_fit_diagnostics.json',dict(rows=allfits))
        write(work.run/'fit_compute.json',dict(differentiable_head_training_queries=sum(r['fitting_head_queries'] for r in allfits),differentiable_head_teacher_queries=sum(r['teacher_head_queries'] for r in allfits),fit_wall_seconds=sum(r['wall_seconds'] for r in allfits),scope='Original LayerNorm/unembedding fitting and closure queries counted separately; not complete Transformer forwards.'))
        work.checks['all_rows']=len(work.metrics)==expected_rows
        work.checks['unique_rows']=len(work.metrics)==len({(r['source_seed'],r['target_seed'],r['factor'],r['mask'],r['consumer'],r['dose'],r['method'],r['row_id']) for r in work.metrics})
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
