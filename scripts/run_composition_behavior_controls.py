"""Behavior-calibrated native gains and DAS-style rank-one raw interchange."""
import argparse,json,traceback
from pathlib import Path
from composition_runtime import CompositionRun,ROOT,write,np,HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor


def train_direction(work,raw,teacher_train,teacher_cal,fit_ids,cal_ids,slot,initial,seed):
    torch=work.torch;cfg=work.cfg;rng=np.random.default_rng(seed)
    parameter=torch.nn.Parameter(torch.as_tensor(initial,dtype=torch.float32,device='cuda'));optimizer=torch.optim.Adam([parameter],lr=cfg['das_learning_rate'])
    raw_gpu=torch.as_tensor(raw,device='cuda',dtype=torch.float32);train_target=torch.as_tensor(np.exp(teacher_train),device='cuda',dtype=torch.float32);fit_lookup={int(i):k for k,i in enumerate(fit_ids)}
    contract=HookPointContract(f"gpt_neox.layers.{cfg['sae_layer']}",cfg['sae_layer'],'resid_post',work.h.shape[-1]);history=[];best_loss=float('inf');best=None
    for step in range(cfg['das_steps']+1):
        if step%cfg['das_checkpoint_every']==0 or step==cfg['das_steps']:
            a=parameter.detach().cpu().numpy().astype(np.float64);a/=np.linalg.norm(a);delta=np.zeros_like(work.h,dtype=np.float64);delta[:,slot]=(raw@a)[:,None]*a
            lp=work.evaluate(delta,cal_ids);loss=float(np.mean(np.sum(np.exp(teacher_cal)*(teacher_cal-lp),axis=1)));history.append(dict(step=step,calibration_kl=loss))
            if loss<best_loss:best_loss=loss;best=a.copy()
        if step==cfg['das_steps']:break
        ids=rng.choice(fit_ids,cfg['das_batch_size'],replace=True);enc=work.tokenizer([work.rows[i]['text'] for i in ids],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');ix=torch.arange(len(ids),device='cuda');last=enc.attention_mask.sum(1)-1
        a=parameter/parameter.norm().clamp_min(1e-12);q=(raw_gpu[ids]@a)[:,None]*a
        def hook(mod,args,out):
            h=extract_primary_hook_tensor(out,contract).clone();h[ix,torch.as_tensor(work.positions[ids,slot],device='cuda')]+=q
            return replace_primary_hook_tensor(out,h,contract)
        handle=work.model.get_submodule(contract.module_path).register_forward_hook(hook)
        try:
            optimizer.zero_grad();hidden=work.model.gpt_neox(**enc,use_cache=False).last_hidden_state[ix,last];lp=torch.log_softmax(work.model.get_output_embeddings()(hidden).float(),dim=-1)
            target=train_target[[fit_lookup[int(i)] for i in ids]];loss=-(target*lp).sum(-1).mean();loss.backward()
            # CausalGym's published optimizer schedule, adapted to the common
            # source-distribution target instead of counterfactual-label CE.
            warm=max(1,cfg['das_steps']//10);progress=step+1;factor=progress/warm if progress<=warm else max(0.,(cfg['das_steps']-progress)/(cfg['das_steps']-warm));optimizer.param_groups[0]['lr']=cfg['das_learning_rate']*factor;optimizer.step()
            with torch.no_grad():parameter.div_(parameter.norm().clamp_min(1e-12))
        finally:handle.remove()
        work.forwards+=len(ids)
    return best,dict(seed=seed,history=history,selected_calibration_kl=best_loss,steps=cfg['das_steps'],training_sequences=cfg['das_steps']*cfg['das_batch_size'],operation='unit-direction raw donor interchange a(aT delta_h)',objective='source full-distribution cross entropy; DAS-style operational adaptation, not original label-CE replication')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args();work=CompositionRun(args.config,'scripts/run_composition_behavior_controls.py',[]);error=None
    try:
        work.load();cfg=work.cfg;torch=work.torch;fit_ids=np.array([i for i,r in enumerate(work.rows) if work.discovery[i] and r['block'] in cfg['fit_blocks']]);cal_ids=np.array([i for i in np.flatnonzero(work.discovery) if i not in set(fit_ids)])
        from sparsify import SparseCoder
        assets={}
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];path=Path(spec['path']);work.checked(path/'sae.safetensors');work.checked(path/'cfg.json');sae=SparseCoder.load_from_disk(path,device='cpu').eval();decoder=sae.W_dec.detach().numpy().astype(np.float64);del sae
            codes=np.load(work.checked(ROOT/cfg['material_run']/f'seed{seed}_codes.npz'));coords=np.load(work.checked(ROOT/cfg['source_run']/f'seed{seed}_source_coordinates.npz'));q=np.zeros_like(work.h,dtype=np.float64);diffs={}
            for factor,slot in [('number',1),('time',0)]:
                budget=cfg['source_budgets'][factor];q[:,slot]=coords[f'{factor}_{budget}_coordinates']@coords[f'{factor}_{budget}_basis'].T;z=codes[factor+'_z'].astype(np.float64);diffs[factor]=z[work.donors[factor]]-z
            assets[seed]=dict(decoder=decoder,q=q,**diffs)
        all_fit=[]
        for source_seed,s in assets.items():
            teacher={};train={};cal={};das=np.zeros_like(work.h,dtype=np.float64);das_details=[]
            for factor,slot in [('number',1),('time',0)]:
                delta=np.zeros_like(work.h,dtype=np.float64);delta[:,slot]=s['q'][:,slot];teacher[factor]=work.measure('source_teacher',factor,delta,source_seed=source_seed)
                train[factor]=work.evaluate(delta,fit_ids);cal[factor]=work.evaluate(delta,cal_ids)
                raw=(work.h[work.donors[factor]]-work.h)[:,slot];_,_,vt=np.linalg.svd(delta[fit_ids,slot],full_matrices=False)
                direction,diag=train_direction(work,raw,train[factor],cal[factor],fit_ids,cal_ids,slot,vt[0],100+10*source_seed+slot);das[:,slot]=(raw@direction)[:,None]*direction;das_details.append(dict(factor=factor,direction=direction.tolist(),**diag))
            teacher['joint']=work.measure('source_teacher','joint',s['q'],source_seed=source_seed)
            for factor,slot in [('number',1),('time',0),('joint',None)]:
                delta=das.copy()
                if slot is not None:delta[:,1-slot]=0
                work.measure('das_style_raw_rank1',factor,delta,teacher[factor],source_seed=source_seed)
            write(work.run/f'das_source{source_seed}.json',dict(source_seed=source_seed,factors=das_details));work.progress('DAS_COMPLETE',source_seed=source_seed)
            for target_seed,t in assets.items():
                if source_seed==target_seed:continue
                maps=np.load(work.checked(ROOT/cfg['correspondence_run']/f'maps_s{source_seed}_t{target_seed}.npz'));variants={name:np.zeros_like(work.h,dtype=np.float64) for name in ['same_members_behavior_gain','direct_target_behavior_gain']};details=[]
                for factor,slot in [('number',1),('time',0)]:
                    members=maps[factor+'_fcc_members'];native=t[factor][:,members]@t['decoder'][members]
                    for name,q in [('same_members_behavior_gain',native),('direct_target_behavior_gain',t['q'][:,slot])]:
                        losses=[]
                        for gain in cfg['native_gain_candidates']:
                            delta=np.zeros_like(work.h,dtype=np.float64);delta[:,slot]=gain*q;lp=work.evaluate(delta,cal_ids);losses.append(float(np.mean(np.sum(np.exp(cal[factor])*(cal[factor]-lp),axis=1))))
                        gain=cfg['native_gain_candidates'][int(np.argmin(losses))];variants[name][:,slot]=gain*q
                        details.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,gain=gain,candidates=cfg['native_gain_candidates'],calibration_kl=losses,calibration_row_ids=cal_ids.tolist(),information='source full-distribution outputs on64calibrationrows; stronger supervision than geometric ridge'))
                for name,delta in variants.items():
                    for factor,slot in [('number',1),('time',0),('joint',None)]:
                        single=delta.copy()
                        if slot is not None:single[:,1-slot]=0
                        work.measure(name,factor,single,teacher[factor],source_seed=source_seed,target_seed=target_seed)
                write(work.run/f'pair_s{source_seed}_t{target_seed}.json',dict(gain_fit=details));all_fit.extend(details);work.progress('GAIN_PAIR_COMPLETE',source_seed=source_seed,target_seed=target_seed)
        write(work.run/'behavior_fit.json',dict(rows=all_fit,fit_row_ids=fit_ids.tolist(),calibration_row_ids=cal_ids.tolist(),joint_refit=False))
        work.checks['all_rows']=len(work.metrics)==len(work.evaluation_ids)*(len(assets)*6+len(assets)*(len(assets)-1)*6)
        work.checks['unique']=len(work.metrics)==len({(r['source_seed'],r.get('target_seed'),r['factor'],r['method'],r['row_id']) for r in work.metrics})
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
