"""Adapt a fixed public functional explanation through its actual programs.

The source and base LM stay frozen. Configurations compare a fixed nonnegative
relation, encoder-derived columns and independent intervention maps. Source
program responses supervise the target execution while requests remain callable.
"""
from pathlib import Path
import argparse, json, sys, time, traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module
from train_shift_response_space import project_rows
from ccad.intervention_transport import transport_delta,refine_columns


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    args=p.parse_args(); c=json.loads(args.config.read_text())
    files=['scripts/train_infinitive_program.py','scripts/train_shift_response_space.py',
           'scripts/run_published_parts.py','scripts/train_shift_dictionaries.py',
           'scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py',
           'src/ccad/intervention_transport.py']
    w=MultisiteWork(c,args.config,files); handle=None; error=None
    try:
        import torch, transformers
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest'); torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats(); w.torch=torch; w.device=torch.device(c['device'])
        w.environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,
                           transformers=transformers.__version__,cpu_threads=2)
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py','Pinned TopK implementation','MIT')
        sb=np.load(w.checked(c['source_parameters'],'Four public Figure18 features','MIT'))
        sp={k:torch.tensor(sb[k],device=w.device) for k in ['encoder','encoder_bias','decoder','center']}
        training_requests=None
        if c.get('training_request_panel'):
            request_data=json.loads(w.checked(c['training_request_panel'],'Source-only training request design').read_text())
            assert request_data['members']=={'resid_4':[0,1,2,3]}
            training_requests=torch.tensor(request_data['requests']['resid_4'],device=w.device,dtype=torch.float32)
            assert training_requests.shape==(c['steps'],4) and bool(torch.isfinite(training_requests).all()) and bool(((training_requests>=0)&(training_requests<=1)).all())
            write(w.run/'training_request_design.json',request_data)
        for f in ['model.safetensors','config.json','tokenizer.json']: w.checked(Path(c['model_local_dir'])/f)
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,
            dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False); model.config.use_cache=False
        tok=transformers.AutoTokenizer.from_pretrained(c['model_local_dir'],local_files_only=True)
        answer=tok.encode(' to',add_special_tokens=False); assert len(answer)==1
        fit=json.loads(w.checked(c['fit_panel']).read_text())['rows']
        panel=json.loads(w.checked(c['panel']).read_text()); rows=panel['rows']; queries=panel['queries']
        assert not {r['text'] for r in fit}&{r['text'] for r in rows}
        assert not {r['pair'].split(':')[0] for r in fit}&{r['pair'].split(':')[0] for r in rows}
        for r in fit+rows: r['tokens']=tok.encode(r['text'],add_special_tokens=False)
        initial=torch.load(w.checked(Path(c['target_directory'])/f'resid_4_seed{c["target_seed"]}.pt'),map_location=w.device,weights_only=True)
        target=AutoEncoderTopK(512,initial['encoder.weight'].shape[0],int(initial['k'])).to(w.device)
        target.load_state_dict(initial)
        training_targets=[]
        for entry in c.get('source_column_training_targets',[]):
            state=torch.load(w.checked(Path(entry['directory'])/f'resid_4_seed{entry["seed"]}.pt'),map_location=w.device,weights_only=True)
            assert state['encoder.weight'].shape==initial['encoder.weight'].shape
            assert int(state['k'])==int(initial['k'])
            training_targets.append(state)
        training_schedule=[]
        if training_targets:
            assert c['steps']%(2*len(training_targets))==0
            training_schedule=np.random.default_rng(c['training_seed']+17).permutation(np.arange(c['steps']//2)%len(training_targets)).tolist()
            write(w.run/'training_dictionary_schedule.json',dict(block_length=2,indices=training_schedule,targets=c['source_column_training_targets']))
        old=np.load(w.checked(Path(c['relation_run'])/'relation.npz'))
        relations={k:torch.tensor(old[k],device=w.device,dtype=torch.long if k=='candidates' else torch.float32) for k in old.files}
        ix=(relations['native'].sum(1)>0).nonzero().flatten(); assert len(ix)==8
        base=relations['native'][ix].clone(); a=torch.nn.Parameter(base.clone())
        gain=torch.nn.Parameter(torch.ones(4,device=w.device),requires_grad=False)
        write_matrix=torch.nn.Parameter(initial['encoder.weight'].clone(),requires_grad=False)
        source_columns=torch.nn.Parameter(sp['decoder'].clone(),requires_grad=False)
        states=np.load(w.checked(c['natural_states'],'Existing natural discovery states'))['h']
        assert len(states)>=4096
        natural=torch.tensor(states,device=w.device)
        mode='none'; q=torch.ones(4,device=w.device)
        def source_codes(h): return torch.relu((h-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
        def hook(module,inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            if mode=='source': h=h-(source_codes(h)*q)@sp['decoder']
            elif mode!='none':
                z=target.encode(h)
                if mode.startswith('transport_'):
                    delta,_,columns=transport_delta(h,target,sp,q,write_matrix,8,active_only='active' in mode)
                    if mode=='transport_refined_open':
                        columns=refine_columns(h,target,sp,columns,8,c.get('refine_steps',64),write_matrix)
                        delta=(columns@q)@target.decoder.weight.T
                    h=h+delta
                elif mode=='student': h=h-(z[...,ix]*(a@q))@target.decoder.weight[:,ix].T
                elif mode in ['native','geometry','geometry_gain']:
                    h=h-(z*(relations[mode]@q))@target.decoder.weight.T
                elif mode=='raw_reconstruction':
                    reconstructed=target.decode(z)
                    h=h-(source_codes(reconstructed)*q)@sp['decoder']
                elif mode in ['native_tangent_relation_8','native_response_relation_8','tangent_gain'] or mode.startswith('source_columns_'):
                    zs=source_codes(h)
                    if mode.startswith(('native_tangent','source_columns_')) or mode=='tangent_gain':
                        decoder=(sp['decoder']*gain[:,None] if mode=='source_columns_scalar_mixed' else
                                 source_columns if mode.startswith('source_columns_') else sp['decoder'])
                        columns=-(target.encoder.weight@decoder.T)*zs.unsqueeze(-2)*(z>0).unsqueeze(-1)
                    else:
                        cf=h.unsqueeze(-2)-zs.unsqueeze(-1)*sp['decoder']
                        columns=(target.encode(cf)-z.unsqueeze(-2)).transpose(-1,-2)*(zs!=0).unsqueeze(-2)
                    if mode=='tangent_gain' and c.get('gain_before_selection',False): columns=columns*gain
                    score=columns.abs().sum(-1)*target.decoder.weight.norm(dim=0)
                    keep=torch.zeros_like(score).scatter(-1,score.topk(8,dim=-1).indices,1)
                    columns=columns*keep.unsqueeze(-1)
                    if mode=='tangent_gain' and not c.get('gain_before_selection',False): columns=columns*gain
                    positive,negative=columns.clamp_min(0),(-columns).clamp_min(0)
                    scale=(z/negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
                    h=h+((positive-negative*scale.unsqueeze(-1))@q)@target.decoder.weight.T
                else: raise ValueError(mode)
            return (h,*output[1:]) if isinstance(output,tuple) else h
        handle=site_module(model,'resid_4').register_forward_hook(hook)
        def batch(rr):
            length=max(len(r['tokens']) for r in rr)
            ids=torch.zeros((len(rr),length),device=w.device,dtype=torch.long); mask=torch.zeros_like(ids)
            for j,r in enumerate(rr): ids[j,:len(r['tokens'])]=torch.tensor(r['tokens'],device=w.device); mask[j,:len(r['tokens'])]=1
            return ids,mask
        projection_checked=False
        def forward(rr):
            nonlocal projection_checked
            ids,mask=batch(rr)
            hidden=model.gpt_neox(ids,attention_mask=mask,use_cache=False).last_hidden_state
            last=hidden[torch.arange(len(ids),device=w.device),mask.sum(1)-1]
            lp=torch.log_softmax(model.get_output_embeddings()(last),dim=-1)[:,answer[0]]
            if c.get('projection_witness') and not projection_checked and len(rr)==c['eval_batch_size']:
                with torch.no_grad():
                    legacy=model.get_output_embeddings()(hidden)[torch.arange(len(ids),device=w.device),mask.sum(1)-1]
                    legacy=torch.log_softmax(legacy,dim=-1)[:,answer[0]]
                    write(w.run/'projection_witness.json',dict(max_abs_difference=float((lp-legacy).abs().max()),
                        current=lp.detach().cpu().tolist(),legacy=legacy.cpu().tolist(),
                        cause_test='Identical hidden tensor and output weights, full-token versus final-token matrix multiplication',rows=[r['text'] for r in rr]))
                projection_checked=True
            w.sequence_forwards+=len(ids); w.token_forwards+=int(mask.sum())
            return hidden,mask,lp
        def token_mse(delta,mask): return (delta.square()*mask[...,None]).sum()/(mask.sum()*delta.shape[-1])
        endpoints=torch.tensor([[1,0],[0,1],[1,1]],device=w.device,dtype=torch.float32)
        def expand(v): return v[torch.tensor([0,0,0,1],device=w.device)]
        scales=[]
        with torch.no_grad():
            for off in range(0,len(fit),c['batch_sequences']):
                rr=fit[off:off+c['batch_sequences']]; mode='none'; ch,mask,cl=forward(rr)
                for v in endpoints:
                    q=expand(v); mode='source'; th,_,tl=forward(rr)
                    scales.append([float(token_mse(th-ch,mask)),float((tl-cl).square().mean())])
        energy=np.maximum(np.mean(scales,axis=0),1e-8)
        write(w.run/'loss_scales.json',dict(final_hidden=float(energy[0]),source_response=float(energy[1]),contexts=len(fit),queries=endpoints.cpu().tolist()))
        values={}; hidden_scores={}; quality=[]
        @torch.no_grad()
        def evaluate(name,execution=None):
            nonlocal mode,q
            mode=execution or name
            lp=np.empty((len(queries),len(rows)),dtype='float32')
            for qi,(nameq,v) in enumerate(queries.items()):
                q=torch.tensor(v,device=w.device,dtype=torch.float32)
                for off in range(0,len(rows),c['eval_batch_size']):
                    _,_,l=forward(rows[off:off+c['eval_batch_size']]); lp[qi,off:off+len(l)]=l.cpu().numpy()
                if (qi+1)%9==0: w.progress('EVALUATION',method=name,query=qi+1,total_queries=len(queries))
            values[name]=lp
            np.savez_compressed(w.run/'responses.npz',**values)
        @torch.no_grad()
        def material_quality(name):
            h=natural[-1024:]; z=target.encode(h); rec=target.decode(z)
            quality.append(dict(method=name,fve=float(1-(rec-h).square().sum()/(h-h.mean(0)).square().sum()),
                l0=float((z>0).sum(-1).float().mean()),alive_in_panel=int((z>0).any(0).sum()),
                selected_active=float((z[:,ix]>0).sum(-1).float().mean()),
                decoder_norm_error=float((target.decoder.weight.norm(dim=0)-1).abs().max())))
        evaluate('none'); evaluate('source'); material_quality('initial')
        for method in c['baselines']: evaluate(method)
        for variant,path in c.get('evaluate_source_columns',{}).items():
            assert variant.startswith('source_columns_')
            source_columns.data.copy_(torch.tensor(np.load(w.checked(path))['decoder'],device=w.device))
            evaluate(variant)
        # A checkpoint can be reevaluated independently without repeating training.
        for variant,folder in c.get('evaluate_checkpoints',{}).items():
            folder=Path(folder)
            target.load_state_dict(torch.load(w.checked(folder/'dictionary.pt'),map_location=w.device,weights_only=True))
            relation=np.load(w.checked(folder/'relation.npz'))['native']
            a.data.copy_(torch.tensor(relation,device=w.device)[ix])
            if variant=='tangent_gain': gain.data.copy_(torch.tensor(np.load(w.checked(folder/'source_gains.npy')),device=w.device))
            if variant.startswith('transport_'):
                write_matrix.data.copy_(torch.load(w.checked(folder/'write_matrix.pt'),map_location=w.device,weights_only=True))
            evaluate(variant,variant if variant.startswith('transport_') else 'tangent_gain' if variant=='tangent_gain' else 'native_tangent_relation_8' if variant.startswith('tangent_') else 'student'); material_quality(variant)
        trace=[]
        for variant in c['variants']:
            target.load_state_dict(initial); a.data.copy_(base)
            independent=variant.startswith('transport_')
            shared_columns=variant.startswith('source_columns_')
            scalar_columns=variant=='source_columns_scalar_mixed'
            assert not training_targets or shared_columns
            target.requires_grad_(not independent and not shared_columns and variant not in ['mixed_relation','tangent_gain'])
            a.requires_grad_(not independent and not shared_columns and variant not in ['natural_only','tangent_natural','tangent_mixed'])
            if variant=='tangent_gain': a.requires_grad_(False)
            gain.data.fill_(1.); gain.requires_grad_(variant=='tangent_gain' or scalar_columns)
            write_matrix.data.copy_(initial['encoder.weight']); write_matrix.requires_grad_(independent)
            source_columns.data.copy_(sp['decoder']);source_columns.requires_grad_(shared_columns and not scalar_columns)
            optimizer=torch.optim.AdamW([dict(params=[p for p in target.parameters() if p.requires_grad]+([write_matrix] if independent else []),lr=c['dictionary_lr']),dict(params=([a] if a.requires_grad else [])+([gain] if gain.requires_grad else []),lr=c['relation_lr']),dict(params=[source_columns] if source_columns.requires_grad else [],lr=c.get('source_column_lr',.001))],weight_decay=0.)
            gen=torch.Generator(device=w.device).manual_seed(c['training_seed'])
            natural_gen=torch.Generator(device=w.device).manual_seed(c['training_seed']+1)
            for step in range(c['steps']):
                if training_targets:
                    # 随机安排字典，保持每个字典的端点、连续请求和更新次数平衡。
                    target.load_state_dict(training_targets[training_schedule[step//2]])
                v=(endpoints[step%3] if variant=='endpoints' else
                   endpoints[(step//2)%3] if ('mixed' in variant or variant=='tangent_gain') and step%2==0 else
                   torch.rand(2,generator=gen,device=w.device))
                q=expand(v)
                if training_requests is not None:q=training_requests[step]
                rr=[fit[(step*c['batch_sequences']+j)%len(fit)] for j in range(c['batch_sequences'])]
                if variant in ['natural_only','tangent_natural']:
                    program=torch.zeros((),device=w.device)
                else:
                    mode='source'
                    with torch.no_grad(): th,mask,tl=forward(rr)
                    mode=variant if independent or shared_columns else 'tangent_gain' if variant=='tangent_gain' else 'native_tangent_relation_8' if variant.startswith('tangent_') else 'student'
                    sh,_,sl=forward(rr)
                    state_loss=token_mse(sh-th,mask)/energy[0]
                    response_loss=(sl-tl).square().mean()/energy[1]
                    program=(1-c['source_response_weight'])*state_loss+c['source_response_weight']*response_loss
                h=natural[torch.randint(len(natural)-1024,(c['natural_batch_states'],),generator=natural_gen,device=w.device)]
                recon=(target(h)-h).square().mean()/h.square().mean().clamp_min(1e-8)
                loss=program+c['reconstruction_weight']*recon
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],1.)
                optimizer.step()
                with torch.no_grad():
                    if target.decoder.weight.requires_grad: target.decoder.weight.div_(target.decoder.weight.norm(dim=0).clamp_min(1e-10))
                    project_rows(a)
                    gain.clamp_(min=0)
                assert torch.isfinite(loss)
                if (step+1)%32==0:
                    point=dict(method=variant,step=step+1,loss=float(loss.detach()),program=float(program.detach()),reconstruction=float(recon.detach()))
                    trace.append(point); w.progress('TRAINING',**point)
                if time.perf_counter()-w.wall_start>c['budget_seconds']: raise TimeoutError('Allocated driver budget reached')
            if training_targets:
                target.load_state_dict(initial)
            folder=w.run/variant; folder.mkdir()
            torch.save(target.state_dict(),folder/'dictionary.pt')
            if independent:
                torch.save(write_matrix.detach().cpu(),folder/'write_matrix.pt')
                unchanged=all(torch.equal(target.state_dict()[key],value) for key,value in initial.items())
                w.checks['dictionary_unchanged_'+variant]=unchanged
                with torch.no_grad():
                    probe=natural[-64:]
                    _,dz,columns=transport_delta(probe,target,sp,torch.ones(4,device=w.device),write_matrix,8,active_only='active' in variant)
                    zero,_,_=transport_delta(probe,target,sp,torch.zeros(4,device=w.device),write_matrix,8,active_only='active' in variant)
                    w.checks['feasible_'+variant]=float((target.encode(probe)+dz).min())>=-1e-5
                    w.checks['zero_query_'+variant]=bool((zero==0).all())
                    w.checks['member_budget_'+variant]=int((columns!=0).any(-1).sum(-1).max())<=8
            export=relations['native'].cpu().numpy().copy(); export[ix.cpu().numpy()]=a.detach().cpu().numpy()
            np.savez_compressed(folder/'relation.npz',native=export)
            if variant=='tangent_gain': np.save(folder/'source_gains.npy',gain.detach().cpu().numpy())
            if shared_columns:
                exported_columns=sp['decoder']*gain[:,None] if scalar_columns else source_columns
                np.savez_compressed(folder/'source_columns.npz',decoder=exported_columns.detach().cpu().numpy())
                w.checks['dictionary_unchanged_'+variant]=all(torch.equal(target.state_dict()[k],v) for k,v in initial.items())
                w.checks['relation_unchanged_'+variant]=torch.equal(a,base)
            evaluate(variant,variant if independent or shared_columns else 'tangent_gain' if variant=='tangent_gain' else 'native_tangent_relation_8' if variant.startswith('tangent_') else 'student'); material_quality(variant)
            if variant.startswith('tangent_'): evaluate('raw_reconstruction_after_'+variant,'raw_reconstruction')
            del optimizer
        write(w.run/'INDEX.json',dict(methods=list(values),queries=list(queries),query_masks=queries,rows=rows,fit_rows=fit,source_answer_id=answer[0],target_indices=ix.cpu().tolist()))
        write(w.run/'quality.json',quality); write(w.run/'training.json',trace)
        for name,lp in values.items():
            if name in ['none','source']: continue
            errors=np.sqrt(((lp-values['source'])**2).mean(1)/((values['source']-values['none'])**2).mean(1))
            for query,value in zip(queries,errors):
                w.record(kind='response_error',task='infinitive',row_id=query,component='all_evaluation_contexts',
                         method=name,operation=query,target_seed=c['target_seed'],nrmse=float(value))
        w.checks.update(finite=all(np.isfinite(v).all() for v in values.values()),capacity=float(a.detach().sum(1).max())<=1.00001,
                       nonnegative=float(a.detach().min())>=0,disjoint_fit_evaluation=True)
    except Exception: error=traceback.format_exc()
    finally:
        if handle is not None: handle.remove()
    return w.finish(error)


if __name__=='__main__': raise SystemExit(main())
