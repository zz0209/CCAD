"""Adapt a fixed public functional explanation through its actual programs.

The source and base LM stay frozen. A single target dictionary and nonnegative
eight-by-four relation execute all later requests. Natural reconstruction and
the source functional response use the existing joint-training objective.
"""
from pathlib import Path
import argparse, json, sys, time, traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module
from train_shift_response_space import project_rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    args=p.parse_args(); c=json.loads(args.config.read_text())
    files=['scripts/train_infinitive_program.py','scripts/train_shift_response_space.py',
           'scripts/run_published_parts.py','scripts/train_shift_dictionaries.py',
           'scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']
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
        old=np.load(w.checked(Path(c['relation_run'])/'relation.npz'))
        relations={k:torch.tensor(old[k],device=w.device,dtype=torch.long if k=='candidates' else torch.float32) for k in old.files}
        ix=(relations['native'].sum(1)>0).nonzero().flatten(); assert len(ix)==8
        base=relations['native'][ix].clone(); a=torch.nn.Parameter(base.clone())
        gain=torch.nn.Parameter(torch.ones(4,device=w.device),requires_grad=False)
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
                if mode=='student': h=h-(z[...,ix]*(a@q))@target.decoder.weight[:,ix].T
                elif mode in ['native','geometry','geometry_gain']:
                    h=h-(z*(relations[mode]@q))@target.decoder.weight.T
                elif mode=='raw_reconstruction':
                    reconstructed=target.decode(z)
                    h=h-(source_codes(reconstructed)*q)@sp['decoder']
                elif mode in ['native_tangent_relation_8','native_response_relation_8','tangent_gain']:
                    zs=source_codes(h)
                    if mode.startswith('native_tangent') or mode=='tangent_gain':
                        columns=-(target.encoder.weight@sp['decoder'].T)*zs.unsqueeze(-2)*(z>0).unsqueeze(-1)
                    else:
                        cf=h.unsqueeze(-2)-zs.unsqueeze(-1)*sp['decoder']
                        columns=(target.encode(cf)-z.unsqueeze(-2)).transpose(-1,-2)*(zs!=0).unsqueeze(-2)
                    score=columns.abs().sum(-1)*target.decoder.weight.norm(dim=0)
                    keep=torch.zeros_like(score).scatter(-1,score.topk(8,dim=-1).indices,1)
                    columns=columns*keep.unsqueeze(-1)
                    if mode=='tangent_gain': columns=columns*gain
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
                q=torch.tensor(v,device=w.device)
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
        # A checkpoint can be reevaluated independently without repeating training.
        for variant,folder in c.get('evaluate_checkpoints',{}).items():
            folder=Path(folder)
            target.load_state_dict(torch.load(w.checked(folder/'dictionary.pt'),map_location=w.device,weights_only=True))
            relation=np.load(w.checked(folder/'relation.npz'))['native']
            a.data.copy_(torch.tensor(relation,device=w.device)[ix])
            if variant=='tangent_gain': gain.data.copy_(torch.tensor(np.load(w.checked(folder/'source_gains.npy')),device=w.device))
            evaluate(variant,'tangent_gain' if variant=='tangent_gain' else 'native_tangent_relation_8' if variant.startswith('tangent_') else 'student'); material_quality(variant)
        trace=[]
        for variant in c['variants']:
            target.load_state_dict(initial); a.data.copy_(base)
            target.requires_grad_(variant not in ['mixed_relation','tangent_gain'])
            a.requires_grad_(variant not in ['natural_only','tangent_natural','tangent_mixed'])
            if variant=='tangent_gain': a.requires_grad_(False)
            gain.data.fill_(1.); gain.requires_grad_(variant=='tangent_gain')
            optimizer=torch.optim.AdamW([dict(params=[p for p in target.parameters() if p.requires_grad],lr=c['dictionary_lr']),dict(params=([a] if a.requires_grad else [])+([gain] if gain.requires_grad else []),lr=c['relation_lr'])],weight_decay=0.)
            gen=torch.Generator(device=w.device).manual_seed(c['training_seed'])
            natural_gen=torch.Generator(device=w.device).manual_seed(c['training_seed']+1)
            for step in range(c['steps']):
                v=(endpoints[step%3] if variant=='endpoints' else
                   endpoints[(step//2)%3] if ('mixed' in variant or variant=='tangent_gain') and step%2==0 else
                   torch.rand(2,generator=gen,device=w.device))
                q=expand(v)
                rr=[fit[(step*c['batch_sequences']+j)%len(fit)] for j in range(c['batch_sequences'])]
                if variant in ['natural_only','tangent_natural']:
                    program=torch.zeros((),device=w.device)
                else:
                    mode='source'
                    with torch.no_grad(): th,mask,tl=forward(rr)
                    mode='tangent_gain' if variant=='tangent_gain' else 'native_tangent_relation_8' if variant.startswith('tangent_') else 'student'
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
            folder=w.run/variant; folder.mkdir()
            torch.save(target.state_dict(),folder/'dictionary.pt')
            export=relations['native'].cpu().numpy().copy(); export[ix.cpu().numpy()]=a.detach().cpu().numpy()
            np.savez_compressed(folder/'relation.npz',native=export)
            if variant=='tangent_gain': np.save(folder/'source_gains.npy',gain.detach().cpu().numpy())
            evaluate(variant,'tangent_gain' if variant=='tangent_gain' else 'native_tangent_relation_8' if variant.startswith('tangent_') else 'student'); material_quality(variant)
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
