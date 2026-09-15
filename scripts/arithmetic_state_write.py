"""Recipient-conditioned functional writing with matched code-state information."""
import time
import numpy as np
import torch
from run_causalgym_multisite import ROOT, write


def context_coordinates(z, arity, mean, projection, clip):
    return torch.einsum('...m,...mk->...k',z-mean[arity],projection[arity]).clamp(-clip,clip)


def state_direction(base, modulation, context, arity, mode, scale, cap):
    direction=base[arity]
    if mode=='scalar':
        direction=direction*(1+(context*modulation[arity]).sum(-1,keepdim=True))
    elif mode=='direction':
        direction=direction+torch.einsum('...k,...km->...m',context,modulation[arity])
    direction=direction*scale[arity]
    return direction*(cap[arity]/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp_max(1)


@torch.enable_grad()
def fit_state_write(w,cfg,model,module,tok,ae,rows,tokenrows,codes,hidden,stored,budget):
    spec=cfg['source_function_refit']['state_write'];rank=spec['context_rank']
    with np.load(w.checked(ROOT/spec['schedule_run']/'counterfactual_seed1_k64_carry_function_arity_scalar.npz')) as f:schedule=f['fit_schedule']
    assert schedule.shape==(256,cfg['batch_size'],2)
    assert len(rows)==405 and all(r['split']=='fit' and r['template']==0 for r in rows)
    ids=torch.tensor(stored['readwrite_code_64'],device=w.device)
    reader=torch.tensor(stored['readwrite_code_64_readout'],device=w.device)
    arities=torch.tensor([int('c' in r) for r in rows],device=w.device)
    x=codes[:,0,ids];decoder=ae.decoder.weight.T[ids]
    means=[];projections=[];variance=[]
    for arity in range(2):
        xx=x[arities==arity].double();mean=xx.mean(0);centered=xx-mean
        _,singular,vh=torch.linalg.svd(centered,full_matrices=False)
        basis=vh[:rank].T
        signs=torch.sign(basis[basis.abs().argmax(0),torch.arange(rank,device=w.device)])
        basis=basis*signs
        projection=basis/(singular[:rank]/np.sqrt(len(xx))).clamp_min(.01)
        means.append(mean.float());projections.append(projection.float())
        variance.append(float(singular[:rank].square().sum()/singular.square().sum()))
    mean=torch.stack(means);projection=torch.stack(projections)
    np.savez_compressed(w.run/'STATE_CONTEXT.npz',member_ids=ids.cpu().numpy(),mean=mean.cpu().numpy(),projection=projection.cpu().numpy(),explained_variance=variance)
    payload=dict(stored);batch=cfg['batch_size'];ix=torch.arange(batch,device=w.device)
    physical_length=max(map(len,tokenrows))+cfg['max_new_tokens']
    for tag in spec['variants']:
        mode,space=tag.split('_')[1:];raw=space=='raw'
        initial=torch.tensor(stored['readwrite_raw_writer' if raw else 'readwrite_code_64_writer'],device=w.device)
        scale=initial.norm(dim=-1,keepdim=True).clamp_min(1e-6)/np.sqrt(initial.shape[-1])
        cap=spec['norm_multiple']*initial.norm(dim=-1,keepdim=True)
        base=torch.nn.Parameter(initial/scale)
        shape=(2,rank) if mode=='scalar' else (2,rank,initial.shape[-1])
        modulation=torch.nn.Parameter(torch.zeros(shape,device=w.device)) if mode!='constant' else None
        params=[base]+([modulation] if modulation is not None else [])
        optimizer=torch.optim.Adam(params,lr=spec['lr']);trace=[];started=time.perf_counter()

        def objective(pairs,values):
            ii,jj=torch.as_tensor(pairs,device=w.device).T
            assert bool((arities[ii]==arities[jj]).all())
            targets=[];positions=[]
            tokens=torch.full((batch,physical_length),tok.eos_token_id,device=w.device,dtype=torch.long)
            attention=torch.zeros_like(tokens)
            for b,(i,j) in enumerate(pairs):
                answer=rows[i]['total']+10*(int(rows[j]['carry'])-int(rows[i]['carry']))
                out=tok.encode(str(answer),add_special_tokens=False)
                assert 10<=answer<100 and len(out)==2
                seq=tokenrows[i]+out;tokens[b,:len(seq)]=torch.tensor(seq,device=w.device);attention[b,:len(seq)]=1
                positions.append(len(tokenrows[i])-1);targets.append(out+tok.encode('\n',add_special_tokens=False))
            pos=torch.tensor(positions,device=w.device);target=torch.tensor(targets,device=w.device)
            sites=pos[:,None]+torch.arange(3,device=w.device)
            def hook(_m,_a,output):
                h=output[0] if isinstance(output,tuple) else output
                z=ae.encode(h[ix,pos])[:,ids]
                context=context_coordinates(z,arities[ii],mean,projection,spec['context_clip'])
                direction=state_direction(values[0],values[1] if len(values)>1 else None,context,arities[ii],mode,scale,cap)
                query=((x[jj]-z)*reader[arities[ii]]).sum(-1,keepdim=True)
                coeff=query*direction
                delta=coeff if raw else ((z+coeff).clamp_min(0)-z)@decoder
                edited=h.clone();edited[ix,pos]+=delta
                return (edited,)+output[1:] if isinstance(output,tuple) else edited
            handle=module.register_forward_hook(hook)
            try:
                output=model(tokens,attention_mask=attention,use_cache=False)
                logits=output.logits[ix[:,None],sites]
                w.sequence_forwards+=batch;w.token_forwards+=batch*physical_length
                return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.flatten())
            finally:handle.remove()

        loss=objective(schedule[0],params);grads=torch.autograd.grad(loss,params)
        checked=[]
        for block,gradient in enumerate(grads):
            coord=int(gradient.flatten().abs().argmax());analytic=float(gradient.flatten()[coord]);eps=.01
            with torch.no_grad():
                plus=[p.detach().clone() for p in params];minus=[p.detach().clone() for p in params]
                plus[block].flatten()[coord]+=eps;minus[block].flatten()[coord]-=eps
                finite=float((objective(schedule[0],plus)-objective(schedule[0],minus))/(2*eps))
            assert abs(finite-analytic)<.015+.05*abs(analytic),(tag,block,finite,analytic)
            checked.append(dict(block=block,coordinate=coord,analytic=analytic,finite=finite,error=abs(finite-analytic)))
        for step,pairs in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True);loss=objective(pairs,params)
            assert bool(torch.isfinite(loss));loss.backward();optimizer.step()
            if (step+1)%32==0:
                trace.append(dict(step=step+1,loss=float(loss.detach())))
                w.progress('STATE_WRITE_FIT',method=tag,**trace[-1])
            budget()
        payload[tag]=ids.cpu().numpy()
        for suffix,value in [('readout',reader),('writer',base.detach()),('context_mean',mean),('context_projection',projection),('write_scale',scale),('write_cap',cap)]:payload[tag+'_'+suffix]=value.cpu().numpy()
        if modulation is not None:payload[tag+'_modulation']=modulation.detach().cpu().numpy()
        payload[tag+'_context_clip']=np.array(spec['context_clip']);payload[tag+'_context_mode']=np.array(mode)
        payload[tag+'_operator']=np.array('state_raw' if raw else 'state_code');payload[tag+'_nonnegative_update']=np.array(not raw)
        np.savez_compressed(w.run/f'{tag}_fit.npz',initial=initial.cpu().numpy(),schedule=schedule,**{k:payload[k] for k in payload if k==tag or k.startswith(tag+'_')})
        write(w.run/f'{tag}_fit.json',dict(method=tag,steps=256,lr=spec['lr'],parameters=sum(p.numel() for p in params),gradient_checks=checked,trace=trace,fit_seconds=time.perf_counter()-started,
            context='Top8 PCA coordinates, centered and whitened within arity on405 source-fit64-code states, clipped at5; same input information for all methods.',
            scope='Same fixed64-code reader, warm-start native/raw writers; new256 output-supervised batches. Per-input write norm capped at4times warm-start norm; raw and code capacities differ. No new source members.'))
        w.checks['state_write_gradient_'+tag]=True
    return payload
