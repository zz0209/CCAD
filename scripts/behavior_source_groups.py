"""Source-only groups from shared natural next-token contribution profiles.

Positive attribution profiles and ordinary nonnegative matrix factorization
are discovery tools. Finite interventions, not NMF reconstruction or names,
determine whether the resulting groups have reusable functional structure.
"""
from __future__ import annotations
import json,time,gc
from run_r011s1_raw_hook_asset import ROOT
from ccad.artifacts import sha256


def prepare(cfg,run,reference,rc,checked,write,log):
    import numpy as np,torch,transformers
    f=cfg['behavior'];timer=time.perf_counter();tr=ROOT/rc['training_run'];tc=json.loads(checked(tr/'config.resolved.json').read_text())
    if f.get('gradient_cache'):
        p=checked(f['gradient_cache'])
        assert sha256(p)==f['gradient_cache_sha256']
        with np.load(p) as a:
            keep=a['packed_positions']%128>=f.get('minimum_context_position',0)
            ix=a['discovery_indices'][keep];G=torch.as_tensor(a['gradients'][keep],device='cuda:0')
            positions=a['packed_positions'][keep]
        meta=dict(discovery_rows=len(ix),forward_sequences=0,backward_batches=0,cache_path=str(p),
            gradient_sha256=sha256(p),minimum_context_position=f.get('minimum_context_position',0),wall_seconds=time.perf_counter()-timer)
        write(run/'behavior_gradient_summary.json',meta);log('BEHAVIOR_GRADIENT_CACHE',**meta)
        return dict(indices=ix,gradients=G,packed_positions=positions,metadata=meta)
    pm=json.loads(checked(rc['paired_manifest']).read_text());p=checked(pm['outputs']['discovery']['path']);assert sha256(p)==pm['outputs']['discovery']['sha256']
    tokens=np.fromfile(p,dtype='<u2').reshape(-1,128)
    with np.load(checked(reference/'natural_discovery_states.npz')) as a:packed=a['packed_positions'];cached=torch.as_tensor(a['hidden'],device='cuda:0')
    allowed=np.where((packed%128<127)&(packed//128<(len(tokens)//rc['batch_size'])*rc['batch_size']))[0]
    for fn in ['config.json','model.safetensors','tokenizer.json']:checked(ROOT/tc['model_local_dir']/fn)
    torch.set_float32_matmul_precision('high')
    model=transformers.AutoModelForCausalLM.from_pretrained(tc['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.requires_grad_(False);model.config.use_cache=False
    module=model.get_submodule(tc['hook_module_path']);grads=[];nll=[];labels=[];max_replay=0.;forwards=0
    for off in range(0,len(allowed),rc['batch_size']):
        select=allowed[off:off+rc['batch_size']];actual=len(select)
        if actual<rc['batch_size']:select=np.r_[select,np.repeat(select[0],rc['batch_size']-actual)]
        ids=torch.tensor(tokens[packed[select]//128].astype('int64'),device='cuda:0');pos=torch.tensor(packed[select]%128,device='cuda:0');idx=torch.arange(len(select),device='cuda:0');cache={}
        def hook(m,i,out):
            x=out[0] if isinstance(out,tuple) else out;x=x.detach().requires_grad_(True);cache['x']=x
            return (x,)+out[1:] if isinstance(out,tuple) else x
        handle=module.register_forward_hook(hook)
        try:lp=model(ids,use_cache=False).logits[idx,pos].log_softmax(-1)
        finally:handle.remove()
        label=ids[idx,pos+1];values=lp[idx,label];grad=torch.autograd.grad(values.sum(),cache['x'])[0][idx,pos]
        max_replay=max(max_replay,float((cache['x'][idx,pos].detach()-cached[select]).abs().max()));assert max_replay<cfg['functional']['hidden_replay_atol'],max_replay
        grads.append(grad[:actual].detach().cpu());nll.append(-values[:actual].detach().cpu());labels.append(label[:actual].cpu());forwards+=len(ids)
        del lp,values,grad,cache
        if off%512==0:log('BEHAVIOR_GRADIENTS',completed=min(off+len(select),len(allowed)),total=len(allowed),replay=max_replay)
        if time.perf_counter()-timer>f['gradient_budget_seconds']:raise TimeoutError('Behavior gradient budget exceeded')
    G=torch.cat(grads).to('cuda:0');np.savez_compressed(run/'behavior_gradients.npz',discovery_indices=allowed,packed_positions=packed[allowed],gradients=G.cpu().numpy(),nll=torch.cat(nll).numpy(),next_token=torch.cat(labels).numpy())
    del model,module;gc.collect();torch.cuda.empty_cache()
    meta=dict(discovery_rows=len(allowed),forward_sequences=forwards,backward_batches=len(grads),max_hidden_replay=max_replay,wall_seconds=time.perf_counter()-timer,gradient_sha256=sha256(run/'behavior_gradients.npz'))
    write(run/'behavior_gradient_summary.json',meta);log('BEHAVIOR_GRADIENTS_COMPLETE',**meta)
    return dict(indices=allowed,gradients=G,packed_positions=packed[allowed],metadata=meta)


def discover(cfg,run,z,d,response,objective,seed,write,log):
    import numpy as np,torch
    f=cfg['behavior'];timer=time.perf_counter();torch.set_float32_matmul_precision('highest')
    ix=response['indices'];zs=z[ix];active=(zs>0).sum(0);eligible=torch.where(active>=f['minimum_feature_activations'])[0]
    credit=(zs[:,eligible]*(response['gradients']@d[eligible].T)).clamp_min(0)
    # Normalize each feature profile for discovery; native interventions still
    # use its actual decoder and code, with no inverse-energy rescaling.
    norms=credit.square().sum(0).sqrt();keep=norms>1e-8;eligible=eligible[keep];credit=credit[:,keep];norms=norms[keep]
    X=credit/norms[None,:];rank=f['components'];rng=torch.Generator(device='cuda:0');rng.manual_seed(f['random_seed'])
    W=torch.rand((len(X),rank),generator=rng,device='cuda:0')+.1;H=torch.rand((rank,X.shape[1]),generator=rng,device='cuda:0')+.1
    trace=[]
    for step in range(f['nmf_iterations']):
        W*=((X@H.T)/(W@(H@H.T)).clamp_min(1e-12)).clamp_min(1e-12)
        H*=((W.T@X)/((W.T@W)@H).clamp_min(1e-12)).clamp_min(1e-12)
        scale=H.square().sum(1).sqrt().clamp_min(1e-12);H/=scale[:,None];W*=scale[None,:]
        if step%25==0 or step+1==f['nmf_iterations']:
            loss=float((X-W@H).square().sum()/X.square().sum());trace.append(dict(step=step+1,relative_error=loss))
    # Soft feature membership across behavioral components. The support size
    # is fixed before any target or calibration model outcome is inspected.
    membership=H/H.sum(0,keepdim=True).clamp_min(1e-12);result=[]
    source_energy=z.square().mean(0)*d.square().sum(1)
    for component in range(rank):
        scores=membership[component];local=torch.argsort(scores,descending=True,stable=True)[:cfg['source_pool']]
        sp=eligible[local];g=scores[local];g=g/g.max().clamp_min(1e-12)
        anchor=int(sp[torch.argmax(source_energy[sp]*g.square())]);result.append(dict(component_id=component,source_members=sp,source_gate=g,anchor=anchor))
        log('BEHAVIOR_SOURCE_GROUP',objective=objective,seed=seed,component=component,anchor=anchor,source_members=len(sp),weight_sum=float(g.sum()))
    np.savez_compressed(run/f'{objective}_s{seed}_behavior_nmf.npz',discovery_indices=ix,eligible_features=eligible.cpu().numpy(),
        W=W.cpu().numpy(),H=H.cpu().numpy(),feature_membership=membership.cpu().numpy(),positive_credit_norm=norms.cpu().numpy())
    write(run/f'{objective}_s{seed}_behavior_nmf.json',dict(trace=trace,source_seed=seed,objective=objective,source_only=True,
        eligible_features=len(eligible),components=rank,wall_seconds=time.perf_counter()-timer,method='Standard multiplicative-update NMF on feature-normalized positive observed-token contribution profiles. Fixed top membership source budget; native effects retain original code/decoder scales.'))
    return result
