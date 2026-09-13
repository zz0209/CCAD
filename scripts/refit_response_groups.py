"""Pointwise model-response fits of fixed source native groups.

Score samples form an unbiased Monte Carlo Fisher at each clean context.
All refits use the same fixed source group, target pool and discovery rows.
This is standard response-weighted bounded least squares, not a new solver.
"""
from __future__ import annotations
import json,time,gc
from run_r011s1_raw_hook_asset import ROOT
from ccad.artifacts import sha256


def refit(cfg,run,reference,rc,results,checked,write,log):
    import numpy as np,torch,transformers
    from ccad.native_coarsening import fit_groups,contribution_gram
    timer=time.perf_counter();f=cfg['response'];torch.set_float32_matmul_precision('high')
    tr=ROOT/rc['training_run'];tc=json.loads(checked(tr/'config.resolved.json').read_text())
    snaps=json.loads(checked(tr/'checkpoints.json').read_text())['checkpoints']
    pm=json.loads(checked(rc['paired_manifest']).read_text());p=checked(pm['outputs']['discovery']['path'])
    assert sha256(p)==pm['outputs']['discovery']['sha256']
    tokens=np.fromfile(p,dtype='<u2').reshape(-1,128)
    with np.load(checked(reference/'natural_discovery_states.npz')) as a:
        packed=a['packed_positions'];cached=torch.as_tensor(a['hidden'],device='cuda:0')
    Z={};D={};groups={}
    for obj in cfg['objectives']:
        for seed in cfg['seeds']:
            with np.load(checked(reference/f'natural_discovery_{obj}_seed{seed}.npz')) as a:
                z=torch.zeros(tuple(a['shape']),device='cuda:0')
                z[torch.as_tensor(a['rows'].astype('int64'),device='cuda:0'),torch.as_tensor(a['columns'].astype('int64'),device='cuda:0')]=torch.as_tensor(a['values'],device='cuda:0');Z[obj,seed]=z
            snap=next(s for s in snaps if s['objective']==obj and s['seed']==seed and s['step']==rc['checkpoint_step'])
            p=checked(snap['path']);assert sha256(p)==snap['sha256'];state=torch.load(p,map_location='cuda:0',weights_only=True)
            D[obj,seed]=(state['decoder.weight'].T if obj=='topk' else state['W_dec']).detach().contiguous()
    allowed=np.where((packed%128<127)&(packed//128<(len(tokens)//rc['batch_size'])*rc['batch_size']))[0]
    selected=set(int(i) for i in allowed[np.linspace(0,len(allowed)-1,f['uniform_contexts'],dtype=int)])
    for r in results:
        with np.load(checked(run/(r['query']+'_groups.npz'))) as a:groups[r['query']]={k:a[k] for k in a.files}
        z=Z[r['objective'],r['source_seed']]
        order=torch.argsort(z[allowed,r['anchor']],descending=True,stable=True).cpu().numpy()
        selected.update(int(i) for i in allowed[order[:f['active_contexts']]])
    selected=np.array(sorted(selected));write(run/'response_selection.json',dict(indices=selected.tolist(),rule=f,scope=cfg['scope']))
    log('RESPONSE_SELECTED',contexts=len(selected),score_draws=f['score_draws'])
    for fn in ['config.json','model.safetensors']:checked(ROOT/tc['model_local_dir']/fn)
    model=transformers.AutoModelForCausalLM.from_pretrained(tc['model_local_dir'],local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.requires_grad_(False);model.config.use_cache=False
    module=model.get_submodule(tc['hook_module_path']);grads=[];samples=[];max_replay=0.;calls=0
    rng=torch.Generator(device='cuda:0');rng.manual_seed(f['random_seed'])
    for off in range(0,len(selected),rc['batch_size']):
        select=selected[off:off+rc['batch_size']];actual=len(select)
        if actual<rc['batch_size']:select=np.r_[select,np.repeat(select[0],rc['batch_size']-actual)]
        ids=torch.tensor(tokens[packed[select]//128].astype('int64'),device='cuda:0')
        pos=torch.tensor(packed[select]%128,device='cuda:0');idx=torch.arange(len(select),device='cuda:0');cache={}
        def hook(m,i,out):
            x=out[0] if isinstance(out,tuple) else out;x=x.detach().requires_grad_(True);cache['x']=x
            return (x,)+out[1:] if isinstance(out,tuple) else x
        handle=module.register_forward_hook(hook)
        try:logits=model(ids,use_cache=False).logits[idx,pos]
        finally:handle.remove()
        lp=logits.log_softmax(-1);max_replay=max(max_replay,float((cache['x'][idx,pos].detach()-cached[select]).abs().max()))
        assert max_replay<cfg['functional']['hidden_replay_atol'],max_replay
        sampled=torch.multinomial(lp.detach().exp(),f['score_draws'],replacement=True,generator=rng)
        local=[]
        for k in range(f['score_draws']):
            g=torch.autograd.grad(lp[idx,sampled[:,k]].sum(),cache['x'],retain_graph=k+1<f['score_draws'])[0]
            local.append(g[idx,pos][:actual].detach())
        grads.append(torch.stack(local,1).cpu());samples.append(sampled[:actual].cpu());calls+=len(ids)
        del logits,lp,g,cache,local
        if off%128==0:log('RESPONSE_GRADIENTS',completed=min(off+rc['batch_size'],len(selected)),total=len(selected),replay=max_replay)
        if time.perf_counter()-timer>f['budget_seconds']:raise TimeoutError('Response gradient budget exceeded')
    G=torch.cat(grads).to('cuda:0');np.savez_compressed(run/'response_gradients.npz',indices=selected,packed_positions=packed[selected],
        gradients=G.cpu().numpy(),sampled_tokens=torch.cat(samples).numpy())
    del model,module;gc.collect();torch.cuda.empty_cache();torch.set_float32_matmul_precision('highest')
    log('RESPONSE_FIT_START',contexts=len(selected))
    fitrows=[]
    def fingerprints(z,d,g):return (z[:,None,:]*torch.einsum('nkh,fh->nkf',g,d)).reshape(-1,len(d))
    def blocks(bs,bt):return tuple(a.double().cpu().numpy() for a in (bs.T@bs/len(bs),bs.T@bt/len(bs),bt.T@bt/len(bs)))
    for r in results:
        obj=r['objective'];s=r['source_seed'];t=r['target_seed'];key=r['query'];a=groups[key]
        sp=torch.tensor(a['source_members'],device='cuda:0');tp=torch.tensor(a['target_members'],device='cuda:0')
        zs=Z[obj,s][selected][:,sp];zt=Z[obj,t][selected][:,tp];ds=D[obj,s][sp];dt=D[obj,t][tp]
        gs=a['source_gate'];old=a['target_gate'];ai=int(np.where(a['source_members']==r['anchor'])[0][0])
        eu=tuple(x.double().cpu().numpy() for x in (contribution_gram(zs,ds,zs,ds),contribution_gram(zs,ds,zt,dt),contribution_gram(zt,dt,zt,dt)))
        fi=blocks(fingerprints(zs,ds,G),fingerprints(zt,dt,G));per=G.roll(max(1,len(G)//2),0)
        sh=blocks(fingerprints(zs,ds,per),fingerprints(zt,dt,per));fits={};gates={}
        for name,kk in [('euclidean_same_rows',eu),('pointwise_fisher',fi),('shuffled_fisher',sh)]:
            _,gt,info=fit_groups(*kk,ai,penalty=cfg['penalty'],maxiter=cfg['maxiter'],fixed_source=gs)
            gates[name]=gt;fits[name]=info
        ss,st,tt=fi;upper=1/max(float(old.max()),1e-12);scale=np.clip(float(gs@st@old)/max(float(old@tt@old),1e-12),0,upper)
        gates['fisher_amplitude']=old*scale;gates['euclidean_full_rows']=old.copy()
        # Preserve the original Euclidean-optimal singleton for this source.
        gates['euclidean_best_atom']=a['best_atom_target_gate'].copy()
        cross=gs@st;gain=np.clip(cross/np.diag(tt).clip(1e-12),0,1)
        errors=gs@ss@gs-2*gain*cross+gain*gain*np.diag(tt);best=int(errors.argmin());one=np.zeros(len(tp));one[best]=gain[best]
        gates['fisher_best_atom']=one
        original=run/(key+'_groups.euclidean.npz');np.savez_compressed(original,**a)
        a['target_gate']=gates['pointwise_fisher'];a['best_atom_target_gate']=one
        for name,gt in gates.items():a['extra_gate_'+name]=gt
        np.savez_compressed(run/(key+'_groups.npz'),**a)
        scores={}
        for name,gt in gates.items():
            scores[name]={metric:dict(error=float(gs@ks@gs-2*gs@kst@gt+gt@kt@gt),source_energy=float(gs@ks@gs),
                relative_error=float((gs@ks@gs-2*gs@kst@gt+gt@kt@gt)/max(float(gs@ks@gs),1e-12))) for metric,(ks,kst,kt) in [('euclidean',eu),('fisher',fi)]}
        row=dict(query=key,objective=obj,source_seed=s,target_seed=t,anchor=r['anchor'],fits=fits,amplitude=float(scale),scores=scores)
        fitrows.append(row);log('RESPONSE_FIT_COMPLETE',query=key,fits=fits)
        with (run/'metrics.raw.jsonl').open('a') as out:out.write(json.dumps(dict(response_refit=row))+'\n')
    write(run/'response_fits.json',dict(queries=fitrows,contexts=len(selected),gradient_file_sha256=sha256(run/'response_gradients.npz')))
    return dict(contexts=len(selected),score_draws=f['score_draws'],forward_sequences=calls,backward_batches=f['score_draws']*len(grads),
        wall_seconds=time.perf_counter()-timer,max_hidden_replay=max_replay,peak_cuda_bytes=torch.cuda.max_memory_allocated())
