"""Fit native deletion fractions against actual finite source effects."""
from __future__ import annotations
import json,time,gc
from run_r011s1_raw_hook_asset import ROOT
from ccad.artifacts import sha256


def finite_refit(cfg,run,reference,rc,results,checked,write,log):
    import numpy as np,torch,transformers
    from ccad.native_coarsening import fit_groups,contribution_gram
    f=cfg['finite_refit'];timer=time.perf_counter();parent=ROOT/f['parent_run']
    assert json.loads(checked(parent/'status.json').read_text())['status']=='PASS'
    with np.load(checked(parent/'response_gradients.npz')) as a:
        all_ix=a['indices'];G=torch.as_tensor(a['gradients'],device='cuda:0')
    tr=ROOT/rc['training_run'];tc=json.loads(checked(tr/'config.resolved.json').read_text());snaps=json.loads(checked(tr/'checkpoints.json').read_text())['checkpoints']
    pm=json.loads(checked(rc['paired_manifest']).read_text());p=checked(pm['outputs']['discovery']['path']);assert sha256(p)==pm['outputs']['discovery']['sha256']
    tokens=np.fromfile(p,dtype='<u2').reshape(-1,128)
    with np.load(checked(reference/'natural_discovery_states.npz')) as a:packed=a['packed_positions'];cached=torch.as_tensor(a['hidden'],device='cuda:0')
    Z={};D={}
    for obj in cfg['objectives']:
        for seed in cfg['seeds']:
            with np.load(checked(reference/f'natural_discovery_{obj}_seed{seed}.npz')) as a:
                z=torch.zeros(tuple(a['shape']),device='cuda:0');z[torch.as_tensor(a['rows'].astype('int64'),device='cuda:0'),torch.as_tensor(a['columns'].astype('int64'),device='cuda:0')]=torch.as_tensor(a['values'],device='cuda:0');Z[obj,seed]=z
            snap=next(s for s in snaps if s['objective']==obj and s['seed']==seed and s['step']==rc['checkpoint_step']);p=checked(snap['path']);assert sha256(p)==snap['sha256']
            state=torch.load(p,map_location='cuda:0',weights_only=True);D[obj,seed]=(state['decoder.weight'].T if obj=='topk' else state['W_dec']).contiguous()
    for fn in ['config.json','model.safetensors']:checked(ROOT/tc['model_local_dir']/fn)
    torch.set_float32_matmul_precision('high')
    model=transformers.AutoModelForCausalLM.from_pretrained(tc['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.requires_grad_(False);model.config.use_cache=False
    module=model.get_submodule(tc['hook_module_path']);calls=0;max_replay=0.;fitrows=[]
    def output(select,delta):
        nonlocal calls,max_replay
        ids=torch.tensor(tokens[packed[select]//128].astype('int64'),device='cuda:0');pos=torch.tensor(packed[select]%128,device='cuda:0');idx=torch.arange(len(select),device='cuda:0')
        def hook(m,i,out):
            nonlocal max_replay
            x=out[0] if isinstance(out,tuple) else out
            max_replay=max(max_replay,float((x[idx,pos].detach()-cached[select]).abs().max()))
            y=x.clone();y[idx,pos]-=delta
            return (y,)+out[1:] if isinstance(out,tuple) else y
        handle=module.register_forward_hook(hook)
        try:lp=model(ids,use_cache=False).logits[idx,pos].log_softmax(-1)
        finally:handle.remove()
        calls+=len(ids);return lp
    for r in results:
        key=r['query'];obj=r['objective'];s=r['source_seed'];t=r['target_seed']
        with np.load(checked(parent/(key+'_groups.npz'))) as a:group={k:a[k] for k in a.files}
        # The pre-existing source operation and complete target pool are fixed.
        with np.load(checked(run/(key+'_groups.npz'))) as a:
            assert np.array_equal(a['source_gate'],group['source_gate']) and np.array_equal(a['source_members'],group['source_members'])
        zs=Z[obj,s];zt=Z[obj,t];sp=torch.tensor(group['source_members'],device='cuda:0');tp=torch.tensor(group['target_members'],device='cuda:0')
        ds=D[obj,s][sp];dt=D[obj,t][tp];gs=group['source_gate'];smt=torch.tensor(gs,dtype=zs.dtype,device='cuda:0');old=group['extra_gate_euclidean_full_rows'];ai=int(np.where(group['source_members']==r['anchor'])[0][0])
        order=torch.argsort(zs[all_ix,r['anchor']],descending=True,stable=True).cpu().numpy()[:f['active_contexts']]
        grid=np.linspace(0,len(all_ix)-1,f['uniform_contexts'],dtype=int);local=np.r_[order,grid];select=all_ix[local]
        xs=zs[select][:,sp];xt=zt[select][:,tp];g=G[local];source=(xs*smt)@ds
        torch.set_float32_matmul_precision('highest')
        eu=tuple(x.double().cpu().numpy() for x in (contribution_gram(xs,ds,xs,ds),contribution_gram(xs,ds,xt,dt),contribution_gram(xt,dt,xt,dt)))
        fs=(xs[:,None,:]*torch.einsum('nkh,fh->nkf',g,ds)).reshape(-1,len(ds));ft=(xt[:,None,:]*torch.einsum('nkh,fh->nkf',g,dt)).reshape(-1,len(dt))
        fi=tuple(x.double().cpu().numpy() for x in (fs.T@fs/len(fs),fs.T@ft/len(fs),ft.T@ft/len(fs)))
        gates={};fits={};pred={}
        for name,kk in [('balanced_euclidean',eu),('balanced_fisher',fi)]:
            _,v,inf=fit_groups(*kk,ai,penalty=cfg['penalty'],maxiter=cfg['maxiter'],fixed_source=gs);gates[name]=v;fits[name]=inf
        torch.set_float32_matmul_precision('high');teacher=[]
        with torch.no_grad():
            for off in range(0,len(select),rc['batch_size']):teacher.append(output(select[off:off+rc['batch_size']],source[off:off+rc['batch_size']]))
        teacher=torch.cat(teacher);assert max_replay<cfg['functional']['hidden_replay_atol'],max_replay
        train_orders=[];rng=torch.Generator(device='cpu');rng.manual_seed(f['random_seed'])
        for step in range(f['updates']):
            active=torch.randint(f['active_contexts'],(rc['batch_size']//2,),generator=rng)
            uniform=torch.randint(f['uniform_contexts'],(rc['batch_size']//2,),generator=rng)+f['active_contexts'];train_orders.append(torch.cat([active,uniform]).to('cuda:0'))
        for name,scalar in [('finite_gates',False),('finite_amplitude',True)]:
            initial=gates['balanced_euclidean'];base=torch.tensor(initial,dtype=zs.dtype,device='cuda:0')
            param=torch.ones(1,device='cuda:0',requires_grad=True) if scalar else base.clone().requires_grad_(True)
            optimizer=torch.optim.Adam([param],lr=f['learning_rate']);trace=[]
            upper=1/max(float(base.max()),1e-12) if scalar else 1.
            for step,ix in enumerate(train_orders):
                optimizer.zero_grad();gate=base*param if scalar else param;delta=(xt[ix]*gate)@dt
                lp=output(select[ix.cpu().numpy()],delta);target=teacher[ix]
                loss=(target.exp()*(target-lp)).sum(1).mean();loss.backward();optimizer.step()
                with torch.no_grad():param.clamp_(0,upper)
                if step%8==0 or step+1==f['updates']:trace.append(dict(step=step+1,source_kl=float(loss.detach())))
                if time.perf_counter()-timer>f['budget_seconds']:raise TimeoutError('Finite native fitting budget exceeded')
            gate=base*param.detach() if scalar else param.detach();gates[name]=gate.cpu().numpy()
            fits[name]=dict(updates=f['updates'],learning_rate=f['learning_rate'],parameters=1 if scalar else len(base),trace=trace,
                initialization='same balanced Euclidean gates',upper_bound=upper,scale=float(param.detach()[0]) if scalar else None)
        for name,gt in gates.items():
            group['extra_gate_'+name]=gt
            pred[name]={metric:dict(error=float(gs@ks@gs-2*gs@kst@gt+gt@kt@gt),source_energy=float(gs@ks@gs)) for metric,(ks,kst,kt) in [('euclidean',eu),('fisher',fi)]}
        group['target_gate']=gates['finite_gates'];np.savez_compressed(run/(key+'_groups.npz'),**group)
        row=dict(query=key,objective=obj,discovery_indices=select.tolist(),fits=fits,scores=pred);fitrows.append(row)
        with (run/'metrics.raw.jsonl').open('a') as out:out.write(json.dumps(dict(finite_refit=row))+'\n')
        write(run/'finite_fits.json',dict(queries=fitrows));log('FINITE_GROUP_FIT_COMPLETE',query=key,trace_gates=fits['finite_gates']['trace'][-1],trace_amplitude=fits['finite_amplitude']['trace'][-1])
    del model,module;gc.collect();torch.cuda.empty_cache()
    return dict(wall_seconds=time.perf_counter()-timer,forward_sequences=calls,backward_batches=2*f['updates']*len(results),max_hidden_replay=max_replay,peak_cuda_bytes=torch.cuda.max_memory_allocated())
