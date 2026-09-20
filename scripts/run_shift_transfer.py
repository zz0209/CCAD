"""Fit source-member counterparts on natural states, then reuse the human decision.

The correspondence fits individual decoder contributions and supports new unions.
The published SHIFT source procedure and annotations are MIT, copyright2024
saprmarks; original license/commit accompany the configured source manifest.
"""
from pathlib import Path
import argparse,json,sys,time,platform,traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork,write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module


def simplex_rows(x):
    x=np.maximum(x,0);need=x.sum(1)>1
    if np.any(need):
        a=x[need];s=np.sort(a,axis=1)[:,::-1];v=(np.cumsum(s,axis=1)-1)/np.arange(1,s.shape[1]+1);k=(s>v).sum(1)-1;x[need]=np.maximum(a-v[np.arange(len(a)),k,None],0)
    return x


def fit_members(k,b,iterations):
    lipschitz=max(np.linalg.eigvalsh(k)[-1],1e-15);a=np.zeros_like(b)
    for _ in range(iterations):a=simplex_rows(a+(b-k@a)/lipschitz)
    return a


def fit_parts(k,b,initial,iterations,total_only=False):
    """Fit source-defined part fields, or only their sum from the same initial lift."""
    a=initial.copy();lipschitz=max(np.linalg.eigvalsh(k)[-1],1e-15)
    if total_only:
        lipschitz*=b.shape[1]
        for _ in range(iterations):
            gradient=(k@a.sum(1)-b.sum(1))[:,None]
            a=simplex_rows(a-gradient/lipschitz)
    else:
        for _ in range(iterations):a=simplex_rows(a+(b-k@a)/lipschitz)
    return a


def input_member_delta(x, target, source, q, mode, allowance, attention):
    """Input-dependent columns, allocated once before the human-part query.

    Evaluate bounded token chunks; inactive source columns are exactly zero.
    Counts include real tokens only. Finite and tangent columns use the same
    source coefficients and query-independent target support/capacity rule.
    """
    import torch
    shape=x.shape
    flat=x.reshape(-1,shape[-1]); keep=attention.reshape(-1).bool()
    result=torch.zeros_like(flat)
    count=dict(states=0,changed=0,increased=0,scaled=0,minimum_final_code=0.)
    directions=-(target.encoder.weight@source['decoder'].T)
    norms=target.decoder.weight.norm(dim=0)
    indices=keep.nonzero().flatten()
    if mode.startswith(('input_tangent','input_fixed')):
        basis=source.get('transport_basis',directions) if mode.startswith('input_tangent') else source['fixed_response_basis']
        bp,bn=basis.clamp_min(0),(-basis).clamp_min(0)
        for ix in indices.split(512):
            h=flat[ix];z=target.encode(h)
            zs=torch.relu((h-source['center'])@source['encoder'].T+source['encoder_bias'])
            active=(z>0) if mode.startswith('input_tangent') else torch.ones_like(z)
            selected=active
            if mode.endswith('_budget'):
                score=(zs@basis.abs().T)*active*norms
                ids=score.topk(allowance,dim=-1).indices
                selected=active*torch.zeros_like(score).scatter(-1,ids,1.)
            total_negative=(zs@bn.T)*selected
            scale=(z/total_negative.clamp_min(1e-20)).clamp_max(1)
            delta=((zs*q)@bp.T-(zs*q)@bn.T*scale)*selected
            result[ix]=delta@target.decoder.weight.T
            count['states']+=len(ix);count['changed']+=int((delta!=0).sum())
            count['increased']+=int((delta>0).sum())
            count['scaled']+=int(((scale<1)&(total_negative>0)).sum())
            count['minimum_final_code']=min(count['minimum_final_code'],float((z+delta).min()))
        return result.reshape(shape),count
    for ix in indices.split(512 if mode=='raw_reconstruction' else 128):
        h=flat[ix];z=target.encode(h)
        zs=torch.relu((h-source['center'])@source['encoder'].T+source['encoder_bias'])
        if mode=='raw_reconstruction':
            rec=z@target.decoder.weight.T+target.b_dec
            zr=torch.relu((rec-source['center'])@source['encoder'].T+source['encoder_bias'])
            result[ix]=-(zr*q)@source['decoder']
            continue
        if mode.startswith('input_tangent'):
            columns=directions*zs.unsqueeze(-2)*(z>0).unsqueeze(-1)
        elif mode.startswith('input_finite'):
            cf=h.unsqueeze(-2)-zs.unsqueeze(-1)*source['decoder']
            columns=(target.encode(cf)-z.unsqueeze(-2)).transpose(-1,-2)
            columns=columns*(zs!=0).unsqueeze(-2)
        elif mode.startswith('input_fixed'):
            columns=source['fixed_response_basis']*zs.unsqueeze(-2)
        else:
            raise ValueError(mode)
        if mode.endswith('_budget'):
            score=columns.abs().sum(-1)*norms
            ids=score.topk(allowance,dim=-1).indices
            selected=torch.zeros_like(score).scatter(-1,ids,1.)
            columns=columns*selected.unsqueeze(-1)
        positive,negative=columns.clamp_min(0),(-columns).clamp_min(0)
        scale=(z/negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
        delta=(positive-negative*scale.unsqueeze(-1))@q
        result[ix]=delta@target.decoder.weight.T
        count['states']+=len(ix);count['changed']+=int((delta!=0).sum())
        count['increased']+=int((delta>0).sum())
        count['scaled']+=int(((scale<1)&(negative.sum(-1)>0)).sum())
        count['minimum_final_code']=min(count['minimum_final_code'],float((z+delta).min()))
    return result.reshape(shape),count


def context_candidates(x, target, source, cosine, geometry, cfg, work, site):
    """Add activation-context partners at the original unique-candidate budget."""
    import torch
    from ccad.semantic_context_matching import top_distributions, retrieve, euclidean_cost
    settings=cfg['context_candidates'];started=time.perf_counter()
    # Capture concatenates clean states and full-source-deletion states. Retrieval
    # uses the clean half; the existing field fit continues to use both halves.
    reference=x[:len(x)//2].detach().cpu().numpy()
    source_values=[];target_values=[];target_indices=[]
    with torch.no_grad():
        for off in range(0,len(reference),512):
            xx=x[off:min(off+512,len(reference))]
            source_values.append(torch.relu((xx-source['center'])@source['encoder'].T+source['encoder_bias']).cpu().numpy())
            values,indices=target.encode(xx).topk(int(target.k),dim=1)
            target_values.append(values.cpu().numpy());target_indices.append(indices.cpu().numpy())
    zs=np.concatenate(source_values);ns=zs.shape[1]
    source_indices=np.broadcast_to(np.arange(ns),(len(zs),ns))
    sd,sc,sn=top_distributions(source_indices,zs,ns,settings['top_k'],reference)
    td,tc,tn=top_distributions(np.concatenate(target_indices),np.concatenate(target_values),
                             target.encoder.weight.shape[0],settings['top_k'],reference)
    rng=np.random.default_rng(settings['distance_scale_seed'])
    sample=rng.choice(len(reference),min(256,len(reference)),replace=False)
    distances=euclidean_cost(reference[sample].astype(float),reference[sample].astype(float))
    scale=float(np.median(distances[np.triu_indices(len(sample),1)]))
    epsilon=scale*settings['sinkhorn_scale_fraction']
    completed=[]
    def progress(row):
        completed.append(row)
        work.progress('CONTEXT_MEMBER',site=site,completed=len(completed),source_members=ns,
                      status=row['status'],retrieval_seconds=time.perf_counter()-started)
        if time.perf_counter()-work.wall_start>cfg['budget_seconds']:
            raise TimeoutError('Context retrieval exceeded bounded run budget')
    rows=retrieve(range(ns),sd,td,sc,tc,sn,tn,reference,
                  candidate_count=settings['centroid_candidates'],minimum_count=settings['minimum_activations'],
                  regularization=epsilon,tolerance=settings['sinkhorn_tolerance'],
                  max_iterations=settings['sinkhorn_max_iterations'],dual_fallback=True,progress=progress)
    # Preserve a geometric core and add context ranks round-robin over source
    # members. Unique pool size equals the original geometry pool exactly.
    old=geometry.cpu().tolist();budget=len(old)
    core=torch.topk(cosine,settings['geometry_core_per_source'],dim=0).indices.flatten().unique().cpu().tolist()
    chosen=set(core);rankings=[[v['target_member'] for v in row['candidates']] for row in rows]
    for rank in range(settings['centroid_candidates']):
        for values in rankings:
            if rank<len(values) and len(chosen)<budget:chosen.add(values[rank])
    for value in old:
        if len(chosen)<budget:chosen.add(value)
    assert len(chosen)==budget
    detail=dict(site=site,reference_states=len(reference),source_activities=sn.tolist(),
                eligible_targets=int((tn>=settings['minimum_activations']).sum()),
                geometry_candidates=old,candidates=sorted(chosen),geometry_core=core,
                outside_geometry=sorted(chosen-set(old)),reference_distance_scale=scale,
                regularization=epsilon,rows=rows,retrieval_seconds=time.perf_counter()-started,
                scope='Natural clean states; same unique pool size and final member allowance; no evaluation rows or target task gradients.')
    return torch.tensor(sorted(chosen),device=x.device,dtype=torch.long),detail


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);a=p.parse_args();c=json.loads(a.config.read_text());w=MultisiteWork(c,a.config,['scripts/run_shift_transfer.py','scripts/run_shift_explanation.py','scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/semantic_context_matching.py','src/ccad/nip_baselines.py']);hooks=[];error=None
    try:
        import torch,transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(c['device']);torch.cuda.reset_peak_memory_stats();w.torch=torch;w.device=torch.device(c['device'])
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']]);from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.environment=dict(python=sys.executable,torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,matmul_precision='highest',cpu_threads=2)
        source=json.loads(w.checked(c['source_manifest'],'Published manual feature decisions','MIT').read_text());groups,annotations=source_groups(w.checked(c['notebook'],'Published source annotations','MIT'),source['members']);sites=list(source['members']);sb=np.load(w.checked(c['source_parameters'],'Published feature parameters','MIT'))
        part_names=list(groups)
        requested_parts={q:part_names if q=='full' else q.split('+') for q in c['queries']}
        assert all(set(names)<=set(part_names) and len(names)==len(set(names)) for names in requested_parts.values())
        sp={site:{k:torch.tensor(sb[site+'__'+k],device=w.device) for k in ['encoder','encoder_bias','decoder','center']} for site in sites};targets={}
        for site in sites:
            state=torch.load(w.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt','Independent natural-data target dictionary','MIT trainer'),map_location=w.device,weights_only=True)
            sae=AutoEncoderTopK(512,state['encoder.weight'].shape[0],int(state['k'])).to(w.device);sae.load_state_dict(state);sae.requires_grad_(False);targets[site]=sae
        for f in ['config.json','tokenizer.json','model.safetensors']:w.checked(Path(c['model_local_dir'])/f,'Pinned Pythia70M','Apache-2.0')
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device);model.requires_grad_(False);model.config.use_cache=False
        mode='capture';alpha=0.;query='full';observed={};responses={};relations={};mask=None;pooled=None;write_counts={};adapted_targets={}
        functional_weight=c.get('functional_weight',0.)
        source_run=Path(c['frozen_source_run']);probe=np.load(w.checked(source_run/'probe.npz','Fixed original source classifier'));pw=torch.tensor(probe['weight'],device=w.device);pb=torch.tensor(probe['bias'],device=w.device)
        def hook(site):
            def f(module,inputs,out):
                nonlocal pooled
                x=out[0] if isinstance(out,tuple) else out
                if mode=='capture' and functional_weight and site=='embed':x=x.detach().requires_grad_(True)
                s=sp[site]
                operation=('source' if site==mode.split('__')[1] else 'native') if mode.startswith('repair__') else mode
                if operation in ('capture','source'):
                    z=torch.relu((x-s['center'])@s['encoder'].T+s['encoder_bias'])
                selected=requested_parts[query]
                q=torch.tensor([any(i in groups[name].get(site,[]) for name in selected) for i in source['members'][site]],device=w.device,dtype=x.dtype)
                if mode=='capture':observed[site]=x.detach();x=x-alpha*(z@s['decoder']);responses[site]=x
                elif operation=='source':x=x-(z*q)@s['decoder']
                elif mode.startswith('input_') or mode=='raw_reconstruction':
                    delta,counts=input_member_delta(x,targets[site],s,q,mode,c['members_per_source']*len(q),mask)
                    x=x+delta
                    key=mode+'/'+query+'/'+site
                    if key not in write_counts:write_counts[key]=counts
                    else:
                        for name,value in counts.items():
                            if name=='minimum_final_code':write_counts[key][name]=min(write_counts[key][name],value)
                            else:write_counts[key][name]+=value
                elif mode!='none':
                    t=adapted_targets.get(mode,{}).get(site,targets[site]);r=relations[site];zt=t.encode(x)
                    rq=torch.tensor([name in selected for name in part_names],device=w.device,dtype=x.dtype) if mode in ('native_groups','native_total') else q
                    coeff=r[operation]@rq
                    if mode=='raw':x=x-(zt[:, :, r['candidates']]@r['raw']*q)@s['decoder']
                    else:x=x-(zt*coeff)@t.decoder.weight.T
                if site=='resid_4' and mode!='capture':pooled=(x*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
                return (x,*out[1:]) if isinstance(out,tuple) else x
            return f
        for site in sites:hooks.append(site_module(model,site).register_forward_hook(hook(site)))
        # The source deletion path supplies input states only; no task labels/logits.
        if c.get('relation_run'):
            saved=np.load(w.checked(Path(c['relation_run'])/'relation.npz','Previously frozen member relations'))
            for site in sites:
                relations[site]={name:torch.tensor(saved[site+'__'+name],device=w.device,
                    dtype=torch.long if name=='candidates' else torch.float32)
                    for name in ['native','geometry','geometry_gain','raw','candidates']}
            fit_summary=json.loads(w.checked(Path(c['relation_run'])/'RELATION_FIT.json').read_text())
            for name,spec in c.get('adapted_programs',{}).items():
                folder=Path(spec['directory']);adapted_targets[name]={}
                fitted=np.load(w.checked(folder/'relation.npz','Frozen adapted program relation'))
                for site in sites:
                    relations[site][name]=torch.tensor(fitted[site+'__native'],device=w.device,dtype=torch.float32)
                    if site in spec['sites']:
                        state=torch.load(w.checked(folder/f'{site}_seed{c["target_seed"]}.pt','Frozen adapted dictionary'),map_location=w.device,weights_only=True)
                        sae=AutoEncoderTopK(512,state['encoder.weight'].shape[0],int(state['k'])).to(w.device)
                        sae.load_state_dict(state);sae.requires_grad_(False);adapted_targets[name][site]=sae
            if c.get('fixed_basis_run'):
                basis=np.load(w.checked(Path(c['fixed_basis_run'])/'fixed_response_basis.npz'))
                for site in sites:sp[site]['fixed_response_basis']=torch.tensor(basis[site],device=w.device)
            elif any(name.startswith('input_fixed') for name in c['methods']):
                mode='capture';alpha=0.;functional_weight=0.
                tokens=np.memmap(w.checked(c['paired_tokens']),dtype='<u2',mode='r').reshape(-1,128)[:c['basis_fit_sequences']]
                sums={site:torch.zeros((targets[site].encoder.weight.shape[0],len(source['members'][site])),device=w.device) for site in sites}
                squares={site:torch.zeros(len(source['members'][site]),device=w.device) for site in sites}
                with torch.no_grad():
                    for off in range(0,len(tokens),c['fit_batch_sequences']):
                        ids=torch.tensor(np.array(tokens[off:off+c['fit_batch_sequences']],dtype='int64'),device=w.device);mask=torch.ones_like(ids)
                        model.gpt_neox(ids,attention_mask=mask,use_cache=False)
                        for site in sites:
                            t=targets[site];ss=sp[site]
                            for h in observed[site].flatten(0,1).split(128):
                                zt=t.encode(h);zs=torch.relu((h-ss['center'])@ss['encoder'].T+ss['encoder_bias'])
                                columns=(t.encode(h.unsqueeze(-2)-zs.unsqueeze(-1)*ss['decoder'])-zt.unsqueeze(-2)).transpose(-1,-2)
                                columns=columns*(zs!=0).unsqueeze(-2)
                                sums[site]+=(columns*zs.unsqueeze(-2)).sum(0);squares[site]+=zs.square().sum(0)
                        w.sequence_forwards+=len(ids);w.token_forwards+=ids.numel()
                basis={}
                for site in sites:
                    sp[site]['fixed_response_basis']=sums[site]/squares[site].clamp_min(1e-20)
                    basis[site]=sp[site]['fixed_response_basis'].cpu().numpy()
                np.savez_compressed(w.run/'fixed_response_basis.npz',**basis)
                w.progress('FIXED_BASIS_FIT',natural_states=int(tokens.size),source_information='live exact coefficients for both fixed and input-dependent columns')
        else:
            pth=w.checked(c['paired_tokens'],'Existing discovery natural split','ODC-By-1.0');nat=np.memmap(pth,dtype='<u2',mode='r').reshape(-1,128)[:c['fit_sequences']]
            cache={site:[] for site in sites};gradient_cache={site:[] for site in sites}
            for alpha in [0.,1.]:
                for off in range(0,len(nat),c['fit_batch_sequences']):
                    ids=torch.tensor(np.array(nat[off:off+c['fit_batch_sequences']],dtype='int64'),device=w.device)
                    with torch.set_grad_enabled(bool(functional_weight)):
                        model.gpt_neox(ids,use_cache=False)
                        if functional_weight:
                            score=(responses['resid_4'].mean(1)@pw.T+pb).sum()*ids.shape[1]
                            gradients=torch.autograd.grad(score,[responses[site] for site in sites])
                            for site,g in zip(sites,gradients):gradient_cache[site].append(g.detach().flatten(0,1).cpu().numpy().copy())
                    for site in sites:cache[site].append(observed[site].flatten(0,1).cpu().numpy().copy())
                    w.sequence_forwards+=len(ids);w.token_forwards+=ids.numel()
            fit_summary={};export={};retrieval={}
            for site in sites:
                x=torch.tensor(np.concatenate(cache.pop(site)),device=w.device);s=sp[site];t=targets[site];ns=len(source['members'][site]);dt=t.decoder.weight.T.detach();ds=s['decoder'];cos=(dt/dt.norm(dim=1)[:,None])@(ds/ds.norm(dim=1)[:,None]).T
                baseidx=torch.topk(cos,c['neighbors_per_source'],dim=0).indices.flatten().unique();nt=len(dt)
                if c.get('context_candidates'):
                    baseidx,retrieval[site]=context_candidates(x,t,s,cos,baseidx,c,w,site)
                    write(w.run/'CANDIDATE_RETRIEVAL.json',retrieval)
                zz=[];ss=[]
                with torch.no_grad():
                    for off in range(0,len(x),512):
                        xx=x[off:off+512];zz.append(t.encode(xx)[:,baseidx].cpu().numpy());ss.append(torch.relu((xx-s['center'])@s['encoder'].T+s['encoder_bias']).cpu().numpy())
                zt=np.concatenate(zz).astype('float64');zs=np.concatenate(ss).astype('float64');di=dt[baseidx].cpu().numpy().astype('float64');dsrc=ds.cpu().numpy().astype('float64');n=len(zt)
                cz=zt.T@zt/n;cross=zt.T@zs/n;ke=cz*(di@di.T);be=cross*(di@dsrc.T);k,b=ke,be;weighted_source=None
                if functional_weight:
                    gs=np.concatenate(gradient_cache.pop(site)).astype('float64');at=zt*(gs@di.T);ass=zs*(gs@dsrc.T)
                    if c.get('context_response'):
                        at=at.reshape(-1,128,at.shape[1]).sum(1);ass=ass.reshape(-1,128,ass.shape[1]).sum(1)
                    kf=at.T@at/len(at);bf=at.T@ass/len(at)
                    se=max(np.trace(ke)/len(ke),1e-15);sf=max(np.trace(kf)/len(kf),1e-15)
                    k=(1-functional_weight)*ke/se+functional_weight*kf/sf;b=(1-functional_weight)*be/se+functional_weight*bf/sf
                    weighted_source=gs@dsrc.T
                candidate_fit=fit_members(k,b,c['fit_iterations'])
                allowance=min(len(baseidx),c['members_per_source']*ns);score=candidate_fit.sum(1)*np.sqrt(np.maximum(np.diag(k),0));chosen=np.argsort(-score,kind='stable')[:allowance];refit=fit_members(k[np.ix_(chosen,chosen)],b[chosen],c['fit_iterations'])
                native=np.zeros((nt,ns));native[baseidx.cpu().numpy()[chosen]]=refit
                part_relations={};part_fits={}
                if c.get('compare_query_granularity',False):
                    partition=np.array([[int(i in groups[name].get(site,[])) for name in part_names] for i in source['members'][site]],dtype='float64')
                    assert np.all(partition.sum(1)==1)
                    grouped_b=b@partition
                    for name,total_only,use_groups in [('native_groups',False,True),('native_total',True,True),('native_refit',False,False)]:
                        target_b=grouped_b if use_groups else b
                        initial=candidate_fit@partition if use_groups else candidate_fit
                        group_fit=fit_parts(k,target_b,initial,c['fit_iterations'],total_only)
                        group_score=group_fit.sum(1)*np.sqrt(np.maximum(np.diag(k),0))
                        group_chosen=np.argsort(-group_score,kind='stable')[:allowance]
                        group_refit=fit_parts(k[np.ix_(group_chosen,group_chosen)],target_b[group_chosen],group_fit[group_chosen],c['fit_iterations'],total_only)
                        value=np.zeros((nt,target_b.shape[1]));value[baseidx.cpu().numpy()[group_chosen]]=group_refit
                        part_relations[name]=value
                        part_fits[name]=dict(selected=int(np.count_nonzero(value.sum(1)>0)),allowance=allowance,row_capacity=float(value.sum(1).max()),total_only=total_only,columns=part_names if use_groups else source['members'][site],initialization='same source-member candidate fit; collapsed only for grouped fits',common_initial_iterations=c['fit_iterations'],refine_iterations=c['fit_iterations'],support_refit_iterations=c['fit_iterations'])
                # Same member allowance, dictionary-only matching, with source-fit gains.
                geom=np.zeros((nt,ns));used=set()
                for j in range(ns):
                    for idx in baseidx[torch.argsort(cos[baseidx,j],descending=True)].cpu().tolist():
                        if idx not in used:geom[idx,j]=1.;used.add(idx)
                        if int(geom[:,j].sum())==c['members_per_source']:break
                geom_gain=geom.copy()
                # Geometric selections are in the retained candidate pool; no labels.
                lookup={int(v):i for i,v in enumerate(baseidx.cpu().tolist())}
                for j in range(ns):
                    selected=np.flatnonzero(geom[:,j]);assert all(int(v) in lookup for v in selected)
                    ix=[lookup[int(v)] for v in selected];den=k[np.ix_(ix,ix)].sum();gain=np.clip(b[ix,j].sum()/max(den,1e-15),0,1);geom_gain[selected,j]*=gain
                # A target-code readout retains every source member column and uses the whole candidate pool.
                ridge=c['ridge']*max(np.trace(cz)/len(cz),1e-12);raw=np.linalg.solve(cz+ridge*np.eye(len(cz)),cross)
                if functional_weight:
                    for j in range(ns):
                        weights=weighted_source[:,j]**2;weights=weights/max(weights.mean(),1e-15);weights=(1-functional_weight)+functional_weight*weights
                        weighted_cov=zt.T@(zt*weights[:,None])/n;weighted_cross=zt.T@(zs[:,j]*weights)/n
                        if c.get('context_response'):
                            ax=(zt*weighted_source[:,j,None]).reshape(-1,128,zt.shape[1]).sum(1);cx=(zs[:,j]*weighted_source[:,j]).reshape(-1,128).sum(1)
                            functional_cov=ax.T@ax/len(ax);functional_cross=ax.T@cx/len(ax);sc=max(np.trace(cz)/len(cz),1e-15);sf=max(np.trace(functional_cov)/len(cz),1e-15)
                            weighted_cov=(1-functional_weight)*cz/sc+functional_weight*functional_cov/sf;weighted_cross=(1-functional_weight)*cross[:,j]/sc+functional_weight*functional_cross/sf
                        raw[:,j]=np.linalg.solve(weighted_cov+c['ridge']*max(np.trace(weighted_cov)/len(cz),1e-12)*np.eye(len(cz)),weighted_cross)
                relations[site]={'native':torch.tensor(native,device=w.device,dtype=torch.float32),'geometry':torch.tensor(geom,device=w.device,dtype=torch.float32),'geometry_gain':torch.tensor(geom_gain,device=w.device,dtype=torch.float32),'raw':torch.tensor(raw,device=w.device,dtype=torch.float32),'candidates':baseidx}
                for name,value in part_relations.items():
                    relations[site][name]=torch.tensor(value,device=w.device,dtype=torch.float32);export[site+'__'+name]=value
                target_mean=zt.mean(0);source_mean=zs.mean(0)
                for name,value in [('native',native),('geometry',geom),('geometry_gain',geom_gain),('raw',raw),('candidates',baseidx.cpu().numpy()),('target_mean',target_mean),('source_mean',source_mean)]:export[site+'__'+name]=value
                full=np.zeros_like(candidate_fit);full[chosen]=refit;source_energy=np.sum(zs**2,axis=0)/n*np.sum(dsrc**2,axis=1);loss=source_energy-2*(full*be).sum(0)+(full*(ke@full)).sum(0)
                fit_summary[site]=dict(source_members=ns,candidates=len(baseidx),selected=allowance,n_states=n,functional_weight=functional_weight,context_response=c.get('context_response',False),field_relative_error=(loss/np.maximum(source_energy,1e-15)).tolist(),row_capacity=float(native.sum(1).max()),natural_source_active=(zs>0).sum(0).tolist(),solver_iterations=c['fit_iterations'],target_mean_norm=float(np.linalg.norm(target_mean)),source_mean_norm=float(np.linalg.norm(source_mean)))
                if site in retrieval:
                    selected_ids=set(np.flatnonzero(native.sum(1)>0).tolist())
                    fit_summary[site]['selected_outside_geometry']=sorted(selected_ids-set(retrieval[site]['geometry_candidates']))
                if part_fits:fit_summary[site]['part_fits']=part_fits
                if functional_weight:
                    energy=(ass**2).mean(0);floss=energy-2*(full*bf).sum(0)+(full*(kf@full)).sum(0);fit_summary[site]['response_relative_error']=(floss/np.maximum(energy,1e-15)).tolist()
                w.progress('RELATION_FIT',site=site,result=fit_summary[site]);del x,zt,zs
            np.savez_compressed(w.run/'relation.npz',**export);write(w.run/'RELATION_FIT.json',fit_summary)
        evaluation_split=c.get('evaluation_split','dev')
        panel_path=Path(c['evaluation_panel']) if c.get('evaluation_panel') else source_run/'panel.json'
        panel=json.loads(w.checked(panel_path,'Fixed correspondence evaluation panel').read_text());dev=[r for r in panel['rows'] if r['split']==evaluation_split]
        assert dev and len({r['document_sha256'] for r in dev})==len(dev)
        if c.get('development_per_group'):
            assert evaluation_split=='dev'
            chosen=[]
            for y in [0,1]:
                for gender in [0,1]:
                    rows=sorted([r for r in dev if r['label']==y and r['gender']==gender],key=lambda r:r['document_sha256'])
                    chosen.extend(rows[:c['development_per_group']])
            dev=chosen
        write(w.run/'evaluation_membership.json',dict(split=evaluation_split,rows=dev))
        results={};pad=0
        for mode in c['methods']:
            for query in (['full'] if mode=='none' else c['queries']):
                logits=np.empty(len(dev),np.float32);order=sorted(range(len(dev)),key=lambda i:len(dev[i]['tokens']))
                start=0
                while start<len(dev):
                    ix=order[start:start+c['eval_batch_size']]
                    if c.get('eval_token_budget'):
                        while len(ix)>1 and len(ix)*max(len(dev[i]['tokens']) for i in ix)>c['eval_token_budget']:ix=ix[:-1]
                    length=max(len(dev[i]['tokens']) for i in ix);ids=torch.full((len(ix),length),pad,device=w.device,dtype=torch.long);mask=torch.zeros_like(ids)
                    for j,i in enumerate(ix):v=dev[i]['tokens'];ids[j,:len(v)]=torch.tensor(v,device=w.device);mask[j,:len(v)]=1
                    with torch.no_grad():model.gpt_neox(ids,attention_mask=mask,use_cache=False);values=(pooled@pw.T+pb).squeeze(-1)
                    logits[ix]=values.cpu().numpy();w.sequence_forwards+=len(ix);w.token_forwards+=ids.numel();start+=len(ix)
                for r,value in zip(dev,logits):w.record(kind='classification',task='profession',row_id=r['row_id'],component=r['document_sha256'],method=mode,operation=query,seed=c['target_seed'],target_seed=c['target_seed'],split=evaluation_split,label=r['label'],gender=r['gender'],prediction=int(value>0),logit=float(value))
                acc={f'{y}/{g}':float(np.mean([(v>0)==r['label'] for r,v in zip(dev,logits) if r['label']==y and r['gender']==g])) for y in [0,1] for g in [0,1]};res=dict(profession=float(np.mean([(v>0)==r['label'] for r,v in zip(dev,logits)])),gender=float(np.mean([(v>0)==r['gender'] for r,v in zip(dev,logits)])),worst_group=min(acc.values()),groups=acc)
                results[f'{mode}/{query}']=res;w.progress('RESULT',method=mode,query=query,result=res)
                if c.get('eval_token_budget'):torch.cuda.empty_cache()
                if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Bounded correspondence consumer')
        write(w.run/'TRANSFER_RESULTS.json',dict(results=results,scope=c['scope'],fit=fit_summary,write_counts=write_counts))
        expected_cells=sum(1 if name=='none' else len(c['queries']) for name in c['methods'])
        w.checks.update(all_query_methods=len(results)==expected_cells,native_capacity=all(v['row_capacity']<=1.000001 for v in fit_summary.values()))
        w.checks['part_capacity']=all(p['row_capacity']<=1.000001 and p['selected']<=p['allowance'] for v in fit_summary.values() for p in v.get('part_fits',{}).values())
        w.checks['adapted_capacity']=all(bool((relations[s][m]>=0).all()) and float(relations[s][m].sum(1).max())<=1.00001 for m in adapted_targets for s in sites)
    except Exception:error=traceback.format_exc()
    finally:
        for h in hooks:h.remove()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
