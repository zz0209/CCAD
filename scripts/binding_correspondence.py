"""Reuse source-country member queries through a frozen target dictionary."""
import json
from pathlib import Path
from arithmetic_relation_transfer import fit_member_fields


def fit_query_relation(x, y, q, ds, dt, initial, steps=256, ridge_fraction=.001,mean_coefficient_penalty=False):
    """Convex least squares on the actual fit-only source requests.

    The field for row n is (x_n * (A q_n)) D_t. The same nonnegative,
    capacity-limited relation accepts every later q in [0,1]^m.
    """
    import torch, math
    from fit_component_correspondence import project_rows
    x,y,q,ds,dt,initial=[v.float() for v in (x,y,q,ds,dt,initial)]
    gram=dt@dt.T; truth=(y*q)@ds; cross=truth@dt.T
    n=len(x); scale=(x.square().mean(0)*gram.diag()).mean().clamp_min(1e-12)
    ridge=(ridge_fraction*truth.square().sum()/(n*initial.numel()) if mean_coefficient_penalty else ridge_fraction*scale)
    def hessian(a):
        return (x*((x*(q@a.T))@gram)).T@q/n+ridge*a
    rhs=(x*cross).T@q/n
    def objective(a):
        f=x*(q@a.T)
        return .5*((f@dt-truth).square().sum()/n+ridge*a.square().sum())
    # An analytic Frobenius/trace bound starts safely; power iteration tightens
    # it and each accepted update satisfies the quadratic majorization check.
    generator=torch.Generator(device=x.device).manual_seed(9412240)
    v=torch.randn(initial.shape,device=x.device,generator=generator)
    v/=v.norm()
    for _ in range(24):
        hv=hessian(v);v=hv/hv.norm().clamp_min(1e-12)
    lip=(v*hessian(v)).sum().clamp_min(1e-12)*1.05
    a=initial.clone(); extrap=a.clone();t=1.; start=float(objective(a));history=[]
    for step in range(steps):
        gradient=hessian(extrap)-rhs
        base=objective(extrap)
        while True:
            new=project_rows(extrap-gradient/lip)
            delta=new-extrap
            upper=base+(gradient*delta).sum()+.5*lip*delta.square().sum()
            if float(objective(new)-upper)<=1e-5*max(1.,abs(float(base))):break
            lip*=2
        tn=(1+math.sqrt(1+4*t*t))/2
        extrap=new+(t-1)/tn*(new-a);a=new;t=tn
        if step%32==0 or step==steps-1:history.append([step,float(objective(a))])
    # Roundoff at the simplex boundary is removed without altering its support.
    a=a.clamp_min(0);a/=a.sum(1,keepdim=True).clamp_min(1)
    end=float(objective(a));assert end<=start+1e-4*max(1.,abs(start))
    return a,dict(objective='observed_source_queries',steps=steps,initial_loss=start,final_loss=end,
        relative_query_field_mse=float(((x*(q@a.T))@dt-truth).square().sum()/truth.square().sum()),
        maximum_row_sum=float(a.sum(1).max()),ridge_fraction=ridge_fraction,history=history)


def fit_union_relation(x,y,q,ds,dt,initial,steps=256,lr=.002,ridge_fraction=.001,capacity=False):
    """A member can fully serve several queries; each execution uses it once."""
    import torch
    from fit_component_correspondence import project_rows
    x,y,q,ds,dt,initial=[v.float() for v in (x,y,q,ds,dt,initial)]
    truth=(y*q)@ds;normalizer=truth.square().sum().clamp_min(1e-12)
    with torch.enable_grad():
        a=initial.clone().requires_grad_();opt=torch.optim.Adam([a],lr=lr);history=[]
        for step in range(steps):
            opt.zero_grad();weights=(q@a.T).clamp(0,1)
            loss=((x*weights)@dt-truth).square().sum()/normalizer+ridge_fraction*a.square().mean()
            loss.backward();opt.step()
            with torch.no_grad():
                a.clamp_(0,1)
                if capacity:a.copy_(project_rows(a))
            if step%32==0 or step==steps-1:history.append([step,float(loss.detach())])
    a=a.detach();prediction=(x*(q@a.T).clamp(0,1))@dt
    return a,dict(objective='source_queries_capped_union',steps=steps,lr=lr,ridge_fraction=ridge_fraction,
        relative_query_field_mse=float((prediction-truth).square().sum()/normalizer),history=history,
        maximum_row_sum=float(a.sum(1).max()),union_rule='min(1, A q); nonnegative entries each at most one')


def build_transfers(w,cfg,rows,hidden,zs,sae,order,conditional,target,zt,seed,target_seed):
    import torch,numpy as np
    from scipy.optimize import linear_sum_assignment
    rc=cfg['binding_transfer'];device=w.device;d=sae.decoder.weight.T.detach();dt=target.decoder.weight.T.detach()
    paired=[r['paired_row'] for r in rows];dx=zt[paired]-zt;ds=zs[paired]-zs;dh=hidden[paired]-hidden
    fitrows=[r['row_id'] for r in rows if r['split']=='fit' and not r['donor']]
    source_gate=torch.zeros(1,zs.shape[-1],device=device);source_gate[0,order[:rc['parent_members']]]=1
    roles=torch.zeros(len(fitrows)*2,dtype=torch.long,device=device)
    w.progress('MEMBER_RELATION_FIT',source_seed=seed,target_seed=target_seed,fit_sites=len(roles))
    gate,info,relation=fit_member_fields(dx[fitrows].flatten(0,1),ds[fitrows].flatten(0,1),roles,
        source_gate,d,dt,rc['target_bank'],rc)
    si=torch.tensor(relation['source_indices'],device=device);ti=torch.tensor(relation['target_indices'],device=device)
    a=torch.tensor(relation['weights'][0],device=device);q=conditional[:,:,si]
    static=torch.isin(si,order[:rc['query_members']]).float().expand_as(q)
    yy=ds[:,:,si];norm=d[si].norm(dim=1)
    fits=torch.tensor(fitrows,device=device)
    contexts=torch.tensor([rows[i]['component'] for i in fitrows],device=device)
    cutoff=sorted(set(contexts.tolist()))[int(.75*len(set(contexts.tolist())))]
    train=contexts<cutoff;val=~train
    def solve(x,y,fraction):
        x,y=x.double(),y.double();scale=x.square().mean(0).sqrt().clamp_min(1e-6);xn=x/scale
        gram=xn@xn.T;ridge=fraction*gram.diag().mean().clamp_min(1e-8)
        dual=torch.linalg.solve(gram+ridge*torch.eye(len(x),device=device,dtype=torch.float64),y)
        return ((xn.T@dual)/scale[:,None]).float()
    readouts={};readout_info={}
    for name,x in [('raw_readout',dh),('code_readout',dx[:,:,ti])]:
        xf=x[fits];yf=yy[fits];scores=[]
        for fraction in rc['readout_ridge']:
            b=solve(xf[train].flatten(0,1),yf[train].flatten(0,1),fraction)
            mse=float((((xf[val]@b-yf[val])*norm).square()).mean());scores.append((mse,fraction))
        _,chosen=min(scores);b=solve(xf.flatten(0,1),yf.flatten(0,1),chosen)
        readouts[name]=(x@b,b);readout_info[name]=dict(chosen_ridge=chosen,source_field_validation=scores)
    w.progress('MEMBER_RELATION_READY',source_seed=seed,target_seed=target_seed,field_mse=info['relative_field_mse'])
    cosine=(d[si]/d[si].norm(dim=1,keepdim=True).clamp_min(1e-12))@(dt/dt.norm(dim=1,keepdim=True).clamp_min(1e-12)).T
    ar,ac=linear_sum_assignment(-cosine.cpu().numpy());assert np.array_equal(ar,np.arange(len(si)))
    ac=torch.tensor(ac,device=device)
    outputs={};masks={}
    def trim(weights,change,decoder):
        score=weights.abs()*change.abs()*decoder.norm(dim=1)
        keep=torch.argsort(score,dim=-1,descending=True,stable=True)[...,:rc['query_members']]
        out=torch.zeros_like(weights);out.scatter_(-1,keep,torch.gather(weights,-1,keep));return out
    for label,query in [('country',q),('static',static)]:
        # Capacity is a sum of nonnegative coefficients; avoid TF32 rounding
        # at that boundary before evaluating the model in its usual precision.
        weights64=query.double()@a.double().T
        assert float(weights64.min())>=0 and float(weights64.max())<=1+1e-6
        weights=weights64.float().clamp(0,1)
        sparse=trim(weights,dx[:,:,ti],dt[ti]);assert bool((sparse>=0).all()) and float(sparse.max())<=1+1e-6
        outputs[f'member_{label}']=(dx[:,:,ti]*sparse)@dt[ti]
        masks[f'member_{label}']=sparse
        outputs[f'assignment_{label}']=(dx[:,:,ac]*query)@dt[ac]
    query_info=None
    union_info=None
    if rc.get('query_fit'):
        old_precision=torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        aq,query_info=fit_query_relation(dx[fits][:,:,ti].flatten(0,1),yy[fits].flatten(0,1),
            q[fits].flatten(0,1),d[si],dt[ti],a,rc['query_fit']['steps'],rc['ridge_fraction'])
        weights=(q.double()@aq.double().T).float().clamp(0,1)
        sparse=trim(weights,dx[:,:,ti],dt[ti])
        outputs['query_member_country']=(dx[:,:,ti]*sparse)@dt[ti]
        outputs['query_member_bank']=(dx[:,:,ti]*weights)@dt[ti]
        masks['query_relation']=aq;masks['query_member_country']=sparse
        torch.set_float32_matmul_precision(old_precision)
        w.progress('QUERY_RELATION_READY',source_seed=seed,target_seed=target_seed,**query_info)
    if rc.get('capacity_fista'):
        old_precision=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
        af,af_info=fit_query_relation(dx[fits][:,:,ti].flatten(0,1),yy[fits].flatten(0,1),q[fits].flatten(0,1),
            d[si],dt[ti],a,rc['query_fit']['steps'],rc['ridge_fraction'],mean_coefficient_penalty=True)
        af_weights=(q@af.T).clamp(0,1);af_sparse=trim(af_weights,dx[:,:,ti],dt[ti])
        outputs['matched_additive_country']=(dx[:,:,ti]*af_sparse)@dt[ti]
        masks['matched_additive_relation']=af;masks['matched_additive_country']=af_sparse
        query_info['matched_coefficient_penalty']=af_info
        torch.set_float32_matmul_precision(old_precision)
    if rc.get('union_fit'):
        old_precision=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
        union_config={key:value for key,value in rc['union_fit'].items() if key!='wrong_query_shift'}
        au,union_info=fit_union_relation(dx[fits][:,:,ti].flatten(0,1),yy[fits].flatten(0,1),
            q[fits].flatten(0,1),d[si],dt[ti],a,**union_config,ridge_fraction=rc['ridge_fraction'])
        union_weights=(q@au.T).clamp(0,1);union_sparse=trim(union_weights,dx[:,:,ti],dt[ti])
        outputs['union_member_country']=(dx[:,:,ti]*union_sparse)@dt[ti]
        outputs['union_member_bank']=(dx[:,:,ti]*union_weights)@dt[ti]
        # Source semantic queries permuted across country pairs, retaining size.
        wrong=q.roll(rc['union_fit'].get('wrong_query_shift',1),0);wrong_weights=(wrong@au.T).clamp(0,1)
        outputs['union_wrong_query']=(dx[:,:,ti]*trim(wrong_weights,dx[:,:,ti],dt[ti]))@dt[ti]
        evalrows=torch.tensor([r['row_id'] for r in rows if r['split']!='fit' and not r['donor']],device=device)
        union_info['wrong_query_changed_fraction']=float((wrong[evalrows]!=q[evalrows]).any(-1).float().mean())
        masks['union_relation']=au;masks['union_member_country']=union_sparse
        if rc.get('capacity_control'):
            acap,cap_info=fit_union_relation(dx[fits][:,:,ti].flatten(0,1),yy[fits].flatten(0,1),
                q[fits].flatten(0,1),d[si],dt[ti],a,**union_config,ridge_fraction=rc['ridge_fraction'],capacity=True)
            cap_weights=(q@acap.T).clamp(0,1);cap_sparse=trim(cap_weights,dx[:,:,ti],dt[ti])
            outputs['adam_capacity_country']=(dx[:,:,ti]*cap_sparse)@dt[ti]
            masks['adam_capacity_relation']=acap;masks['adam_capacity_country']=cap_sparse
            union_info['matched_capacity_control']=cap_info
        torch.set_float32_matmul_precision(old_precision)
        w.progress('UNION_RELATION_READY',source_seed=seed,target_seed=target_seed,**union_info)
    for name,(pred,_) in readouts.items():outputs[name+'_country']=(pred*q)@d[si]
    adaptive_info={}
    if rc.get('adaptive_execution'):
        from adaptive_native_execution import realize
        acfg=rc['adaptive_execution'];modes=acfg.get('candidate_modes',['bank','active'])
        if acfg.get('scalar_control'):
            field=outputs['code_readout_country'];base=outputs['union_member_country']
            alpha=((field*base).sum(-1)/base.square().sum(-1).clamp_min(1e-12)).clamp_min(0)
            old_coef=dx[:,:,ti]*masks['union_member_country']
            limit=torch.where(old_coef<0,zt[:,:,ti]/(-old_coef).clamp_min(1e-12),torch.inf).amin(-1)
            alpha=torch.minimum(alpha,limit)
            outputs['native_scalar_country']=alpha[...,None]*base;masks['native_scalar_alpha']=alpha
        for origin in acfg.get('fields',['code','source']):
            field=((yy if origin=='source' else readouts[origin+'_readout'][0])*q)@d[si]
            if acfg.get('reencode'):
                edited=target.encode((hidden+field).flatten(0,1)).reshape_as(zt)
                recoded=edited-zt
                score=recoded.abs()*dt.norm(dim=1)
                ix=torch.argsort(score,dim=-1,descending=True,stable=True)[...,:rc['query_members']]
                coef=recoded.gather(-1,ix)
                sparse_delta=(coef[...,None]*dt[ix]).sum(-2)
                for suffix,pred in [('sparse',sparse_delta),('full',recoded@dt)]:
                    name=f'reencoded_{origin}_{suffix}';outputs[name]=pred
                    ev=torch.tensor([r['row_id'] for r in rows if r['split']!='fit' and not r['donor']],device=device)
                    truth=(yy*q)@d[si]
                    adaptive_info[name]=dict(operation='nonnegative target code difference after encoding predicted edited state',
                        source_relative_mse_evaluation=float((pred[ev]-truth[ev]).square().sum()/truth[ev].square().sum()),
                        max_changes=int((coef!=0).sum(-1).max()) if suffix=='sparse' else int((recoded!=0).sum(-1).max()),
                        field_information='exact source codes at execution' if origin=='source' else 'frozen source-code readout from target changes')
                masks[f'reencoded_{origin}_sparse_indices']=ix
                masks[f'reencoded_{origin}_sparse_coefficients']=coef
                masks[f'reencoded_{origin}_full_coefficients']=recoded
            for mode in modes:
                bank_mode=mode in ['bank','shared_support']
                xx,dd=(dx[:,:,ti],dt[ti]) if bank_mode else (dx,dt)
                general=acfg.get('general_codes',False)
                name=f"{'synthesized' if general else 'adaptive'}_{origin}_{mode}"
                pred,coef,ix,ainfo=realize(field,xx,dd,members=rc['query_members'],
                    current_codes=(zt[:,:,ti] if bank_mode else zt) if general else None,
                    support_mask=(dx[:,:,ti]*masks['union_member_country']!=0) if mode=='shared_support' else None,
                    **{k:v for k,v in acfg.items() if k not in ['candidate_modes','fields','reencode','general_codes','scalar_control']})
                outputs[name]=pred
                absolute=ti[ix] if bank_mode else ix
                masks[name+'_coefficients']=coef;masks[name+'_indices']=absolute
                truth=(yy*q)@d[si]
                ainfo['source_relative_mse_fit']=float((pred[fits]-truth[fits]).square().sum()/truth[fits].square().sum())
                ev=torch.tensor([r['row_id'] for r in rows if r['split']!='fit' and not r['donor']],device=device)
                ainfo['source_relative_mse_evaluation']=float((pred[ev]-truth[ev]).square().sum()/truth[ev].square().sum())
                ainfo['field_information']='exact source codes at execution' if origin=='source' else 'frozen source-code readout from target changes'
                adaptive_info[name]=ainfo
                w.progress('ADAPTIVE_NATIVE_READY',source_seed=seed,target_seed=target_seed,method=name,**ainfo)
    # A target-only competitor can use the same country metadata and fit contexts.
    countries=sorted({v for r in rows for v in r['countries']});means=[]
    for country in countries:
        indices=[(r['row_id'],j) for r in rows if r['split']=='fit' for j in range(2) if r['countries'][j]==country]
        means.append(torch.stack([zt[i,j] for i,j in indices]).mean(0))
    means=torch.stack(means);direct=torch.zeros_like(zt)
    for r in rows:
        other=rows[r['paired_row']]
        for j in range(2):
            u,v=[countries.index(x['countries'][j]) for x in [r,other]]
            score=(means[v]-means[u]).abs()*dt.norm(dim=1)
            keep=torch.argsort(score,descending=True,stable=True)[:rc['query_members']]
            direct[r['row_id'],j,keep]=1
    outputs['target_country']=(dx*direct)@dt
    outputs['member_parent']=(dx*gate[0])@dt
    np.savez_compressed(w.run/f'binding_relation_s{seed}_t{target_seed}.npz',
        **relation,gate=gate.cpu().numpy(),assignment=ac.cpu().numpy(),
        raw_readout=readouts['raw_readout'][1].cpu().numpy(),code_readout=readouts['code_readout'][1].cpu().numpy(),
        source_query=q.cpu().numpy(),**{k:v.cpu().numpy() for k,v in masks.items()})
    with (w.run/'binding_relations.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(source_seed=seed,target_seed=target_seed,field_fit=info,readouts=readout_info,
            fit_contexts=len(set(contexts.tolist())),query_fit=query_info,union_fit=union_info,adaptive_execution=adaptive_info,target_output_labels=0,target_output_gradients=0,
            budget='256 stored members; 64 per edited token for subset requests; member_parent uses256'))+'\n')
    return {name:delta.detach() for name,delta in outputs.items()}
