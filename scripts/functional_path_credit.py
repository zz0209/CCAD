"""Fit source-component response relations along finite deletion paths.

Codes are evaluated at the clean state and remain fixed along each path.
The path moves the shared model hook; it does not re-encode interpolated states.
"""
import math


def response_fit(credits, effects, steps=800, ridge=.01):
    import torch
    from fit_component_correspondence import project_rows
    # Different source requests have different path-conditioned response Grams.
    X=credits.double();Y=effects.double()
    scale=Y.square().mean(1).sqrt().clamp_min(1e-8)
    X=X/scale[:,None,None];Y=Y/scale[:,None]
    K=X.transpose(1,2)@X/X.shape[1]
    rhs=torch.einsum('knp,kn->pk',X,Y)/X.shape[1]
    diagonal=K.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-12)
    A=K+ridge*diagonal[:,None,None]*torch.eye(K.shape[-1],device=K.device,dtype=K.dtype)
    lipschitz=torch.linalg.eigvalsh(A)[:,-1].max().clamp_min(1e-12)
    def grad(v):return torch.einsum('kij,jk->ik',A,v)-rhs
    x=torch.zeros_like(rhs);y=x.clone();t=1.
    for _ in range(steps):
        new=project_rows(y-grad(y)/lipschitz)
        tn=(1+math.sqrt(1+4*t*t))/2
        y=new+(t-1)/tn*(new-x);x=new;t=tn
    pred=torch.einsum('knp,pk->kn',credits.double(),x)
    return x.float(),dict(steps=steps,ridge=ridge,maximum_row_sum=float(x.sum(1).max()),projected_gradient_max=float((x-project_rows(x-grad(x)/lipschitz)).abs().max()),source_response_mae=(pred-effects).abs().mean(1).cpu().tolist(),normalized_response_mse=((pred-effects).square().mean(1)/scale.square()).cpu().tolist())


def propose(cfg,w,D,saes,forward,log,budget,obj,s,t,h,grad,source_panel,tp,sp,sg):
    import numpy as np,torch
    ix=[i for i,r in enumerate(source_panel) if cfg['path_source_range'][0]<=r['row_id']<cfg['path_source_range'][1]]
    rows=[source_panel[i] for i in ix];hidden=h[ix];clean_gradient=grad[ix]
    with torch.no_grad():zs=saes[obj,s].encode(hidden);zt=saes[obj,t].encode(hidden)[:,tp]
    ds=D[obj,s][sp];dt=D[obj,t][tp]
    source=torch.stack([(zs[:,sp]*sg[:,k])@ds for k in range(3)])
    groups=[[i for i,r in enumerate(rows) if r['task']==task] for task in cfg['source_tasks']]
    def score(C,k):
        mean=torch.stack([C[ids].mean(0) for ids in groups],1)
        others=[j for j in range(3) if j!=k]
        return (mean[:,k]-mean[:,others].clamp_min(0).mean(1)).clamp_min(0)
    batch=cfg['gradient_batch_pairs'];grads=[];actual=[];diagnostics=[]
    for k in range(3):
        gbar=torch.zeros_like(hidden);delta=source[k];ends=[]
        for off in range(0,len(rows),batch):
            rr=rows[off:off+batch];dd=delta[off:off+len(rr)]
            clean,hh,_,_=forward(rr,gradient=True,phase='path_clean_endpoint')
            assert float((hh-hidden[off:off+len(rr)]).abs().max())<cfg['hidden_atol']
            end,_,_,_=forward(rr,dd,gradient=True,phase='path_source_endpoint')
            ends.append(clean-end)
            for alpha in cfg['path_midpoints']:
                _,_,_,g=forward(rr,alpha*dd,gradient=True,phase='path_gradient')
                gbar[off:off+len(rr)]+=g/len(cfg['path_midpoints'])
                if obj==cfg['objectives'][0] and s==cfg['seeds'][0] and k==0 and off==0 and alpha==cfg['path_midpoints'][0]:
                    direction=g/g.norm(dim=1,keepdim=True).clamp_min(1e-12);eps=.01
                    plus,_,_,_=forward(rr,alpha*dd-eps*direction,gradient=True,phase='path_gradient_check')
                    minus,_,_,_=forward(rr,alpha*dd+eps*direction,gradient=True,phase='path_gradient_check')
                    err=float(((plus-minus)/(2*eps)-(g*direction).sum(1)).abs().max())
                    assert err<.02,err
                    w.checks['finite_difference_of_intervened_shared_prefix_gradient']=True
                    log('PATH_GRADIENT_CHECK',maximum_absolute_error=err,epsilon=eps)
            budget()
        exact=torch.cat(ends);approx=(gbar*delta).sum(1)
        info=dict(component=k,source_margin_decrement=float(exact.mean()),quadrature_mae=float((approx-exact).abs().mean()),quadrature_relative_mae=float((approx-exact).abs().mean()/exact.abs().mean().clamp_min(1e-8)),maximum_gradient_norm=float(gbar.norm(dim=1).max()))
        diagnostics.append(info);grads.append(gbar);actual.append(exact)
        log('SOURCE_PATH_INTEGRATED',objective=obj,source=s,**info)
    G=torch.stack(grads);Y=torch.stack(actual)
    C=zt[None,:,:]*torch.einsum('knd,pd->knp',G,dt)
    C0=zt*(clean_gradient@dt.T)
    path_score=torch.stack([score(C[k],k) for k in range(3)],1)
    wrong_score=torch.stack([score(C[(k+1)%3],k) for k in range(3)],1)
    clean_score=torch.stack([score(C0,k) for k in range(3)],1)
    M,fit=response_fit(C,Y,cfg['path_fit_steps'],cfg['path_ridge'])
    M0,fit0=response_fit(C0[None,:,:].expand(3,-1,-1),Y,cfg['path_fit_steps'],cfg['path_ridge'])
    path=w.run/f'{obj}_s{s}_t{t}_path_relation.npz'
    np.savez_compressed(path,target_members=tp.cpu().numpy(),source_members=sp.cpu().numpy(),source_gate=sg.cpu().numpy(),row_ids=np.asarray([r['row_id'] for r in rows]),task_ids=np.asarray([cfg['source_tasks'].index(r['task']) for r in rows]),source_path_gradient=G.cpu().numpy(),source_finite_effect=Y.cpu().numpy(),target_response_credit=C.cpu().numpy(),clean_response_credit=C0.cpu().numpy(),path_membership=M.cpu().numpy(),clean_membership=M0.cpu().numpy(),source_path_score=path_score.cpu().numpy(),wrong_path_score=wrong_score.cpu().numpy(),clean_score=clean_score.cpu().numpy())
    from run_causalgym_multisite import write
    write(w.run/f'{obj}_s{s}_t{t}_path_fit.json',dict(diagnostics=diagnostics,path_fit=fit,clean_fit=fit0,source_pairs=len(rows),scope='Both fits receive the same exact finite source effects. Only the response design differs. All coefficients are fitted on source-selection pairs before target validation.'))
    log('FUNCTIONAL_PATH_RELATION_FITTED',objective=obj,source=s,path_fit=fit,clean_fit=fit0)
    return dict(clean_gradient_matched=(tp,clean_score),source_path_gradient=(tp,path_score),wrong_path_gradient=(tp,wrong_score),response_membership=(tp,M*path_score),response_clean_membership=(tp,M0*clean_score))


def propose_consensus(cfg,w,D,saes,checked,log,obj,s,t,h,grad,source_panel,tp):
    """Combine cached source-component response paths, excluding the target SAE."""
    import numpy as np,torch
    parent=checked(cfg['path_ensemble_parent']+'/status.json').parent
    import json
    assert json.loads((parent/'status.json').read_text())['status']=='PASS'
    ix=[i for i,r in enumerate(source_panel) if cfg['path_source_range'][0]<=r['row_id']<cfg['path_source_range'][1]]
    rows=[source_panel[i] for i in ix];hidden=h[ix];g0=grad[ix]
    with torch.no_grad():zt=saes[obj,t].encode(hidden)[:,tp]
    dt=D[obj,t][tp];groups=[[i for i,r in enumerate(rows) if r['task']==task] for task in cfg['source_tasks']]
    def score(C):
        means=torch.stack([C[:,ids,:].mean(1) for ids in groups],1)
        return torch.stack([(means[k,k]-means[k,[j for j in range(3) if j!=k]].clamp_min(0).mean(0)).clamp_min(0) for k in range(3)],1)
    allG=[];used=[]
    for source in cfg['seeds']:
        if source==t:continue
        target=cfg['seeds'][(cfg['seeds'].index(source)+1)%len(cfg['seeds'])]
        path=checked(parent/f'{obj}_s{source}_t{target}_path_relation.npz')
        with np.load(path) as data:
            assert np.array_equal(data['row_ids'],[r['row_id'] for r in rows])
            assert np.array_equal(data['task_ids'],[cfg['source_tasks'].index(r['task']) for r in rows])
            allG.append(torch.tensor(data['source_path_gradient'],device=w.device));used.append(source)
    assert len(used)==len(cfg['seeds'])-1 and t not in used
    G=torch.stack(allG);C=zt[None,None,:,:]*torch.einsum('sknd,pd->sknp',G,dt)
    mean_score=score(C.mean(0));robust_score=torch.stack([score(c) for c in C]).amin(0)
    factor=(G*g0[None,None,:,:]).sum(-1)/g0.square().sum(1).clamp_min(1e-16)[None,None,:]
    C0=zt*(g0@dt.T);scale_score=score(factor.mean(0)[:,:,None]*C0[None,:,:])
    path=w.run/f'{obj}_s{s}_t{t}_consensus.npz'
    np.savez_compressed(path,target_members=tp.cpu().numpy(),source_seeds=np.asarray(used),mean_score=mean_score.cpu().numpy(),robust_score=robust_score.cpu().numpy(),scale_score=scale_score.cpu().numpy(),source_role_scores=torch.stack([score(c) for c in C]).cpu().numpy())
    log('FUNCTIONAL_CONSENSUS_READY',objective=obj,source=s,target=t,source_seeds=used,target_source_component_used=False)
    return dict(consensus_mean_credit=(tp,mean_score),consensus_robust_credit=(tp,robust_score),consensus_scale_credit=(tp,scale_score))
