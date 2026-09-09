"""Native group candidates and observable selection scores.

Fixed convex native participation reuses the R15 execution class. New experiments
compare natural, operation, and source-endpoint linearized fitting/selection.
No optimizer or Taylor expansion is asserted to be new mathematics.
"""
from __future__ import annotations
import numpy as np


def bounded_quadratic(Q, b, budget, *, iterations=240):
    """Deterministic projected search, followed by convex fixed-support fitting.

    The box and fixed-support solve are convex. Cardinality support discovery is
    heuristic; no global optimality certificate is returned.
    """
    import torch
    Q=(Q+Q.T)/2
    ridge=Q.diag().mean().clamp_min(1e-12)*1e-5
    Q=Q+ridge*torch.eye(len(Q),device=Q.device,dtype=Q.dtype)
    L=torch.linalg.eigvalsh(Q)[-1].clamp_min(1e-12)
    w=(b/Q.diag().clamp_min(1e-12)).clamp(0,1)
    for step in range(80):
        w=(w-(Q@w-b)/L).clamp(0,1)
        if step%8==7:
            keep=torch.argsort(w,descending=True,stable=True)[:budget]
            mask=torch.zeros_like(w);mask[keep]=1;w=w*mask
    support=torch.argsort(w,descending=True,stable=True)[:budget]
    A=Q[support][:,support];c=b[support]
    lip=torch.linalg.eigvalsh(A)[-1].clamp_min(1e-12)
    v=w[support]
    for _ in range(iterations):v=(v-(A@v-c)/lip).clamp(0,1)
    w.zero_();w[support]=v
    residual=torch.linalg.vector_norm(v-(v-(A@v-c)).clamp(0,1))
    return w,dict(support=int((w>1e-7).sum()),l1=float(w.sum()),objective=float(w@Q@w-2*b@w),fixed_support_projected_gradient_residual=float(residual),ridge=float(ridge),global_optimality_claim=False)


def contribution_moments(Z, D, desired):
    """E|| (Z*w)D - desired ||^2 without an N-by-M-by-D design tensor."""
    Q=(Z.T@Z/len(Z))*(D@D.T)
    b=(Z*(desired@D.T)).mean(0)
    return Q,b


def source_members(delta_codes, signs, budget, *, decoder=None, gradients=None, mode='contrast'):
    """Source-only class contrast; no target information or result filtering."""
    import torch
    contrast=(delta_codes*signs[:,None]).mean(0) if mode=='contrast' else (delta_codes*(gradients@decoder.T)).mean(0)
    score=contrast.abs() if mode=='contrast' else contrast
    members=torch.argsort(score,descending=True,stable=True)[:budget]
    return members,contrast


def source_path_prediction(base, source, gbase, gsource, qsource, qtarget):
    """Cubic Hermite interpolation along the source edit, linear off that path.

    Endpoint values and directional derivatives determine a standard cubic.
    This is a surrogate, with no accuracy bound for unmeasured transverse curvature.
    """
    t=(qtarget*qsource).sum(1)/qsource.square().sum(1).clamp_min(1e-12)
    p0=(gbase*qsource).sum(1);p1=(gsource*qsource).sum(1)
    h=(2*t**3-3*t**2+1)*base+(t**3-2*t**2+t)*p0+(-2*t**3+3*t**2)*source+(t**3-t**2)*p1
    transverse=qtarget-t[:,None]*qsource
    return h+(((1-t[:,None])*gbase+t[:,None]*gsource)*transverse).sum(1)


def candidate_groups(Zs, Zt, means, Ds, Dt, members, task_dzs, task_dzt,
                     gradients, *, budget=64, pool_size=256, semantic_weights=None,source_weights=None):
    """Same-source native candidate family; records each candidate's information."""
    import torch
    from scipy.optimize import linear_sum_assignment
    source_mean,target_mean=means
    sc=Zs[:,members]-source_mean[members];tc=Zt-target_mean
    a=torch.ones(len(members),device=Ds.device) if source_weights is None else source_weights[members]
    desired=(sc*a)@Ds[members];desired_op=(task_dzs[:,members]*a)@Ds[members]
    width=Dt.shape[0];group={};details={}
    cosine=Ds[members]@Dt.T
    xs=Zs[:,members]-Zs[:,members].mean(0);xt=Zt-Zt.mean(0)
    corr=(xs.T@xt)/((xs.square().sum(0).sqrt()[:,None]*xt.square().sum(0).sqrt()[None,:]).clamp_min(1e-12))
    for name,similarity,n in [('cosine_nn',cosine,1),('cosine_top2',cosine,2),('activation_nn',corr.abs(),1)]:
        chosen=torch.argsort(similarity,dim=1,descending=True,stable=True)[:,:n]
        w=torch.zeros(width,device=Dt.device)
        w.scatter_add_(0,chosen.flatten(),a[:,None].expand_as(chosen).flatten()/n)
        w=w.clamp(0,1);keep=torch.argsort(w,descending=True,stable=True)[:budget]
        mask=torch.zeros_like(w);mask[keep]=1;group[name]=w*mask
    _,cols=linear_sum_assignment(-corr.abs().detach().cpu().numpy())
    w=torch.zeros(width,device=Dt.device);w[torch.as_tensor(cols,device=Dt.device)]=a;group['pw_mcc']=w
    choices=torch.argsort(corr.abs(),dim=1,descending=True,stable=True)[:,:4]
    mass=torch.softmax(corr.abs().gather(1,choices)*10,dim=1)
    w=torch.zeros(width,device=Dt.device);w.scatter_add_(0,choices.flatten(),(mass*a[:,None]).flatten());w=w.clamp(0,1)
    keep=torch.argsort(w,descending=True,stable=True)[:budget];mask=torch.zeros_like(w);mask[keep]=1;group['soft_correlation']=w*mask
    if semantic_weights is not None:group['semantic_ot']=semantic_weights
    # Candidate support discovery uses declared data, not held-out outcomes.
    variants=[('native_natural',tc,desired,None),('native_operation',task_dzt,desired_op,None),('native_endpoint',task_dzt,desired_op,gradients)]
    for name,Z,Y,g in variants:
        diag=Z.square().mean(0)*Dt.square().sum(1)
        b=(Z*(Y@Dt.T)).mean(0)
        if g is None:rank=b.clamp_min(0).square()/diag.clamp_min(1e-10)
        else:
            K=Z*(g@Dt.T);response=(g*Y).sum(1);fb=(K*response[:,None]).mean(0);rank=fb.clamp_min(0).square()/K.square().mean(0).clamp_min(1e-10)
        pool=torch.argsort(rank,descending=True,stable=True)[:pool_size]
        P=Z[:,pool];Q,c=contribution_moments(P,Dt[pool],Y)
        if g is not None:
            K=P*(g@Dt[pool].T);a=(g*Y).sum(1);F=K.T@K/len(K);fb=K.T@a/len(K)
            # Scale each loss by source energy; mixing coefficient fixed in cfg.
            physical=Y.square().sum(1).mean().clamp_min(1e-10)
            functional=a.square().mean().clamp_min(1e-10)
            Q=.1*Q/physical+F/functional;c=.1*c/physical+fb/functional
        fitted,diag_fit=bounded_quadratic(Q,c,budget)
        w=torch.zeros(width,device=Dt.device);w[pool]=fitted;group[name]=w
        details[name]=dict(**diag_fit,pool=pool.detach().cpu().tolist(),information='natural paired discovery' if name=='native_natural' else 'original task train code differences and source operation'+(' endpoint gradients' if g is not None else ''))
    # A standard ridge reader is retained as a physically unconstrained reference.
    support=torch.argsort(group['native_natural'],descending=True,stable=True)[:budget]
    U=tc[:,support];A=U.T@U/len(U);ridge=A.diag().mean().clamp_min(1e-10)*.01
    reader=torch.linalg.solve(A+ridge*torch.eye(len(A),device=A.device),U.T@desired/len(U))
    group['random']=torch.zeros(width,device=Dt.device)
    generator=torch.Generator(device=Dt.device).manual_seed(719)
    group['random'][torch.randperm(width,generator=generator,device=Dt.device)[:budget]]=1
    return group,details,dict(support=support,reader=reader,ridge=float(ridge)),cosine,corr


def support_matching_score(similarity, weights):
    """Weighted bidirectional support agreement, with absolute correlation input."""
    selected=weights>1e-7
    if not selected.any():return 0.
    w=weights[selected];S=similarity[:,selected]
    return float(.5*(S.max(1).values.mean()+(S.max(0).values*w).sum()/w.sum()))


def group_semantic_ot(source_acts,target_acts,reference,*,top_k=32):
    """SemanticOT distribution recipe extended explicitly to aggregate groups."""
    from .semantic_context_matching import euclidean_cost,entropic_dual_plan
    from .nip_baselines import _balanced_log_sinkhorn
    def distribution(a):
        keep=np.flatnonzero(a>0);keep=keep[np.argsort(-a[keep],kind='stable')[:top_k]]
        if len(keep)==0:return keep,np.array([])
        return keep,a[keep].astype(float)/a[keep].sum()
    ix,a=distribution(source_acts);iy,b=distribution(target_acts)
    if len(ix)==0 or len(iy)==0:return float('inf'),dict(source_contexts=len(ix),target_contexts=len(iy))
    C=euclidean_cost(reference[ix].astype(float),reference[iy].astype(float));C[ix[:,None]==iy[None,:]]=0
    reg=max(float(np.median(C)),1e-6)*.1
    plan,steps,ok,err=_balanced_log_sinkhorn(C,a,b,regularization=reg,tolerance=1e-6,max_iterations=256)
    if not ok:plan,extra,ok,err=entropic_dual_plan(C,a,b,reg,1e-6);steps+=extra
    return float((plan*C).sum()),dict(source_contexts=len(ix),target_contexts=len(iy),marginal_error=err,converged=bool(ok),iterations=steps,regularization=reg,scope='Aggregate-group SemanticOT extension; fixed top32 contexts, not full original feature-level system')
