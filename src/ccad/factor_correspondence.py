"""Compact contribution coordinates and matched linear correspondence controls.

Group support selection, ridge, reduced-rank regression, assignment, and OT are
standard algorithms. The output coefficients always use one frozen source
operation. No language-model endpoint is consumed here.
"""
from __future__ import annotations
import numpy as np
from ccad.nip_baselines import _balanced_log_sinkhorn


def compact_source(dz, decoder, discovery, sign, budget):
    dz=np.asarray(dz,dtype=np.float64);decoder=np.asarray(decoder,dtype=np.float64)
    oriented=np.mean(dz[discovery]*sign[discovery,None],axis=0)
    direction=oriented@decoder;direction/=max(np.linalg.norm(direction),1e-12)
    score=oriented*(decoder@direction);support=np.argsort(-np.abs(score),kind='stable')[:budget]
    # SVD retains the numerical decoder span; no task-variance truncation.
    _,singular,vt=np.linalg.svd(decoder[support],full_matrices=False)
    rank=int(np.sum(singular>singular[0]*1e-10));basis=vt[:rank].T
    coefficients=decoder[support]@basis;coordinates=dz[:,support]@coefficients
    native=dz[:,support]@decoder[support];reconstructed=coordinates@basis.T
    error=float(np.linalg.norm(native-reconstructed)/max(np.linalg.norm(native),1e-12))
    if error>1e-9:raise ValueError('Source coordinate operation differs from compact native teacher')
    singular_process=np.linalg.svd(coordinates[discovery],compute_uv=False)
    return dict(support=support,basis=basis,coefficients=coefficients,coordinates=coordinates,score=score[support],rank=rank,span_error=error,discovery_singular_values=singular_process)


def ridge(x,y,alpha,scaling='feature_rms'):
    """Ridge in per-feature RMS or common native decoder units; no intercept."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    scale=np.sqrt(np.mean(x*x,axis=0));active=scale>max(float(scale.max())*1e-8,1e-12)
    out=np.zeros((x.shape[1],y.shape[1]))
    if not active.any():return out
    unit=scale[active] if scaling=='feature_rms' else np.full(np.sum(active),np.sqrt(np.mean(x[:,active]**2)))
    if scaling not in ['feature_rms','native_units']:raise ValueError('Unknown code scaling')
    xx=x[:,active]/unit;penalty=len(xx)*alpha
    if xx.shape[1]>len(xx):w=xx.T@np.linalg.solve(xx@xx.T+penalty*np.eye(len(xx)),y)
    else:w=np.linalg.solve(xx.T@xx+penalty*np.eye(xx.shape[1]),xx.T@y)
    out[active]=w/unit[:,None];return out


def choose_ridge(x,y,fit,calibration,alphas,scaling='feature_rms'):
    errors=[]
    for alpha in alphas:
        w=ridge(x[fit],y[fit],alpha,scaling);errors.append(float(np.mean((x[calibration]@w-y[calibration])**2)))
    index=int(np.argmin(errors));return alphas[index],errors


def group_support(x,y,budget,iterations=10000,scaling='feature_rms'):
    """Established MultiTaskLasso path followed by an explicit support budget.

    RMS scaling is fitted only to discovery. Three fixed lambda fractions are
    used until enough nonzero rows exist; final top-budget rows are refit by
    the common ridge kernel. A diagnostic records sparsity and solver gap.
    """
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    scale=np.sqrt(np.mean(x*x,axis=0));active=np.flatnonzero(scale>max(float(scale.max())*1e-8,1e-12))
    if not len(active):return active,dict(active_features=0)
    unit=scale[active] if scaling=='feature_rms' else np.full(len(active),np.sqrt(np.mean(x[:,active]**2)));xx=x[:,active]/unit
    from sklearn.linear_model import MultiTaskLasso
    cross=xx.T@y/len(xx);lam_max=float(np.linalg.norm(cross,axis=1).max());diag=[]
    model=MultiTaskLasso(fit_intercept=False,max_iter=iterations,tol=1e-5,warm_start=True,selection='cyclic')
    for fraction in [.2,.05,.01]:
        lam=max(lam_max*fraction,1e-12);model.set_params(alpha=lam);model.fit(xx,y);w=model.coef_.T
        strength=np.linalg.norm(w,axis=1);nonzero=int(np.sum(strength>1e-10));gap_tolerance=1e-5*float(np.mean(np.sum(y*y,axis=1)));diag.append(dict(lambda_fraction=fraction,nonzero=nonzero,iterations=int(model.n_iter_),dual_gap=float(model.dual_gap_),dual_gap_tolerance=gap_tolerance,converged=bool(model.dual_gap_<=gap_tolerance),objective=float(np.mean(np.sum((xx@w-y)**2,axis=1))/2+lam*strength.sum())))
        if nonzero>=min(budget,len(active)):break
    order=np.argsort(-strength,kind='stable')[:min(budget,max(1,nonzero))]
    return active[order],dict(active_features=len(active),path=diag,scaling=scaling,selection='group-L1 nonzero coefficient norm then at most requested budget; common ridge refit')


def assigned_readout(xs,xt,source_coefficients):
    """Distinct target member per source member, maximized conditional correlation."""
    from scipy.optimize import linear_sum_assignment
    ss=np.sqrt(np.mean(xs*xs,axis=0));ts=np.sqrt(np.mean(xt*xt,axis=0));active=np.flatnonzero(ts>max(float(ts.max())*1e-8,1e-12));corr=xs.T@xt[:,active]/len(xs)/np.maximum(ss[:,None]*ts[active],1e-12)
    ii,jj=linear_sum_assignment(-np.abs(corr));chosen=active[jj];w=np.zeros((xt.shape[1],source_coefficients.shape[1]))
    slopes=np.sum(xs[:,ii]*xt[:,chosen],axis=0)/np.maximum(np.sum(xt[:,chosen]**2,axis=0),1e-12)
    for i,j,slope in zip(ii,chosen,slopes):w[j]+=slope*source_coefficients[i]
    return w,dict(source_indices=ii.tolist(),target_members=chosen.tolist(),slopes=slopes.tolist(),objective='maximum absolute conditional correlation; distinct target indices; independently least-square calibrated pairs')


def conditional_ot(xs,xt,source_coefficients,budget=None):
    """Balanced concept-conditioned correlation OT, lifted to signed units.

    This operational adaptation is not an exact reproduction of Bhalla et al.'s
    similarity plot. OT uses conditional code changes and uniform marginals.
    The compact variant retains targets by their largest source correlation
    before solving OT at the explicit member budget.
    """
    ss=np.sqrt(np.mean(xs*xs,axis=0));ts=np.sqrt(np.mean(xt*xt,axis=0));si=np.flatnonzero(ss>1e-10);ti=np.flatnonzero(ts>max(float(ts.max())*1e-8,1e-12))
    corr=np.clip(xs[:,si].T@xt[:,ti]/len(xs)/(ss[si,None]*ts[ti]),-1,1)
    if budget is not None and len(ti)>budget:
        keep=np.argsort(-np.max(np.abs(corr),axis=0),kind='stable')[:budget];ti=ti[keep];corr=corr[:,keep]
    plan,iterations,converged,error=_balanced_log_sinkhorn(1-np.abs(corr),np.full(len(si),1/len(si)),np.full(len(ti),1/len(ti)),regularization=.05,tolerance=1e-8,max_iterations=50000)
    if not converged:raise RuntimeError('Conditional OT marginal tolerance failed')
    lift=plan/plan.sum(axis=1,keepdims=True)*np.sign(corr)*ss[si,None]/ts[ti]
    w=np.zeros((xt.shape[1],source_coefficients.shape[1]));w[ti]=lift.T@source_coefficients[si]
    prediction=xt@w;teacher=xs@source_coefficients;gain=float(np.sum(prediction*teacher)/max(np.sum(prediction**2),1e-12));w*=gain
    return w,dict(source_members=si.tolist(),target_members=ti.tolist(),gain=gain,iterations=iterations,max_iterations=50000,marginal_error=error,plan=plan.tolist(),signed_lift=lift.tolist(),variant='conditional correlation balanced OT plus one discovery gain; adapted operational comparison')


def fit_controls(xt,xraw,xs,source,fit,calibration,discovery,budgets,alphas,random_seed=0):
    """All controls predict the exact same frozen source coordinate matrix."""
    y=source['coordinates'];out={};diagnostics={};alpha,errors=choose_ridge(xt,y,fit,calibration,alphas);full=ridge(xt[discovery],y[discovery],alpha)
    out['full_code_ridge']=(xt@full,full);diagnostics['full_code_ridge']=dict(alpha=alpha,calibration_mse=errors)
    # RRR on the same task contrasts is the operational shared-component
    # baseline; this does not claim to reproduce every dSCA preprocessing step.
    _,_,vt=np.linalg.svd(xt[discovery]@full,full_matrices=False)
    for rank in [1,4]:
        v=vt[:min(rank,len(vt))].T;w=full@v@v.T;out['contrast_rrr_rank'+str(rank)]=(xt@w,w);diagnostics['contrast_rrr_rank'+str(rank)]=dict(rank=v.shape[1],alpha=alpha)
    ra,re=choose_ridge(xraw,y,fit,calibration,alphas);rw=ridge(xraw[discovery],y[discovery],ra);out['raw_ridge']=(xraw@rw,None);diagnostics['raw_ridge']=dict(alpha=ra,calibration_mse=re,coefficients_shape=list(rw.shape))
    variance=np.mean(xt[discovery]**2,axis=0);cross=xt[discovery].T@y[discovery]/np.sum(discovery);single_w=np.divide(cross,variance[:,None]*(1+alpha),out=np.zeros_like(cross),where=variance[:,None]>1e-16)
    single_error=np.mean(np.sum(y[discovery]**2,axis=1))-2*np.sum(single_w*cross,axis=1)+variance*np.sum(single_w**2,axis=1);atom=int(np.argmin(single_error));aw=np.zeros_like(full);aw[atom]=single_w[atom];out['single_atom']=(xt@aw,aw);diagnostics['single_atom']=dict(members=[atom],candidate_count=xt.shape[1],objective='exact minimum discovery vector error over all individually ridge-fitted atoms',alpha=alpha)
    one,od=assigned_readout(xs[discovery][:,source['support']],xt[discovery],source['coefficients']);out['one_to_one']=(xt@one,one);diagnostics['one_to_one']=od
    ot,otd=conditional_ot(xs[discovery][:,source['support']],xt[discovery],source['coefficients']);out['conditional_ot_full']=(xt@ot,ot);diagnostics['conditional_ot_full']=otd
    rng=np.random.default_rng(random_seed);rms=np.sqrt(np.mean(xt[discovery]**2,axis=0));active=np.flatnonzero(rms>max(float(rms.max())*1e-8,1e-12));dense_energy=rms*np.linalg.norm(full,axis=1)
    for budget in budgets:
        selected,gd=group_support(xt[discovery],y[discovery],budget);w=np.zeros_like(full);w[selected]=ridge(xt[discovery][:,selected],y[discovery],alpha);out['fcc_group'+str(budget)]=(xt@w,w);diagnostics['fcc_group'+str(budget)]=dict(members=selected.tolist(),alpha=alpha,**gd)
        # Independent support ranking from the dense fit, same ridge refit.
        dense_keep=np.argsort(-dense_energy,kind='stable')[:budget];dw=np.zeros_like(full);dw[dense_keep]=ridge(xt[discovery][:,dense_keep],y[discovery],alpha);out['dense_select_refit'+str(budget)]=(xt@dw,dw);diagnostics['dense_select_refit'+str(budget)]=dict(members=dense_keep.tolist(),alpha=alpha)
        random=rng.choice(active,min(budget,len(active)),replace=False);rr=np.zeros_like(full);rr[random]=ridge(xt[discovery][:,random],y[discovery],alpha);out['random_refit'+str(budget)]=(xt@rr,rr);diagnostics['random_refit'+str(budget)]=dict(members=random.tolist(),alpha=alpha)
        ot,od=conditional_ot(xs[discovery][:,source['support']],xt[discovery],source['coefficients'],budget);out['conditional_ot'+str(budget)]=(xt@ot,ot);diagnostics['conditional_ot'+str(budget)]=od
    # Unit decoder columns already put SAE codes in common hook-space units.
    # Retain the per-feature scaling comparison while testing this physically
    # meaningful alternative; do not discard the earlier scaling's failures.
    alpha_native,errors_native=choose_ridge(xt,y,fit,calibration,alphas,'native_units');nw=ridge(xt[discovery],y[discovery],alpha_native,'native_units')
    out['full_code_native_units']=(xt@nw,nw);diagnostics['full_code_native_units']=dict(alpha=alpha_native,calibration_mse=errors_native,scaling='one common RMS factor, unit-norm decoder feature units retained')
    na,ne=choose_ridge(xraw,y,fit,calibration,alphas,'native_units');nr=ridge(xraw[discovery],y[discovery],na,'native_units');out['raw_native_units']=(xraw@nr,None);diagnostics['raw_native_units']=dict(alpha=na,calibration_mse=ne)
    for budget in budgets:
        selected,sd=group_support(xt[discovery],y[discovery],budget,scaling='native_units');w=np.zeros_like(full);w[selected]=ridge(xt[discovery][:,selected],y[discovery],alpha_native,'native_units');name='fcc_native_units'+str(budget);out[name]=(xt@w,w);diagnostics[name]=dict(members=selected.tolist(),alpha=alpha_native,**sd)
        energy=rms*np.linalg.norm(nw,axis=1);keep=np.argsort(-energy,kind='stable')[:budget];w=np.zeros_like(full);w[keep]=ridge(xt[discovery][:,keep],y[discovery],alpha_native,'native_units');name='dense_native_select'+str(budget);out[name]=(xt@w,w);diagnostics[name]=dict(members=keep.tolist(),alpha=alpha_native)
        random=rng.choice(active,min(budget,len(active)),replace=False);w=np.zeros_like(full);w[random]=ridge(xt[discovery][:,random],y[discovery],alpha_native,'native_units');name='random_native_refit'+str(budget);out[name]=(xt@w,w);diagnostics[name]=dict(members=random.tolist(),alpha=alpha_native)
    return out,diagnostics
