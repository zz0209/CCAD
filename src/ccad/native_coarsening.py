"""Anchored soft coarsening of actual SAE contribution operators.

The objective is a bounded convex quadratic, solved with ordinary L-BFGS-B.
The scientific hypothesis is about compatible intervention groups, not novelty
of the numerical optimizer. Both groups use native coefficients in [0, 1].
"""
from __future__ import annotations
import numpy as np


def fit_groups(Kss, Kst, Ktt, anchor, *, penalty=.01, maxiter=400,
               fixed_source=None):
    from scipy.optimize import minimize
    Kss=np.asarray(Kss,dtype=np.float64);Kst=np.asarray(Kst,dtype=np.float64)
    Ktt=np.asarray(Ktt,dtype=np.float64);ns,nt=Kst.shape
    assert Kss.shape==(ns,ns) and Ktt.shape==(nt,nt)
    assert 0<=anchor<ns and penalty>=0
    energy=max(float(Kss[anchor,anchor]),1e-12)
    scale=np.sqrt(np.maximum(np.r_[np.diag(Kss),np.diag(Ktt)],1e-20)/energy)
    A=np.block([[Kss,-Kst],[-Kst.T,Ktt]])/energy
    A=(A+A.T)/2
    bounds=[(0.,1.)]*(ns+nt);bounds[anchor]=(1.,1.)
    initial=np.zeros(ns+nt);initial[anchor]=1.
    if fixed_source is not None:
        fixed_source=np.asarray(fixed_source,dtype=float)
        assert fixed_source.shape==(ns,) and np.all((fixed_source>=0)&(fixed_source<=1))
        bounds[:ns]=[(float(v),float(v)) for v in fixed_source]
        initial[:ns]=fixed_source
    cost=penalty*scale;cost[anchor]=0.
    def objective(v):
        Av=A@v
        return float(v@Av+cost@v),2*Av+cost
    result=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',
                    options=dict(maxiter=maxiter,ftol=1e-12,gtol=1e-8,maxls=30))
    s,t=result.x[:ns],result.x[ns:]
    residual=float(s@Kss@s-2*s@Kst@t+t@Ktt@t)
    return s,t,dict(success=bool(result.success),message=str(result.message),
        iterations=int(result.nit),objective=float(result.fun),penalty=penalty,
        anchor_energy=energy,absolute_residual=max(0.,residual),
        source_members=int((s>1e-5).sum()),target_members=int((t>1e-5).sum()),
        source_weight_sum=float(s.sum()),target_weight_sum=float(t.sum()),
        source_energy=float(s@Kss@s),target_energy=float(t@Ktt@t))


def contribution_gram(Za, Da, Zb, Db):
    """Exact mean Gram of vector-valued contributions z_i(x) d_i.

Pass actual nonnegative states for zero-ablation fitting. Passing states
centered by the independent mean split instead measures Phi variation.
"""
    assert len(Za)==len(Zb) and Za.shape[1]==len(Da) and Zb.shape[1]==len(Db)
    return (Za.T@Zb/len(Za))*(Da@Db.T)
