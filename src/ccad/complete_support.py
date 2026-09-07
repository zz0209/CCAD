"""Shared support selection for contrast and complete contribution prediction.

The weighted pair decomposition and multi-output orthogonal least-squares
forward selection are standard linear algebra, not new recovery algorithms.
The selected support is shared by all source coordinates; no model output or
evaluation row is used here. Coefficients are subsequently fit by pair_ridge.
"""
from __future__ import annotations
import numpy as np
from ccad.pair_complete_correspondence import pair_parts


def pair_design(x, y, donors, common_weight):
    if not np.isfinite(common_weight) or common_weight < 0:
        raise ValueError('Finite nonnegative common weight required')
    xm, xp = pair_parts(x, donors); ym, yp = pair_parts(y, donors)
    if common_weight == 0:
        return xm, ym
    if common_weight == 1:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    weight = np.sqrt(common_weight)
    return np.vstack([xm, weight*xp]), np.vstack([ym, weight*yp])


def orthogonal_least_squares_support(x, y, budget):
    """Greedily maximize exact unpenalized multi-output residual SSE decrease.

For support S, let q_j=(I-P_S)x_j and R=(I-P_S)Y. Adding j reduces
the optimally refitted SSE by ||q_j.T R||² / ||q_j||². This is the
orthogonal-least-squares criterion, not raw-correlation OMP. It chooses one
shared support for every output, unlike separate per-output sklearn OMPs.
"""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x):
        raise ValueError('Aligned nonempty matrices required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or int(budget)!=budget or budget<1:
        raise ValueError('Finite data and positive integer budget required')
    norm=np.linalg.norm(x,axis=0)
    active=np.flatnonzero(norm>max(float(norm.max())*1e-8,1e-12))
    if not len(active):return active,dict(active_features=0,path=[])
    a=x[:,active]/norm[active];r=y.copy();chosen=[];path=[];basis=[]
    original_sse=float(np.sum(y*y))
    for _ in range(min(int(budget),len(active),len(x))):
        norm2=np.sum(a*a,axis=0);cross=a.T@r
        gain=np.divide(np.sum(cross*cross,axis=1),norm2,out=np.full(len(active),-np.inf),where=norm2>1e-18)
        if chosen:gain[chosen]=-np.inf
        j=int(np.argmax(gain))
        if not np.isfinite(gain[j]) or gain[j]<=1e-14*max(original_sse,1e-20):break
        q=a[:,j].copy()
        # A second pass limits numerical drift for highly correlated SAE codes.
        if basis:
            qs=np.column_stack(basis);q-=qs@(qs.T@q)
        q/=np.linalg.norm(q);before=float(np.sum(r*r))
        r-=q[:,None]*(q@r)[None,:];a-=q[:,None]*(q@a)[None,:]
        chosen.append(j);basis.append(q)
        after=float(np.sum(r*r))
        path.append(dict(member=int(active[j]),predicted_sse_gain=float(gain[j]),actual_sse_gain=before-after,residual_sse=after))
    return active[chosen],dict(active_features=len(active),requested_budget=int(budget),selected_budget=len(chosen),path=path,
        objective='Exact unpenalized forward least-squares SSE reduction; ridge refit is a separate estimator')
