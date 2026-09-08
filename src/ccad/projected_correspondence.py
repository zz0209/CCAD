"""Established ridge/RRR with a physical operation metric and identity prior.

Rows are shared-hook contributions centered on an independent mean split.
The source operation family and all fitting/selection data are external.
No endpoint labels, source-query selection or native-code claims occur here.
"""
from __future__ import annotations
import itertools
import numpy as np


def ordered_operator(projectors, control):
    p=np.asarray(projectors,dtype=np.float64);c=np.asarray(control,dtype=np.float64)
    if p.ndim!=3 or p.shape[1]!=p.shape[2] or c.shape!=(len(p),):raise ValueError('Projector/control dimensions differ')
    if not np.isfinite(c).all() or np.any(c<0) or np.any(c>1):raise ValueError('Controls must lie in the unit cube')
    eye=np.eye(p.shape[1]);remaining=eye.copy()
    for a in range(len(p)):remaining=(eye-c[a]*p[a])@remaining
    return eye-remaining


def family_metric(projectors):
    controls=[list(c) for c in itertools.product([0,1],repeat=len(projectors)) if any(c)]
    operators=[ordered_operator(projectors,c) for c in controls]
    m=sum(b.T@b for b in operators);m=(m+m.T)/2
    values,vectors=np.linalg.eigh(m);keep=values>max(float(values[-1])*1e-9,1e-12)
    factor=vectors[:,keep]*np.sqrt(values[keep])
    inverse=(vectors[:,keep]/np.sqrt(values[keep])).T
    return dict(controls=controls,operators=operators,matrix=m,factor=factor,inverse=inverse,
                eigenvalues=values,effective_rank=int(keep.sum()))


def penalized_low_rank(full, cross, rank, factor=None, inverse=None):
    """Rank-r ridge minimizer, with optional output metric factor.

full=A^-1 X.T Y and cross=X.T Y for already weighted observations.
The penalized right Gram is Y.T X A^-1 X.T Y, not (X full).T X full.
If L is supplied, optimize C=W L and return W=C L^+ in hook coordinates.
"""
    gram=cross.T@full
    if factor is not None:
        gram=factor.T@gram@factor;coefficient=full@factor
    else:coefficient=full
    gram=(gram+gram.T)/2
    values,vectors=np.linalg.eigh(gram)
    order=np.argsort(-values,kind='stable')
    numerical=int(np.sum(values>max(float(values[-1])*1e-10,1e-14)))
    keep=min(rank,numerical);v=vectors[:,order[:keep]]
    left=coefficient@v;right=v.T if factor is None else v.T@inverse
    return left,right,dict(requested_rank=rank,effective_rank=keep,singular_energy=values[order].clip(min=0).tolist())


def balanced_document_weights(document_ids):
    docs=np.asarray(document_ids);_,inv,counts=np.unique(docs,return_inverse=True,return_counts=True)
    w=1/counts[inv];return w/w.sum()
