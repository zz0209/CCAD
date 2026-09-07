"""Exact value decomposition of a cone projection in its current active set.

This is a post-hoc algebraic readout of saved projected states. It neither
fits a correspondence nor claims differentiability across active boundaries.
"""
from __future__ import annotations
import numpy as np


def active_state_operator(projected_state,metric):
    """Construct B such that u=B v for the recorded NNLS free set u_i>0.

For free coordinates F and constrained A, B_FF=I, B_FA=M_FF^-1 M_FA,
and B_A*=0. At a boundary this is a value representation, not a unique
Jacobian. It must not replace reprojecting a changed donor/recipient input.
"""
    u=np.asarray(projected_state,dtype=np.float64);m=np.asarray(metric,dtype=np.float64)
    if u.ndim!=1 or m.shape!=(len(u),len(u)) or not np.isfinite(u).all() or np.any(u<0):
        raise ValueError('A nonnegative projected state and matching metric required')
    free=np.flatnonzero(u>0);active=np.flatnonzero(u==0);b=np.zeros_like(m)
    b[free,free]=1
    if len(free) and len(active):b[np.ix_(free,active)]=np.linalg.solve(m[np.ix_(free,free)],m[np.ix_(free,active)])
    return b


def contextual_allocation_terms(selected_codes,allocation,projected_state,metric):
    """One row per target member, summing to the current projected source state."""
    x=np.asarray(selected_codes,dtype=np.float64);h=np.asarray(allocation,dtype=np.float64)
    if x.ndim!=1 or h.ndim!=2 or h.shape[0]!=len(x) or not np.isfinite(x).all() or not np.isfinite(h).all():
        raise ValueError('Finite selected code vector and target-by-source allocation required')
    b=active_state_operator(projected_state,metric)
    if h.shape[1]!=len(b):raise ValueError('Source state width differs from allocation')
    return x[:,None]*(h@b.T),b
