"""Classical nonnegative projection of absolute source allocations.

Project each absolute state before forming a signed donor-recipient difference.
The metric-state guarantee does not imply a contrast or finite-output guarantee.
"""
from __future__ import annotations
import numpy as np
from ccad.component_correspondence import component_metric, metric_factor


def project_source_state(values, decoder, method='cone_half'):
    """Return projected rows and the state metric; preserve input shape.

    Decoder rows are fixed source feature axes. The signed correspondence is
    unchanged; the context-dependent projection is an additional operation.
    """
    v=np.asarray(values,dtype=np.float64);d=np.asarray(decoder,dtype=np.float64)
    if v.ndim<1 or d.ndim!=2 or v.shape[-1]!=len(d) or not np.isfinite(v).all():
        raise ValueError('Finite source allocations and matching decoder required')
    if method=='clip':return np.maximum(v,0),np.eye(len(d))
    if method!='cone_half':raise ValueError('Unknown source-state projection')
    from scipy.optimize import nnls
    metric=component_metric(d,'half');a=metric_factor(metric).T
    flat=v.reshape(-1,len(d));result=flat.copy()
    for i in np.flatnonzero(np.any(flat<0,axis=1)):
        result[i]=nnls(a,a@flat[i],maxiter=10*len(d))[0]
    return result.reshape(v.shape),metric


def projection_diagnostics(values, projected, source, metric):
    v,u,z,m=map(lambda x:np.asarray(x,dtype=np.float64),(values,projected,source,metric))
    q=lambda x:np.einsum('ni,ij,nj->n',x,m,x)
    gradient=(u-v)@m
    scale=max(float(np.max(np.abs(gradient))),1.)
    gap=q(v-z)-q(u-z)-q(u-v)
    return dict(negative_input_fraction=float(np.mean(v<0)),negative_output_fraction=float(np.mean(u<0)),
                source_nonnegative=bool(np.all(z>=0)),metric_error_before=float(np.sum(q(v-z))),
                metric_error_after=float(np.sum(q(u-z))),projection_displacement=float(np.sum(q(u-v))),
                minimum_projection_inequality_slack=float(np.min(gap)),
                dual_violation_relative=float(max(0.,-np.min(gradient))/scale),
                complementarity_relative=float(np.max(np.abs(u*gradient))/max(scale*float(np.max(np.abs(u))),1.)))
