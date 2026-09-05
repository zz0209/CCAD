"""Paired-correlation unbalanced OT with a signed linear readout.

This is a generic OT comparison, not SCOTM, Semantic OT, or MAS. The nonnegative
coupling is lifted back to signed, dimensional activation predictions before
projection onto the fixed source coordinate. No endpoint labels are consumed.
"""
from __future__ import annotations

import numpy as np

from ccad.nip_baselines import _unbalanced_log_sinkhorn


def signed_ot_readout(source_codes, target_codes, source_coordinate, weights, *,
                      regularization=0.05, marginal_relaxation=1.0,
                      ridge_fraction=0.001, tolerance=1e-9, max_iterations=1000):
    """Return target-code coefficients and an auditable source-atom coupling.

    Fit on paired discovery rows only. Center both sides using weighted
    conditional means (the resulting intercept cancels in donor differences).
    Cost is 1 - abs(weighted Pearson correlation); each active side has uniform
    reference mass. UOT minimizes cost plus entropy and KL marginal penalties.
    Normalize each coupling row, restore correlation sign and sigma_s/sigma_t,
    then compose with the fixed decoded source coordinate. Finally fit ONE
    scalar ridge gain on the same discovery rows. The output intervention rank
    is the rank of source_coordinate, currently one, not the coupling rank.
    """
    xs=np.asarray(source_codes,dtype=np.float64)
    xt=np.asarray(target_codes,dtype=np.float64)
    coordinate=np.asarray(source_coordinate,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if xs.ndim!=2 or xt.ndim!=2 or len(xs)!=len(xt) or not len(xs):
        raise ValueError('Expected nonempty paired matrices')
    if coordinate.shape!=(xs.shape[1],) or w.shape!=(len(xs),):
        raise ValueError('Coordinate or weights shape mismatch')
    if any(not np.isfinite(a).all() for a in (xs,xt,coordinate,w)) or np.any(w<0) or w.sum()<=0:
        raise ValueError('Finite inputs and nonnegative nonzero weights required')
    if ridge_fraction<=0:
        raise ValueError('Positive ridge fraction required')
    w=w/w.sum()
    # Constant columns can acquire roundoff variance after weighted centering.
    # Exclude them exactly, without a scale-dependent small-variance cutoff.
    varying_s=np.ptp(xs[w>0],axis=0)>0
    varying_t=np.ptp(xt[w>0],axis=0)>0
    xs=xs-w@xs;xt=xt-w@xt
    sigma_s=np.sqrt(np.sum(w[:,None]*xs*xs,axis=0))
    sigma_t=np.sqrt(np.sum(w[:,None]*xt*xt,axis=0))
    active_s=np.flatnonzero((sigma_s>0)&varying_s)
    active_t=np.flatnonzero((sigma_t>0)&varying_t)
    coefficients=np.zeros(xt.shape[1])
    source=xs@coordinate
    base={'active_source_atoms':active_s.tolist(),'active_target_atoms':active_t.tolist(),
          'source_conditional_energy':float(w@(source*source)),
          'cost':'1 - abs(weighted conditional Pearson correlation)',
          'marginals':'uniform over positive-variance atoms on each side',
          'regularization':regularization,'marginal_relaxation':marginal_relaxation,
          'ridge_fraction':ridge_fraction,'rank':1}
    if not len(active_s) or not len(active_t):
        return coefficients,np.zeros((len(active_s),len(active_t))),dict(base,status='NO_CONDITIONAL_VARIATION',gain=0.0)
    ss=xs[:,active_s]/sigma_s[active_s]
    tt=xt[:,active_t]/sigma_t[active_t]
    correlation=np.clip(ss.T@(w[:,None]*tt),-1,1)
    plan,iterations,converged,change=_unbalanced_log_sinkhorn(
        1-np.abs(correlation),np.full(len(active_s),1/len(active_s)),
        np.full(len(active_t),1/len(active_t)),regularization=regularization,
        marginal_relaxation=marginal_relaxation,tolerance=tolerance,max_iterations=max_iterations)
    if not converged:
        raise RuntimeError(f'UOT solver did not converge in {iterations} iterations; change={change}')
    row_mass=plan.sum(axis=1)
    conditional=plan/row_mass[:,None]
    lift=conditional*np.sign(correlation)*sigma_s[active_s,None]/sigma_t[None,active_t]
    coefficients[active_t]=coordinate[active_s]@lift
    prediction=xt@coefficients
    predictor_energy=float(w@(prediction*prediction))
    gain=float(w@(prediction*source))/(predictor_energy*(1+ridge_fraction)) if predictor_energy else 0.0
    coefficients*=gain
    error=float(w@((source-xt@coefficients)**2))
    diagnostics=dict(base,status='OK',iterations=iterations,scaling_change=change,
                     mass=float(plan.sum()),gain=gain,weighted_error=error,
                     target_coefficient_nonzero=int(np.count_nonzero(coefficients)),
                     signed_lift_negative_entries=int(np.count_nonzero(lift<0)))
    return coefficients,plan,diagnostics
