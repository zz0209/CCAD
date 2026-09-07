"""Linear correspondence with both parts of a reciprocal input pair.

The symmetric and antisymmetric projections, and the ridge solution, are
standard linear algebra. This module does not select source groups, use model
outputs, or identify semantic concepts. It predicts uncentered source-group
coordinates; a decoder bias is not part of a group's contribution.
"""
from __future__ import annotations

import numpy as np


def pair_parts(values, donors):
    values = np.asarray(values, dtype=np.float64)
    donors = np.asarray(donors)
    if values.ndim != 2 or donors.shape != (len(values),) or donors.dtype.kind not in 'iu':
        raise ValueError('Expected a matrix and one integer donor per row')
    if np.any(donors < 0) or np.any(donors >= len(values)):
        raise ValueError('Donor outside the panel')
    if not np.array_equal(donors[donors], np.arange(len(values))):
        raise ValueError('The donor operation must be an involution')
    return (values - values[donors]) / 2, (values + values[donors]) / 2


def pair_ridge(x, y, donors, alpha, common_weight=1.0):
    """Fit ||E_minus||² + eta||E_plus||² + n*alpha*||unit*W||².

    One common input unit is derived from contrast variation, with an absolute
    RMS fallback for an entirely inactive contrast. Absolute-active columns
    are retained: dropping contrast-invariant columns would erase exactly
    the information the new consumer requires. No intercept is fitted.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError('Nonempty aligned input/output matrices are required')
    if alpha <= 0 or not np.isfinite(alpha) or common_weight < 0 or not np.isfinite(common_weight):
        raise ValueError('Positive ridge penalty and finite nonnegative common weight required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite input')
    xm, xp = pair_parts(x, donors); ym, yp = pair_parts(y, donors)
    rms = np.sqrt(np.mean(x*x, axis=0))
    active = rms > max(float(rms.max())*1e-8, 1e-12)
    w = np.zeros((x.shape[1], y.shape[1]))
    if not active.any():
        return w, dict(active_features=0, unit=0.0, common_weight=float(common_weight))
    unit = float(np.sqrt(np.mean(xm[:,active]**2)))
    if unit <= 1e-12: unit = float(np.sqrt(np.mean(x[:,active]**2)))
    if common_weight == 1.0:
        # Orthogonal projector identity reduces the exact objective to full-state ridge.
        design=x[:,active]/unit; response=y
    elif common_weight == 0.0:
        design=xm[:,active]/unit; response=ym
    else:
        weight=np.sqrt(common_weight)
        design=np.vstack([xm[:,active], weight*xp[:,active]])/unit
        response=np.vstack([ym, weight*yp])
    penalty=len(x)*alpha
    if design.shape[1] > len(design):
        fitted=design.T@np.linalg.solve(design@design.T+penalty*np.eye(len(design)),response)
    else:
        fitted=np.linalg.solve(design.T@design+penalty*np.eye(design.shape[1]),design.T@response)
    w[active]=fitted/unit
    residual=x@w-y; em,ep=pair_parts(residual,donors)
    return w, dict(active_features=int(active.sum()),unit=unit,alpha=float(alpha),common_weight=float(common_weight),
        contrast_sse=float(np.sum(em**2)),common_sse=float(np.sum(ep**2)),
        objective=float(np.sum(em**2)+common_weight*np.sum(ep**2)+penalty*np.sum(fitted**2)))


def balanced_common_weight(y, donors):
    minus,plus=pair_parts(y,donors)
    return float(np.sum(minus**2)/max(np.sum(plus**2),1e-20))
