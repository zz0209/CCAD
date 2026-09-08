"""Sparse convex native participation in source-controlled donor operations.

G maps source component controls to target participation. Its entries are not
signed change-of-basis coefficients: signed physical effects use the actual
donor-minus-base target codes and the target decoder.
"""
from __future__ import annotations
import numpy as np


def project_participation_rows(value, budget):
    """Euclidean row simplex projection followed by top-row hard thresholding.

The row projection is exact. Their composition is a feasible projected-search
step, not an assertion of globally optimal joint sparse matrix fitting.
"""
    import torch
    with torch.no_grad():
        positive = value.clamp_min(0)
        active = positive.sum(dim=1)>1
        if active.any():
            v = value[active]
            ordered = torch.sort(v,dim=1,descending=True).values
            j = torch.arange(1,v.shape[1]+1,device=v.device,dtype=v.dtype)
            excess = ordered.cumsum(1)-1
            rho = (ordered-excess/j>0).sum(1)
            theta = excess.gather(1,(rho-1)[:,None])/rho[:,None]
            positive[active] = (v-theta).clamp_min(0)
        if budget<positive.shape[0]:
            keep = torch.argsort(torch.linalg.vector_norm(positive,dim=1),descending=True,stable=True)[:budget]
            selected = torch.zeros_like(positive)
            selected[keep] = positive[keep]
            positive = selected
        value.copy_(positive)
    return value


def participant_delta(base, donor, decoder, participation, controls):
    """NumPy reference; position axes are preserved and only code members mix."""
    base,donor,decoder,participation,controls = map(np.asarray,(base,donor,decoder,participation,controls))
    weight = participation@controls
    code_change = (donor-base)*weight
    return code_change@decoder,base+code_change


def project_exclusive_rows(value, budget):
    """Nearest feasible row-exclusive matrix, with at most budget active rows.

    Each row retains one coordinate in [0,1]. Selecting rows by the actual
    reduction in squared distance gives the Euclidean projection, including
    inputs outside the box. Ties use the first column and stable row order.
    """
    import torch
    with torch.no_grad():
        clipped = value.clamp(0,1)
        gains = 2*value*clipped-clipped.square()
        best,columns = gains.max(dim=1)
        rows = torch.argsort(best,descending=True,stable=True)[:budget]
        result = torch.zeros_like(value)
        result[rows,columns[rows]] = clipped[rows,columns[rows]]
        value.copy_(result)
    return value


def feasibility(participation, tolerance=1e-6):
    g = np.asarray(participation)
    return dict(nonnegative=bool(g.min(initial=0)>=-tolerance),
                row_sum_at_most_one=bool(g.sum(axis=1).max(initial=0)<=1+tolerance),
                actual_members=int(np.any(g!=0,axis=1).sum()),
                nonzero_entries=int(np.count_nonzero(g)),
                shared_members=int((np.count_nonzero(g,axis=1)>1).sum()),
                row_sum_max=float(g.sum(axis=1).max(initial=0)))
