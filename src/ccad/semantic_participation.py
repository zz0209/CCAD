"""Bounded, overlapping native controls with explicit joint-operation semantics.

Algebraic fuzzy union and sparse hypercube projection are established operations.
This module does not claim novelty for either, or infer semantic identity.
"""
from __future__ import annotations


def union_participation(gates, controls):
    """Coordinatewise donor weight 1 - product_a(1 - c_a G_ja)."""
    import torch
    return 1-torch.prod(1-gates*controls,dim=-1)


def project_sparse_gates(value, weight_budget, exclusive=False):
    """Exact Euclidean projection onto a sparse [0,1] gate matrix.

Exclusive additionally permits at most one nonzero per row. The retained
positive-weight budget is identical in both classes; row counts may differ.
This projection does not make the full downstream optimization convex.
"""
    import torch
    with torch.no_grad():
        feasible=value.clamp(0,1)
        gain=2*value*feasible-feasible.square()
        result=torch.zeros_like(value)
        if exclusive:
            best_gain,column=gain.max(dim=1)
            row=torch.argsort(best_gain,descending=True,stable=True)[:weight_budget]
            result[row,column[row]]=feasible[row,column[row]]
        else:
            keep=torch.argsort(gain.flatten(),descending=True,stable=True)[:weight_budget]
            result.view(-1)[keep]=feasible.flatten()[keep]
        value.copy_(result)


def state_delta(base_codes, donor_codes, decoder, gates, controls):
    """Actual decoder edit, preserving all unselected codes and base residual."""
    alpha=union_participation(gates,controls)
    return ((donor_codes-base_codes)*alpha)@decoder
