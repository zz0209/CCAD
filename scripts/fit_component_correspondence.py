"""Convex native component allocation with an explicit per-feature capacity.

For any component weights c in [0,1]^K, nonnegative rows summing to at
most one imply 0 <= M c <= 1. This is an operation constraint, not a
claim of semantic disentanglement. The quadratic is a standard least
squares fit of decoder contribution fields.
"""
import math


def project_rows(x, allowed=None, capacity=True):
    import torch
    if allowed is not None:
        x = x.masked_fill(~allowed, -1e30)
    positive = x.clamp_min(0)
    if not capacity:
        return positive.clamp_max(1)
    ordered = x.sort(dim=1, descending=True).values
    arange = torch.arange(1, x.shape[1]+1, device=x.device, dtype=x.dtype)
    excess = (ordered.cumsum(1)-1)/arange
    rho = (ordered > excess).sum(1).clamp_min(1)-1
    theta = excess.gather(1, rho[:, None])
    return torch.where(positive.sum(1, keepdim=True) <= 1,
                       positive, (x-theta).clamp_min(0))


def fit(K, B, *, steps=800, ridge_fraction=.001, allowed=None, capacity=True):
    import torch
    scale = K.diag().mean().clamp_min(1e-12)
    A = K/scale + ridge_fraction*torch.eye(len(K), device=K.device, dtype=K.dtype)
    rhs = B/scale
    lipschitz = torch.linalg.eigvalsh(A)[-1].clamp_min(1e-12)
    x = torch.zeros_like(B); y = x.clone(); t = 1.
    for _ in range(steps):
        new = project_rows(y-(A@y-rhs)/lipschitz, allowed, capacity)
        tn = (1+math.sqrt(1+4*t*t))/2
        y = new+(t-1)/tn*(new-x); x = new; t = tn
    residual = x-project_rows(x-(A@x-rhs)/lipschitz, allowed, capacity)
    return x, dict(steps=steps, projected_gradient_max=float(residual.abs().max()),
                   maximum_row_sum=float(x.sum(1).max()),
                   objective_without_source_constant=float((x*(K@x-2*B)).sum()),
                   ridge_fraction=ridge_fraction)
