"""Explicit target-decoder operations and their feasibility diagnostics.

The greedy subspace selection and projected least squares are classical
algorithms. Reading members and writing members have separate budgets.
"""
from __future__ import annotations

import numpy as np


def writable_support(decoder, signal, budget, eligible=None):
    """Select decoder rows minimizing a multi-response projection residual.

    ``signal`` contains source-only, covariance-weighted operation directions.
    Rank-one updates avoid forming the full dictionary Gram matrix.
    """
    d = np.asarray(decoder, dtype=np.float64)
    b = np.asarray(signal, dtype=np.float64)
    if d.ndim != 2 or b.ndim != 2 or d.shape[1] != b.shape[1]:
        raise ValueError('Decoder and operation directions must share a hook dimension')
    allowed = np.ones(len(d), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool).copy()
    norms = np.einsum('ij,ij->i', d, d)
    residual_norms = norms.copy()
    correlations = d @ b.T
    qvectors = []
    selected = []
    gains = []
    residual_energy = float(np.sum(b * b))
    for _ in range(min(int(budget), int(allowed.sum()))):
        valid = allowed & (residual_norms > np.maximum(norms * 1e-10, 1e-14))
        if not valid.any():
            break
        scores = np.full(len(d), -np.inf)
        scores[valid] = np.sum(correlations[valid] ** 2, axis=1) / residual_norms[valid]
        index = int(np.argmax(scores))
        q = d[index].copy()
        if qvectors:
            basis = np.asarray(qvectors)
            for _ in range(2):
                q -= (basis @ q) @ basis
        q /= np.linalg.norm(q)
        dot = d @ q
        bq = b @ q
        gain = float(bq @ bq)
        correlations -= dot[:, None] * bq[None, :]
        residual_norms = np.maximum(0, residual_norms - dot * dot)
        allowed[index] = False
        qvectors.append(q)
        selected.append(index)
        residual_energy = max(0., residual_energy - gain)
        gains.append(dict(index=index, gain=gain, residual_energy=residual_energy))
    return np.asarray(selected, dtype=int), gains


def project_native(vectors, target_codes, decoder, constrained=True, max_steps=800, tolerance=1e-8):
    """Minimize ||u D - v||^2 subject to u >= -z, with KKT diagnostics.

    The untouched target coordinates and original reconstruction residual are
    retained by the caller. This is a code intervention, not an assertion that
    the resulting code is a TopK encoder output.
    """
    v = np.asarray(vectors, dtype=np.float64)
    z = np.asarray(target_codes, dtype=np.float64)
    d = np.asarray(decoder, dtype=np.float64)
    if v.shape != (len(z), d.shape[1]) or z.shape[1] != len(d):
        raise ValueError('Native operation dimensions disagree')
    if np.min(z) < -1e-12:
        raise ValueError('Expected nonnegative target states')
    gram = d @ d.T
    rhs = v @ d.T
    initial = np.linalg.lstsq(gram, rhs.T, rcond=1e-10)[0].T
    if not constrained:
        return initial, dict(steps=0, relative_projected_gradient=None,
                            minimum_final_state=float(np.min(z + initial)))
    step = 1. / max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
    u = np.maximum(initial, -z)
    y = u.copy()
    acceleration = 1.
    reference = max(float(np.linalg.norm(rhs)), 1e-12)
    for iteration in range(int(max_steps)):
        new = np.maximum(y - step * (y @ gram - rhs), -z)
        next_acceleration = (1. + np.sqrt(1. + 4. * acceleration ** 2)) / 2.
        y = new + (acceleration - 1.) / next_acceleration * (new - u)
        u = new
        acceleration = next_acceleration
        if iteration % 20 == 19:
            projected_gradient = (u - np.maximum(u - step * (u @ gram - rhs), -z)) / step
            if np.linalg.norm(projected_gradient) / reference < tolerance:
                break
    projected_gradient = (u - np.maximum(u - step * (u @ gram - rhs), -z)) / step
    return u, dict(steps=iteration + 1,
                   relative_projected_gradient=float(np.linalg.norm(projected_gradient) / reference),
                   minimum_final_state=float(np.min(z + u)),
                   constrained_squared_error=float(np.sum((u @ d - v) ** 2)),
                   clipped_initial_squared_error=float(np.sum((np.maximum(initial, -z) @ d - v) ** 2)))


def adaptive_writable_support(vectors, target_codes, decoder, budget, eligible, device='cuda:0'):
    """Classical matching pursuit with the exact one-coordinate state bound.

    Selection reads the candidate dictionary and current target states, which
    must be included in the information/compute budget. It never reads an LM
    endpoint. Each selected coordinate is changed at most once before the
    caller's joint constrained refit.
    """
    import torch
    candidates = np.flatnonzero(eligible)
    limit = min(int(budget), len(candidates))
    if limit <= 0:
        raise ValueError('No writable candidate')
    d = torch.as_tensor(decoder[candidates], dtype=torch.float32, device=device)
    z = torch.as_tensor(target_codes[:, candidates], dtype=torch.float32, device=device)
    residual = torch.as_tensor(vectors, dtype=torch.float32, device=device).clone()
    norms = torch.sum(d*d, dim=1).clamp_min(1e-12)
    used = torch.zeros_like(z, dtype=torch.bool)
    selected = []
    batch = torch.arange(len(z), device=device)
    for _ in range(limit):
        inner = residual @ d.T
        amplitude = torch.maximum(inner / norms[None, :], -z)
        improvement = 2*amplitude*inner - amplitude.square()*norms[None, :]
        improvement[used] = -torch.inf
        index = torch.argmax(improvement, dim=1)
        amount = amplitude[batch, index]
        residual -= amount[:, None] * d[index]
        used[batch, index] = True
        selected.append(index.cpu().numpy())
    return candidates[np.stack(selected, axis=1)]
