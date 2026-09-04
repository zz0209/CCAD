"""Numerical primitives for query-conditioned causal subspace transport.

The functions in this module are split-agnostic. R011-S1 callers are responsible
for fitting means on the independent mean split, bases on discovery, and evaluating
only frozen objects on calibration/audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class WeightedSupport:
    indices: np.ndarray
    weights: np.ndarray
    active_count: int
    effective_sample_size: float
    retained_weight_fraction: float


@dataclass(frozen=True)
class TransferMetrics:
    source_energy: float
    target_energy: float
    cross_energy: float
    residual_energy: float
    normalized_residual: float | None
    bcc: float | None
    source_effect_fraction: float | None
    target_effect_fraction: float | None


def stable_seed(*parts: object) -> int:
    """Return a platform-stable 63-bit seed for randomized linear algebra."""

    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def select_weighted_support(code: np.ndarray, max_rows: int, power: float = 2.0) -> WeightedSupport:
    """Select the largest source-only code weights with deterministic index ties."""

    values = np.asarray(code, dtype=np.float64).reshape(-1)
    if max_rows <= 0 or power <= 0:
        raise ValueError("max_rows and power must be positive")
    raw = np.abs(values) ** power
    active = np.flatnonzero(raw > 0)
    if active.size == 0:
        return WeightedSupport(
            indices=np.empty(0, dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            active_count=0,
            effective_sample_size=0.0,
            retained_weight_fraction=0.0,
        )
    order = np.lexsort((active, -raw[active]))
    chosen = active[order[:max_rows]].astype(np.int64, copy=False)
    selected = raw[chosen]
    selected_total = float(np.sum(selected))
    weights = selected / selected_total
    ess = 1.0 / float(np.sum(weights * weights))
    return WeightedSupport(
        indices=chosen,
        weights=weights,
        active_count=int(active.size),
        effective_sample_size=ess,
        retained_weight_fraction=selected_total / float(np.sum(raw[active])),
    )


def weighted_mean(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x, w = _samples_and_weights(samples, weights)
    return w @ x


def weighted_total_energy(samples: np.ndarray, weights: np.ndarray, mean: np.ndarray) -> float:
    x, w = _samples_and_weights(samples, weights)
    centered = x - _mean_vector(mean, x.shape[1])
    return float(np.sum(w[:, None] * centered * centered))


def fit_weighted_pca(
    samples: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    max_rank: int,
    *,
    random_seed: int,
    oversample: int = 8,
    power_iterations: int = 1,
    relative_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic randomized eigenspace of a weighted second moment."""

    x, w = _samples_and_weights(samples, weights)
    center = _mean_vector(mean, x.shape[1])
    if max_rank <= 0 or max_rank > x.shape[1]:
        raise ValueError("max_rank must be in [1, hidden_size]")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power_iterations must be nonnegative")
    xw = (x - center) * np.sqrt(w[:, None])
    sketch_width = min(x.shape[1], max_rank + oversample)
    rng = np.random.default_rng(random_seed)
    omega = rng.standard_normal((x.shape[1], sketch_width))
    q, _ = np.linalg.qr(xw.T @ (xw @ omega), mode="reduced")
    for _ in range(power_iterations):
        q, _ = np.linalg.qr(xw.T @ (xw @ q), mode="reduced")
    small = xw @ q
    gram = small.T @ small
    covariance = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(0.0, values[order])
    vectors = vectors[:, order]
    if values.size == 0 or values[0] <= 0:
        return np.empty((x.shape[1], 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    keep = min(max_rank, int(np.sum(values > relative_tolerance * values[0])))
    basis, _ = np.linalg.qr(q @ vectors[:, :keep], mode="reduced")
    rayleigh = np.einsum("ij,ij->j", xw @ basis, xw @ basis)
    reorder = np.argsort(rayleigh)[::-1]
    return basis[:, reorder], rayleigh[reorder]


def fit_weighted_stitching(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    weights: np.ndarray,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    max_rank: int,
    *,
    random_seed: int,
    oversample: int = 8,
    power_iterations: int = 1,
    relative_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit paired singular subspaces of a weighted cross moment.

    This is the relaxed stitching/MAS-style control. Unlike primary SCT
    projectors it jointly consumes source and target discovery data.
    """

    xs, w = _samples_and_weights(source_samples, weights)
    xt, wt = _samples_and_weights(target_samples, weights)
    if xs.shape != xt.shape or not np.array_equal(w, wt):
        raise ValueError("paired samples and weights must be aligned")
    if max_rank <= 0 or max_rank > xs.shape[1]:
        raise ValueError("max_rank must be in [1, hidden_size]")
    sw = (xs - _mean_vector(source_mean, xs.shape[1])) * np.sqrt(w[:, None])
    tw = (xt - _mean_vector(target_mean, xt.shape[1])) * np.sqrt(w[:, None])
    width = min(xs.shape[1], max_rank + oversample)
    rng = np.random.default_rng(random_seed)
    omega = rng.standard_normal((xs.shape[1], width))

    def c_right(matrix: np.ndarray) -> np.ndarray:
        return sw.T @ (tw @ matrix)

    def ct_right(matrix: np.ndarray) -> np.ndarray:
        return tw.T @ (sw @ matrix)

    left, _ = np.linalg.qr(c_right(omega), mode="reduced")
    for _ in range(power_iterations):
        right, _ = np.linalg.qr(ct_right(left), mode="reduced")
        left, _ = np.linalg.qr(c_right(right), mode="reduced")
    right, _ = np.linalg.qr(ct_right(left), mode="reduced")
    small = left.T @ c_right(right)
    u, singular, vt = np.linalg.svd(small, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0:
        empty = np.empty((xs.shape[1], 0), dtype=np.float64)
        return empty, empty.copy(), np.empty(0, dtype=np.float64)
    keep = min(max_rank, int(np.sum(singular > relative_tolerance * singular[0])))
    left_basis, _ = np.linalg.qr(left @ u[:, :keep], mode="reduced")
    right_basis, _ = np.linalg.qr(right @ vt.T[:, :keep], mode="reduced")
    return left_basis, right_basis, singular[:keep]


def random_orthonormal_basis(hidden_size: int, rank: int, random_seed: int) -> np.ndarray:
    if hidden_size <= 0 or rank <= 0 or rank > hidden_size:
        raise ValueError("rank must be in [1, hidden_size]")
    rng = np.random.default_rng(random_seed)
    q, _ = np.linalg.qr(rng.standard_normal((hidden_size, rank)), mode="reduced")
    return q


def projector_subspace_similarity(left_basis: np.ndarray, right_basis: np.ndarray) -> dict[str, object]:
    left = _basis(left_basis)
    right = _basis(right_basis)
    if left.shape[1] == 0 or right.shape[1] == 0:
        return {
            "psc": None, "projector_distance_sq": None,
            "rank_left": left.shape[1], "rank_right": right.shape[1], "principal_cosines": [],
        }
    cosines = np.linalg.svd(left.T @ right, compute_uv=False)
    overlap = float(np.sum(cosines * cosines))
    distance = float(left.shape[1] + right.shape[1] - 2.0 * overlap)
    return {
        "psc": overlap / max(left.shape[1], right.shape[1]),
        "projector_distance_sq": max(0.0, distance),
        "rank_left": left.shape[1], "rank_right": right.shape[1],
        "principal_cosines": cosines.tolist(),
    }


def projected_dynamic(samples: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    b = _basis(basis)
    if x.ndim != 2 or x.shape[1] != b.shape[0]:
        raise ValueError("samples and basis dimensions differ")
    centered = x - _mean_vector(mean, x.shape[1])
    return (centered @ b) @ b.T


def transfer_metrics(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    weights: np.ndarray,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    source_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    energy_epsilon: float = 1e-12,
) -> TransferMetrics:
    source, w = _samples_and_weights(source_samples, weights)
    target, wt = _samples_and_weights(target_samples, weights)
    if source.shape != target.shape or not np.array_equal(w, wt):
        raise ValueError("source/target processes must be aligned")
    source_projected = projected_dynamic(source, source_mean, source_basis)
    target_projected = projected_dynamic(target, target_mean, target_basis)
    source_energy = float(np.sum(w[:, None] * source_projected * source_projected))
    target_energy = float(np.sum(w[:, None] * target_projected * target_projected))
    cross = float(np.sum(w[:, None] * source_projected * target_projected))
    residual = max(0.0, source_energy + target_energy - 2.0 * cross)
    denominator = source_energy + target_energy
    source_total = weighted_total_energy(source, w, source_mean)
    target_total = weighted_total_energy(target, w, target_mean)
    return TransferMetrics(
        source_energy=source_energy, target_energy=target_energy, cross_energy=cross,
        residual_energy=residual,
        normalized_residual=residual / max(source_energy, energy_epsilon) if source_energy > energy_epsilon else None,
        bcc=2.0 * cross / denominator if denominator > energy_epsilon else None,
        source_effect_fraction=source_energy / source_total if source_total > energy_epsilon else None,
        target_effect_fraction=target_energy / target_total if target_total > energy_epsilon else None,
    )


def mean_transfer_metrics(source_mean: np.ndarray, target_mean: np.ndarray, energy_epsilon: float = 1e-12) -> dict[str, float | None]:
    left = np.asarray(source_mean, dtype=np.float64).reshape(-1)
    right = np.asarray(target_mean, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("means must have the same shape")
    residual = float(np.sum((left - right) ** 2))
    source_energy = float(left @ left)
    target_energy = float(right @ right)
    cross = float(left @ right)
    return {
        "source_mean_energy": source_energy, "target_mean_energy": target_energy,
        "mean_cross_energy": cross, "mean_residual": residual,
        "normalized_mean_residual": residual / source_energy if source_energy > energy_epsilon else None,
        "mean_bcc": 2.0 * cross / (source_energy + target_energy) if source_energy + target_energy > energy_epsilon else None,
    }


def subspace_ablation(hook: np.ndarray, contribution_samples: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    state = np.asarray(hook)
    contribution = projected_dynamic(contribution_samples, mean, basis)
    if state.shape != contribution.shape:
        raise ValueError("hook and projected contribution shapes differ")
    return state - contribution.astype(state.dtype, copy=False)


def _samples_and_weights(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != w.size or x.shape[0] == 0:
        raise ValueError("samples must be nonempty [n,d] and align with weights")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("samples/weights must be finite and weights nonnegative")
    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("weights must have positive mass")
    return x, w / total


def _mean_vector(mean: np.ndarray, hidden_size: int) -> np.ndarray:
    value = np.asarray(mean, dtype=np.float64).reshape(-1)
    if value.shape != (hidden_size,) or not np.isfinite(value).all():
        raise ValueError("mean must be a finite hidden-size vector")
    return value


def _basis(basis: np.ndarray) -> np.ndarray:
    value = np.asarray(basis, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("basis must be a finite matrix")
    if value.shape[1] and not np.allclose(value.T @ value, np.eye(value.shape[1]), atol=1e-8, rtol=1e-8):
        raise ValueError("basis columns must be orthonormal")
    return value
