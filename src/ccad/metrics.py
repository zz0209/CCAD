"""Numerically explicit CBSM metrics used by the synthetic conformance suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BCCResult:
    value: float | None
    normalized_residual: float | None
    cross_inner: float
    energy_left: float
    energy_right: float
    status: str


@dataclass(frozen=True)
class PSCResult:
    value: float | None
    rank_left: int
    rank_right: int
    projector_distance_sq: float | None
    status: str


@dataclass(frozen=True)
class CancellationResult:
    cancellation_energy_ratio: float | None
    max_leave_one_out_energy_ratio: float | None
    per_feature_energy_ratios: tuple[float, ...]
    aggregate_energy: float
    status: str


@dataclass(frozen=True)
class OccupancyResult:
    active_token_count: int
    active_document_count: int
    token_energy_kish_ess: float
    document_energy_kish_ess: float
    total_token_count: int
    total_document_count: int


@dataclass(frozen=True)
class BootstrapBCCResult:
    values: tuple[float, ...]
    inactive_replicates: int
    inactive_fraction: float
    ci_lower: float | None
    ci_upper: float | None
    ci_width: float | None
    replicates: int


def center_codes(z_mean: np.ndarray, z_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center evaluation codes using constants estimated only on the mean split."""
    if z_mean.ndim != 2 or z_eval.ndim != 2 or z_mean.shape[1] != z_eval.shape[1]:
        raise ValueError("mean and evaluation codes must be rank-2 with matching feature count")
    constants = np.mean(z_mean, axis=0, keepdims=True)
    return z_eval - constants, constants.squeeze(0)


def contribution_kernel(d_left: np.ndarray, z_left: np.ndarray, d_right: np.ndarray, z_right: np.ndarray) -> np.ndarray:
    """Compute the paired contribution Gram kernel without materializing n*d processes."""
    if d_left.ndim != 2 or d_right.ndim != 2 or z_left.ndim != 2 or z_right.ndim != 2:
        raise ValueError("decoder and code arrays must all be rank-2")
    if d_left.shape[0] != d_right.shape[0]:
        raise ValueError("decoders must share hook dimension")
    if z_left.shape[0] != z_right.shape[0]:
        raise ValueError("codes must share paired observation count")
    if d_left.shape[1] != z_left.shape[1] or d_right.shape[1] != z_right.shape[1]:
        raise ValueError("decoder feature count must match code feature count")
    n = z_left.shape[0]
    if n == 0:
        raise ValueError("at least one paired observation is required")
    return (d_left.T @ d_right) * ((z_left.T @ z_right) / n)


def group_inner(kernel: np.ndarray, left_ids: np.ndarray, right_ids: np.ndarray) -> float:
    return float(np.sum(kernel[np.ix_(left_ids, right_ids)]))


def absolute_code_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Absolute Pearson correlation for two paired one-dimensional code series."""
    left_vector = np.asarray(left, dtype=np.float64).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_vector.shape != right_vector.shape or left_vector.size == 0:
        raise ValueError("code series must have the same nonzero length")
    left_centered = left_vector - np.mean(left_vector)
    right_centered = right_vector - np.mean(right_vector)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        raise ValueError("code correlation is undefined for a constant series")
    return abs(float(left_centered @ right_centered) / denominator)


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand index for two hard partitions without external dependencies."""
    truth = np.asarray(labels_true).reshape(-1)
    prediction = np.asarray(labels_pred).reshape(-1)
    if truth.shape != prediction.shape or truth.size < 2:
        raise ValueError("partitions must have the same length of at least two")
    true_values, true_inverse = np.unique(truth, return_inverse=True)
    pred_values, pred_inverse = np.unique(prediction, return_inverse=True)
    contingency = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)

    def choose_two(values: np.ndarray) -> float:
        values = values.astype(np.float64)
        return float(np.sum(values * (values - 1.0) / 2.0))

    sum_cells = choose_two(contingency)
    sum_true = choose_two(np.sum(contingency, axis=1))
    sum_pred = choose_two(np.sum(contingency, axis=0))
    total_pairs = truth.size * (truth.size - 1.0) / 2.0
    expected = sum_true * sum_pred / total_pairs
    maximum = 0.5 * (sum_true + sum_pred)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0 if np.array_equal(true_inverse, pred_inverse) else 0.0
    return (sum_cells - expected) / denominator


def bcc_from_kernels(
    k_ll: np.ndarray,
    k_lr: np.ndarray,
    k_rr: np.ndarray,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    *,
    inactive_atol: float = 1e-15,
) -> BCCResult:
    energy_left = group_inner(k_ll, left_ids, left_ids)
    energy_right = group_inner(k_rr, right_ids, right_ids)
    cross = group_inner(k_lr, left_ids, right_ids)
    denom = energy_left + energy_right
    if denom <= inactive_atol:
        return BCCResult(None, None, cross, energy_left, energy_right, "INACTIVE")
    residual = (energy_left + energy_right - 2.0 * cross) / denom
    return BCCResult(2.0 * cross / denom, residual, cross, energy_left, energy_right, "OK")


def explicit_group_contribution(d: np.ndarray, z: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return z[:, ids] @ d[:, ids].T


def projector_subspace_consistency(
    d_left: np.ndarray,
    d_right: np.ndarray,
    *,
    rank_rtol: float = 1e-12,
) -> PSCResult:
    def projector(d: np.ndarray) -> tuple[np.ndarray, int]:
        if d.ndim != 2:
            raise ValueError("decoder blocks must be rank-2")
        u, singular, _ = np.linalg.svd(d, full_matrices=False)
        if singular.size == 0 or singular[0] == 0:
            rank = 0
        else:
            rank = int(np.sum(singular > rank_rtol * singular[0]))
        basis = u[:, :rank]
        return basis @ basis.T, rank

    p, rank_left = projector(d_left)
    q, rank_right = projector(d_right)
    if rank_left == 0 or rank_right == 0:
        return PSCResult(None, rank_left, rank_right, None, "DEGENERATE_PSC")
    distance_sq = float(np.sum((p - q) ** 2))
    value = 1.0 - distance_sq / (rank_left + rank_right)
    return PSCResult(value, rank_left, rank_right, distance_sq, "OK")


def cancellation_diagnostics(
    d: np.ndarray,
    z_centered: np.ndarray,
    ids: np.ndarray,
    *,
    inactive_atol: float = 1e-15,
) -> CancellationResult:
    """Measure large component energies hidden by an additively canceling group sum."""
    selected = np.asarray(ids, dtype=int)
    if selected.size == 0:
        raise ValueError("cancellation diagnostics require a nonempty group")
    contributions = z_centered[:, selected, None] * d[:, selected].T[None, :, :]
    individual_energy = np.mean(np.sum(contributions ** 2, axis=2), axis=0)
    aggregate = np.sum(contributions, axis=1)
    aggregate_energy = float(np.mean(np.sum(aggregate ** 2, axis=1)))
    if aggregate_energy <= inactive_atol:
        return CancellationResult(None, None, (), aggregate_energy, "INACTIVE")
    ratios = tuple(float(value / aggregate_energy) for value in individual_energy)
    return CancellationResult(
        cancellation_energy_ratio=float(np.sum(individual_energy) / aggregate_energy),
        max_leave_one_out_energy_ratio=max(ratios),
        per_feature_energy_ratios=ratios,
        aggregate_energy=aggregate_energy,
        status="OK",
    )


def occupancy_effective_sample_size(
    d: np.ndarray,
    z_raw: np.ndarray,
    ids: np.ndarray,
    document_ids: np.ndarray,
    *,
    activity_atol: float = 1e-15,
) -> OccupancyResult:
    """Token and document energy ESS for a sparse group, preserving document clustering."""
    selected = np.asarray(ids, dtype=int)
    documents = np.asarray(document_ids).reshape(-1)
    if z_raw.shape[0] != documents.size:
        raise ValueError("document id count must match code observations")
    contribution = explicit_group_contribution(d, z_raw, selected)
    token_energy = np.sum(contribution ** 2, axis=1)
    active = token_energy > activity_atol
    unique_documents = np.unique(documents)
    document_energy = np.asarray([np.sum(token_energy[documents == doc]) for doc in unique_documents])

    def kish(weights: np.ndarray) -> float:
        denominator = float(np.sum(weights ** 2))
        return 0.0 if denominator == 0.0 else float(np.sum(weights) ** 2 / denominator)

    return OccupancyResult(
        active_token_count=int(np.sum(active)),
        active_document_count=int(np.sum(document_energy > activity_atol)),
        token_energy_kish_ess=kish(token_energy),
        document_energy_kish_ess=kish(document_energy),
        total_token_count=z_raw.shape[0],
        total_document_count=len(unique_documents),
    )


def document_bootstrap_bcc(
    d_left: np.ndarray,
    z_left_centered: np.ndarray,
    left_ids: np.ndarray,
    d_right: np.ndarray,
    z_right_centered: np.ndarray,
    right_ids: np.ndarray,
    document_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    inactive_atol: float = 1e-15,
) -> BootstrapBCCResult:
    """Paired document-cluster bootstrap for BCC, retaining inactive resamples explicitly."""
    if replicates < 1:
        raise ValueError("bootstrap requires at least one replicate")
    documents = np.asarray(document_ids).reshape(-1)
    if z_left_centered.shape[0] != documents.size or z_right_centered.shape[0] != documents.size:
        raise ValueError("document id count must match both centered code matrices")
    unique_documents = np.unique(documents)
    indices_by_document = [np.flatnonzero(documents == doc) for doc in unique_documents]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    inactive = 0
    for _ in range(replicates):
        sampled = rng.integers(0, len(unique_documents), size=len(unique_documents))
        indices = np.concatenate([indices_by_document[index] for index in sampled])
        left = explicit_group_contribution(d_left, z_left_centered[indices], np.asarray(left_ids, dtype=int))
        right = explicit_group_contribution(d_right, z_right_centered[indices], np.asarray(right_ids, dtype=int))
        energy_left = float(np.mean(np.sum(left ** 2, axis=1)))
        energy_right = float(np.mean(np.sum(right ** 2, axis=1)))
        denominator = energy_left + energy_right
        if denominator <= inactive_atol:
            inactive += 1
            continue
        cross = float(np.mean(np.sum(left * right, axis=1)))
        values.append(2.0 * cross / denominator)
    if values:
        lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
        ci_lower = float(lower)
        ci_upper = float(upper)
        ci_width = ci_upper - ci_lower
    else:
        ci_lower = ci_upper = ci_width = None
    return BootstrapBCCResult(
        values=tuple(values),
        inactive_replicates=inactive,
        inactive_fraction=inactive / replicates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        replicates=replicates,
    )


def pw_mcc_absolute_cosine(d_left: np.ndarray, d_right: np.ndarray) -> float:
    """Maximum-weight one-to-one absolute-cosine score via a rectangular Hungarian algorithm."""
    left_norm = np.linalg.norm(d_left, axis=0)
    right_norm = np.linalg.norm(d_right, axis=0)
    if np.any(left_norm == 0) or np.any(right_norm == 0):
        raise ValueError("PW-MCC excludes zero-norm decoder columns")
    similarity = np.abs((d_left.T @ d_right) / np.outer(left_norm, right_norm))
    if similarity.shape[0] > similarity.shape[1]:
        similarity = similarity.T
    n, m = similarity.shape
    cost = -similarity
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    matched = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return float(np.mean([similarity[i, j] for i, j in matched]))
