"""Linear-algebra primitives for fuzzy many-to-many concept correspondence.

The object fitted here is a paired low-rank relation between two banks of
already-centered, token-dependent hook-space contribution processes.  It is
not a one-to-one assignment and it is not a claim that either SAE contains a
canonical concept atom.  Discovery/calibration/audit separation, source-only
query construction, and causal validation remain responsibilities of callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeMetric:
    matrix: np.ndarray
    factor: np.ndarray
    rank: int
    explained_trace_fraction: float


@dataclass(frozen=True)
class FuzzyCorrespondence:
    source_loadings: np.ndarray
    target_loadings: np.ndarray
    canonical_values: np.ndarray
    full_canonical_values: np.ndarray
    rank_boundary_relative_gap: float | None
    cross_operator: np.ndarray
    coupling: np.ndarray
    source_membership: np.ndarray
    target_membership: np.ndarray
    source_effective_support: float
    target_effective_support: float


@dataclass(frozen=True)
class FixedRelationMetrics:
    source_energy: float
    target_energy: float
    cross_energy: float
    normalized_residual: float | None
    bcc: float | None


@dataclass(frozen=True)
class ContributionKernels:
    """Feature-space kernels without materializing token-feature-hook tensors."""

    source_gram: np.ndarray
    target_gram: np.ndarray
    cross_gram: np.ndarray
    negative_source_gram: np.ndarray | None = None
    negative_target_gram: np.ndarray | None = None
    negative_cross_gram: np.ndarray | None = None


def fit_probe_metric(
    probe_directions: np.ndarray,
    output_effects: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    ridge_fraction: float = 1e-6,
    relative_tolerance: float = 1e-8,
) -> ProbeMetric:
    """Estimate a PSD downstream-sensitivity metric from frozen probes.

    Ridge regression estimates the local linear map from hook perturbations to
    an output sketch.  Its pullback Gram matrix is the causal seminorm.  The
    matrix is trace-normalized, so later thresholds do not depend on the output
    sketch's arbitrary scale.
    """

    directions = _matrix(probe_directions, "probe_directions")
    effects = _matrix(output_effects, "output_effects")
    if directions.shape[0] != effects.shape[0]:
        raise ValueError("probe directions and output effects must share rows")
    if ridge_fraction < 0 or relative_tolerance <= 0:
        raise ValueError("ridge_fraction must be nonnegative and tolerance positive")
    sample_weights = _weights(weights, directions.shape[0])
    root = np.sqrt(sample_weights)[:, None]
    x = directions * root
    y = effects * root
    gram = x.T @ x
    scale = float(np.trace(gram)) / directions.shape[1]
    ridge = ridge_fraction * max(scale, np.finfo(np.float64).eps)
    linear_map = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), x.T @ y)
    metric = linear_map @ linear_map.T
    metric = 0.5 * (metric + metric.T)
    trace = float(np.trace(metric))
    if trace <= np.finfo(np.float64).eps:
        return ProbeMetric(
            matrix=np.zeros_like(metric), factor=np.empty((metric.shape[0], 0)),
            rank=0, explained_trace_fraction=0.0,
        )
    metric *= metric.shape[0] / trace
    factor, retained = metric_factor(metric, relative_tolerance=relative_tolerance)
    return ProbeMetric(
        matrix=metric,
        factor=factor,
        rank=factor.shape[1],
        explained_trace_fraction=retained,
    )


def fit_crossed_probe_metric(
    shared_probe_directions: np.ndarray,
    state_output_effects: np.ndarray,
    *,
    state_weights: np.ndarray | None = None,
    ridge_fraction: float = 1e-6,
    relative_tolerance: float = 1e-8,
) -> ProbeMetric:
    """Estimate ``E_x[J_x.T J_x]`` from a crossed shared-direction design.

    ``state_output_effects[x, j]`` is the output derivative at state ``x``
    along shared hook direction ``j``. Concatenating equally weighted state
    outputs turns the regression target into a stacked Jacobian, so its
    pullback Gram is the average squared downstream sensitivity rather than
    the Gram of an average Jacobian.
    """

    directions = _matrix(shared_probe_directions, "shared_probe_directions")
    effects = np.asarray(state_output_effects, dtype=np.float64)
    if effects.ndim != 3:
        raise ValueError("state_output_effects must be [state, direction, output]")
    if effects.shape[1] != directions.shape[0]:
        raise ValueError("every state must use the same complete direction rows")
    if ridge_fraction < 0 or relative_tolerance <= 0:
        raise ValueError("ridge_fraction must be nonnegative and tolerance positive")
    weights = _weights(state_weights, effects.shape[0])
    weights = weights / np.sum(weights)
    stacked = np.concatenate(
        [np.sqrt(weights[index]) * effects[index] for index in range(effects.shape[0])],
        axis=1,
    )
    gram = directions.T @ directions
    scale = float(np.trace(gram)) / directions.shape[1]
    ridge = ridge_fraction * max(scale, np.finfo(np.float64).eps)
    stacked_jacobian = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), directions.T @ stacked,
    )
    metric = stacked_jacobian @ stacked_jacobian.T
    metric = 0.5 * (metric + metric.T)
    trace = float(np.trace(metric))
    if trace <= np.finfo(np.float64).eps:
        return ProbeMetric(
            matrix=np.zeros_like(metric), factor=np.empty((metric.shape[0], 0)),
            rank=0, explained_trace_fraction=0.0,
        )
    metric *= metric.shape[0] / trace
    factor, retained = metric_factor(metric, relative_tolerance=relative_tolerance)
    return ProbeMetric(
        matrix=metric,
        factor=factor,
        rank=factor.shape[1],
        explained_trace_fraction=retained,
    )


def metric_factor(
    metric: np.ndarray,
    *,
    relative_tolerance: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Return ``L`` with ``M = L L.T`` after numerical rank truncation."""

    matrix = _matrix(metric, "metric")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("metric must be square")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if values.size == 0 or values[-1] <= 0:
        return np.empty((matrix.shape[0], 0)), 0.0
    if values[0] < -relative_tolerance * values[-1]:
        raise ValueError("metric must be positive semidefinite")
    values = np.maximum(values, 0.0)
    keep = values > relative_tolerance * values[-1]
    retained = float(np.sum(values[keep]) / np.sum(values))
    return vectors[:, keep] * np.sqrt(values[keep]), retained


def fit_fuzzy_correspondence(
    source_contributions: np.ndarray,
    target_contributions: np.ndarray,
    positive_weights: np.ndarray,
    *,
    negative_weights: np.ndarray | None = None,
    metric: np.ndarray | None = None,
    rank: int = 1,
    contrast_strength: float = 0.0,
    ridge_fraction: float = 1e-6,
) -> FuzzyCorrespondence:
    """Fit a query-contrastive low-rank relation between contribution banks.

    Inputs have shape ``[observations, features, hook_dim]`` and must already
    use means frozen on an independent split.  Positive and hard-negative
    weights are source-query-defined.  The returned cross operator and soft
    coupling permit overlapping one-to-many and many-to-one correspondences.
    """

    source = _contribution_bank(source_contributions, "source_contributions")
    target = _contribution_bank(target_contributions, "target_contributions")
    if source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError("source and target banks must share observations and hook_dim")
    if rank <= 0 or rank > min(source.shape[1], target.shape[1]):
        raise ValueError("rank exceeds the available feature dimensions")
    if contrast_strength < 0 or ridge_fraction <= 0:
        raise ValueError("contrast_strength must be nonnegative and ridge positive")
    positive = _weights(positive_weights, source.shape[0])
    negative = None if negative_weights is None else _weights(negative_weights, source.shape[0])
    if contrast_strength > 0 and negative is None:
        raise ValueError("contrastive fitting requires negative_weights")

    if metric is None:
        factor = np.eye(source.shape[2])
    else:
        factor, _ = metric_factor(metric)
        if factor.shape[0] != source.shape[2]:
            raise ValueError("metric and hook dimensions differ")
        if factor.shape[1] == 0:
            raise ValueError("metric has zero numerical rank")

    source_metric = np.einsum("npd,dk->npk", source, factor, optimize=True)
    target_metric = np.einsum("nqd,dk->nqk", target, factor, optimize=True)
    source_pos = _weighted_feature_matrix(source_metric, positive)
    target_pos = _weighted_feature_matrix(target_metric, positive)
    source_gram = source_pos.T @ source_pos
    target_gram = target_pos.T @ target_pos
    cross = source_pos.T @ target_pos
    if negative is not None and contrast_strength > 0:
        source_neg = _weighted_feature_matrix(source_metric, negative)
        target_neg = _weighted_feature_matrix(target_metric, negative)
        source_gram += contrast_strength * (source_neg.T @ source_neg)
        target_gram += contrast_strength * (target_neg.T @ target_neg)
        cross -= contrast_strength * (source_neg.T @ target_neg)

    return fit_fuzzy_correspondence_from_kernels(
        ContributionKernels(source_gram, target_gram, cross),
        rank=rank,
        ridge_fraction=ridge_fraction,
    )


def sparse_contribution_kernels(
    source_codes,
    target_codes,
    source_decoders: np.ndarray,
    target_decoders: np.ndarray,
    positive_weights: np.ndarray,
    *,
    source_mean_codes: np.ndarray,
    target_mean_codes: np.ndarray,
    negative_weights: np.ndarray | None = None,
    metric: np.ndarray | None = None,
) -> ContributionKernels:
    """Build contribution kernels directly from sparse codes and decoders.

    The code means must come from the independent mean split.  Centering is
    performed algebraically, so sparse code matrices remain sparse and the
    dense ``[tokens, features, hook_dim]`` contribution bank is never built.
    Callers should pass a discovery-frozen local feature universe; this helper
    intentionally does not perform candidate selection.
    """

    source_decoder = _matrix(source_decoders, "source_decoders")
    target_decoder = _matrix(target_decoders, "target_decoders")
    if source_decoder.shape[1] != target_decoder.shape[1]:
        raise ValueError("source and target decoders must share hook_dim")
    source_shape = _code_shape(source_codes, "source_codes")
    target_shape = _code_shape(target_codes, "target_codes")
    if source_shape[0] != target_shape[0]:
        raise ValueError("source and target codes must share observations")
    if source_shape[1] != source_decoder.shape[0] or target_shape[1] != target_decoder.shape[0]:
        raise ValueError("code and decoder feature dimensions differ")
    source_mean = _vector(source_mean_codes, "source_mean_codes", source_shape[1])
    target_mean = _vector(target_mean_codes, "target_mean_codes", target_shape[1])
    positive = _weights(positive_weights, source_shape[0])

    if metric is None:
        source_metric_decoder = source_decoder
        target_metric_decoder = target_decoder
    else:
        factor, _ = metric_factor(metric)
        if factor.shape[0] != source_decoder.shape[1] or factor.shape[1] == 0:
            raise ValueError("metric and hook dimensions differ or metric rank is zero")
        source_metric_decoder = source_decoder @ factor
        target_metric_decoder = target_decoder @ factor

    positive_kernels = _kernel_triplet(
        source_codes, target_codes, source_metric_decoder, target_metric_decoder,
        positive, source_mean, target_mean,
    )
    if negative_weights is None:
        return ContributionKernels(*positive_kernels)
    negative = _weights(negative_weights, source_shape[0])
    negative_kernels = _kernel_triplet(
        source_codes, target_codes, source_metric_decoder, target_metric_decoder,
        negative, source_mean, target_mean,
    )
    return ContributionKernels(*positive_kernels, *negative_kernels)


def fit_fuzzy_correspondence_from_kernels(
    kernels: ContributionKernels,
    *,
    rank: int = 1,
    contrast_strength: float = 0.0,
    ridge_fraction: float = 1e-6,
) -> FuzzyCorrespondence:
    """Fit FCC from frozen contribution Gram and cross-kernel matrices."""

    source_gram = _square(kernels.source_gram, "source_gram")
    target_gram = _square(kernels.target_gram, "target_gram")
    cross = _matrix(kernels.cross_gram, "cross_gram")
    if cross.shape != (source_gram.shape[0], target_gram.shape[0]):
        raise ValueError("cross kernel dimensions differ from marginal kernels")
    if rank <= 0 or rank > min(cross.shape):
        raise ValueError("rank exceeds the available feature dimensions")
    if contrast_strength < 0 or ridge_fraction <= 0:
        raise ValueError("contrast_strength must be nonnegative and ridge positive")
    negative_parts = (
        kernels.negative_source_gram,
        kernels.negative_target_gram,
        kernels.negative_cross_gram,
    )
    if contrast_strength > 0:
        if any(part is None for part in negative_parts):
            raise ValueError("contrastive fitting requires negative kernels")
        negative_source = _square(negative_parts[0], "negative_source_gram")
        negative_target = _square(negative_parts[1], "negative_target_gram")
        negative_cross = _matrix(negative_parts[2], "negative_cross_gram")
        if negative_source.shape != source_gram.shape or negative_target.shape != target_gram.shape or negative_cross.shape != cross.shape:
            raise ValueError("negative and positive kernel dimensions differ")
        source_gram = source_gram + contrast_strength * negative_source
        target_gram = target_gram + contrast_strength * negative_target
        cross = cross - contrast_strength * negative_cross

    source_whitener = _inverse_sqrt(source_gram, ridge_fraction)
    target_whitener = _inverse_sqrt(target_gram, ridge_fraction)
    whitened_cross = source_whitener @ cross @ target_whitener
    left, singular, right_t = np.linalg.svd(whitened_cross, full_matrices=False)
    keep = min(rank, singular.size)
    source_loadings = source_whitener @ left[:, :keep]
    target_loadings = target_whitener @ right_t.T[:, :keep]
    canonical = singular[:keep]
    cross_operator = source_loadings @ np.diag(canonical) @ target_loadings.T
    magnitude = np.abs(cross_operator)
    total = float(np.sum(magnitude))
    coupling = magnitude / total if total > 0 else np.zeros_like(magnitude)
    source_membership = np.sum(coupling, axis=1)
    target_membership = np.sum(coupling, axis=0)
    return FuzzyCorrespondence(
        source_loadings=source_loadings,
        target_loadings=target_loadings,
        canonical_values=canonical,
        full_canonical_values=singular,
        rank_boundary_relative_gap=(
            float((singular[keep - 1] - singular[keep]) / max(singular[0], np.finfo(np.float64).eps))
            if keep < singular.size else None
        ),
        cross_operator=cross_operator,
        coupling=coupling,
        source_membership=source_membership,
        target_membership=target_membership,
        source_effective_support=_effective_support(source_membership),
        target_effective_support=_effective_support(target_membership),
    )


def soft_membership_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Bhattacharyya overlap for auditing cross-query correspondence collision."""

    first = _probability_vector(left, "left")
    second = _probability_vector(right, "right")
    if first.shape != second.shape:
        raise ValueError("membership vectors must have the same shape")
    return float(np.sum(np.sqrt(first * second)))


def evaluate_fixed_correspondence(
    source_contributions: np.ndarray,
    target_contributions: np.ndarray,
    weights: np.ndarray,
    source_loadings: np.ndarray,
    target_loadings: np.ndarray,
    *,
    metric: np.ndarray | None = None,
    energy_epsilon: float = 1e-12,
) -> FixedRelationMetrics:
    """Evaluate discovery-frozen paired loadings on another weighted sample."""

    source = _contribution_bank(source_contributions, "source_contributions")
    target = _contribution_bank(target_contributions, "target_contributions")
    if source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError("source and target banks must share observations and hook_dim")
    sample_weights = _weights(weights, source.shape[0])
    if metric is None:
        factor = np.eye(source.shape[2])
    else:
        factor, _ = metric_factor(metric)
        if factor.shape[0] != source.shape[2] or factor.shape[1] == 0:
            raise ValueError("metric and hook dimensions differ or metric rank is zero")
    source_metric = np.einsum("npd,dk->npk", source, factor, optimize=True)
    target_metric = np.einsum("nqd,dk->nqk", target, factor, optimize=True)
    source_matrix = _weighted_feature_matrix(source_metric, sample_weights)
    target_matrix = _weighted_feature_matrix(target_metric, sample_weights)
    left = _matrix(source_loadings, "source_loadings")
    right = _matrix(target_loadings, "target_loadings")
    if left.shape[0] != source.shape[1] or right.shape[0] != target.shape[1] or left.shape[1] != right.shape[1]:
        raise ValueError("loading and contribution-bank dimensions differ")
    source_latent = source_matrix @ left
    target_latent = target_matrix @ right
    source_energy = float(np.sum(source_latent * source_latent))
    target_energy = float(np.sum(target_latent * target_latent))
    cross_energy = float(np.sum(source_latent * target_latent))
    denominator = source_energy + target_energy
    residual = max(0.0, denominator - 2.0 * cross_energy)
    return FixedRelationMetrics(
        source_energy=source_energy,
        target_energy=target_energy,
        cross_energy=cross_energy,
        normalized_residual=(residual / source_energy if source_energy > energy_epsilon else None),
        bcc=(2.0 * cross_energy / denominator if denominator > energy_epsilon else None),
    )


def _weighted_feature_matrix(bank: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = bank * np.sqrt(weights)[:, None, None]
    return weighted.transpose(0, 2, 1).reshape(-1, bank.shape[1])


def _kernel_triplet(
    source_codes,
    target_codes,
    source_decoders: np.ndarray,
    target_decoders: np.ndarray,
    weights: np.ndarray,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_cov = _centered_code_cross(
        source_codes, source_codes, weights, source_mean, source_mean,
    )
    target_cov = _centered_code_cross(
        target_codes, target_codes, weights, target_mean, target_mean,
    )
    cross_cov = _centered_code_cross(
        source_codes, target_codes, weights, source_mean, target_mean,
    )
    source_decoder_gram = source_decoders @ source_decoders.T
    target_decoder_gram = target_decoders @ target_decoders.T
    decoder_cross = source_decoders @ target_decoders.T
    source_kernel = source_cov * source_decoder_gram
    target_kernel = target_cov * target_decoder_gram
    cross_kernel = cross_cov * decoder_cross
    return (
        0.5 * (source_kernel + source_kernel.T),
        0.5 * (target_kernel + target_kernel.T),
        cross_kernel,
    )


def _centered_code_cross(
    left_codes,
    right_codes,
    weights: np.ndarray,
    left_mean: np.ndarray,
    right_mean: np.ndarray,
) -> np.ndarray:
    weighted_right = right_codes.multiply(weights[:, None]) if hasattr(right_codes, "multiply") else np.asarray(right_codes, dtype=np.float64) * weights[:, None]
    raw = left_codes.T @ weighted_right
    raw_array = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw, dtype=np.float64)
    left_weighted = np.asarray(left_codes.T @ weights, dtype=np.float64).reshape(-1)
    right_weighted = np.asarray(right_codes.T @ weights, dtype=np.float64).reshape(-1)
    return (
        raw_array
        - np.outer(left_weighted, right_mean)
        - np.outer(left_mean, right_weighted)
        + np.outer(left_mean, right_mean) * float(np.sum(weights))
    )


def _inverse_sqrt(gram: np.ndarray, ridge_fraction: float) -> np.ndarray:
    matrix = 0.5 * (gram + gram.T)
    scale = float(np.trace(matrix)) / matrix.shape[0]
    ridge = ridge_fraction * max(scale, np.finfo(np.float64).eps)
    values, vectors = np.linalg.eigh(matrix + ridge * np.eye(matrix.shape[0]))
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _effective_support(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    denominator = float(np.sum(values * values))
    return 0.0 if denominator == 0 else 1.0 / denominator


def _probability_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(vector))
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return vector / total


def _weights(values: np.ndarray | None, rows: int) -> np.ndarray:
    if values is None:
        return np.full(rows, 1.0 / rows)
    weights = _probability_vector(values, "weights")
    if weights.size != rows:
        raise ValueError("weights and observations differ")
    return weights


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix")
    return matrix


def _square(values: np.ndarray, name: str) -> np.ndarray:
    matrix = _matrix(values, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    return 0.5 * (matrix + matrix.T)


def _vector(values: np.ndarray, name: str, size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with the feature dimension")
    return vector


def _code_shape(values, name: str) -> tuple[int, int]:
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a two-dimensional code matrix")
    if not hasattr(values, "multiply"):
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    elif hasattr(values, "data") and not np.all(np.isfinite(values.data)):
        raise ValueError(f"{name} must be finite")
    return int(shape[0]), int(shape[1])


def _contribution_bank(values: np.ndarray, name: str) -> np.ndarray:
    bank = np.asarray(values, dtype=np.float64)
    if bank.ndim != 3 or not np.all(np.isfinite(bank)):
        raise ValueError(f"{name} must be a finite [rows, features, hook] array")
    return bank
