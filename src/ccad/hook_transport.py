"""Query-conditioned reduced-rank transport in the shared hook space.

The source process and target process must already be centered with constants
from the independent mean split.  Fitting is discovery-only; this module does
not select queries, tune on calibration, or access audit data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HookTransport:
    target_factors: np.ndarray
    source_factors: np.ndarray
    full_singular_values: np.ndarray
    requested_rank: int
    effective_rank: int
    rank_boundary_relative_gap: float | None
    ridge: float
    status: str

    def predict_components(self, target_process: np.ndarray) -> np.ndarray:
        """Return paired transported components as ``[rank, observation, hook]``."""

        target = _matrix(target_process, "target_process")
        if target.shape[1] != self.target_factors.shape[0]:
            raise ValueError("target hook dimension differs from fitted transport")
        scores = target @ self.target_factors
        return np.einsum("nr,dr->rnd", scores, self.source_factors, optimize=True)

    def predict(self, target_process: np.ndarray) -> np.ndarray:
        """Return the aggregate transported source process."""

        components = self.predict_components(target_process)
        return np.sum(components, axis=0)


@dataclass(frozen=True)
class TransportMetrics:
    source_energy: float
    transported_energy: float
    cross_energy: float
    bcc: float | None
    normalized_residual: float | None


@dataclass(frozen=True)
class TransportGate:
    decision: str
    reason: str | None
    specificity: float | None
    best_control_specificity: float | None


@dataclass(frozen=True)
class NuisanceProjector:
    basis: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    explained_variance_fraction: float
    status: str


def fit_hook_space_transport(
    target_process: np.ndarray,
    source_process: np.ndarray,
    weights: np.ndarray,
    *,
    rank: int,
    ridge_fraction: float = 1e-3,
    rank_relative_tolerance: float = 1e-10,
) -> HookTransport:
    """Fit an asymmetric ridge reduced-rank map from target to source hook process.

    The dual solve scales with the number of selected observations rather than
    the hook dimension.  Rank reduction is applied to the weighted fitted
    source process, which is the standard reduced-rank-regression solution.
    """

    target = _matrix(target_process, "target_process")
    source = _matrix(source_process, "source_process")
    if target.shape[0] != source.shape[0]:
        raise ValueError("source and target processes must share observations")
    if rank <= 0 or rank > min(target.shape[1], source.shape[1], target.shape[0]):
        raise ValueError("requested rank exceeds available dimensions")
    if ridge_fraction <= 0 or rank_relative_tolerance <= 0:
        raise ValueError("ridge fraction and rank tolerance must be positive")
    sample_weights = _weights(weights, target.shape[0])
    root = np.sqrt(sample_weights)[:, None]
    x = target * root
    y = source * root
    trace = float(np.sum(x * x))
    if trace <= np.finfo(np.float64).eps:
        return HookTransport(
            target_factors=np.empty((target.shape[1], 0)),
            source_factors=np.empty((source.shape[1], 0)),
            full_singular_values=np.empty(0), requested_rank=rank, effective_rank=0,
            rank_boundary_relative_gap=None, ridge=0.0, status="TARGET_INACTIVE",
        )
    ridge = ridge_fraction * trace / min(target.shape)
    dual = np.linalg.solve(x @ x.T + ridge * np.eye(x.shape[0]), y)
    coefficient = x.T @ dual
    fitted = x @ coefficient
    _, singular, right_t = np.linalg.svd(fitted, full_matrices=False)
    numerical_rank = int(np.sum(singular > rank_relative_tolerance * singular[0])) if singular.size and singular[0] > 0 else 0
    keep = min(rank, numerical_rank)
    source_factors = right_t.T[:, :keep]
    target_factors = coefficient @ source_factors
    boundary_gap = (
        float((singular[rank - 1] - singular[rank]) / singular[0])
        if rank < singular.size and singular[0] > 0 else None
    )
    return HookTransport(
        target_factors=target_factors,
        source_factors=source_factors,
        full_singular_values=singular,
        requested_rank=rank,
        effective_rank=keep,
        rank_boundary_relative_gap=boundary_gap,
        ridge=ridge,
        status="OK" if numerical_rank >= rank else "RANK_DEFICIENT",
    )


def fit_basis_constrained_transport(
    target_process: np.ndarray,
    source_coordinates: np.ndarray,
    source_basis: np.ndarray,
    weights: np.ndarray,
    *,
    ridge_fraction: float = 1e-3,
    rank_relative_tolerance: float = 1e-10,
) -> HookTransport:
    """Fit ridge transport into a frozen ordered source-query basis.

    This is the efficient real-screen form: the source-only conditional PCA
    basis is frozen first, then one dual solve learns all ordered coordinates.
    Prefixes therefore implement the pre-registered nested rank family without
    refitting or changing the source query after calibration is observed.
    """

    target = _matrix(target_process, "target_process")
    coordinates = _matrix(source_coordinates, "source_coordinates")
    basis = _matrix(source_basis, "source_basis")
    if target.shape[0] != coordinates.shape[0]:
        raise ValueError("target process and source coordinates must share observations")
    if coordinates.shape[1] == 0 or basis.shape[1] != coordinates.shape[1]:
        raise ValueError("source basis columns must match positive source coordinates")
    if ridge_fraction <= 0 or rank_relative_tolerance <= 0:
        raise ValueError("ridge fraction and rank tolerance must be positive")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-8, rtol=1e-8):
        raise ValueError("source basis must have orthonormal columns")
    sample_weights = _weights(weights, target.shape[0])
    root = np.sqrt(sample_weights)[:, None]
    x = target * root
    y = coordinates * root
    trace = float(np.sum(x * x))
    if trace <= np.finfo(np.float64).eps:
        return HookTransport(
            target_factors=np.empty((target.shape[1], 0)),
            source_factors=np.empty((basis.shape[0], 0)),
            full_singular_values=np.empty(0), requested_rank=basis.shape[1], effective_rank=0,
            rank_boundary_relative_gap=None, ridge=0.0, status="TARGET_INACTIVE",
        )
    ridge = ridge_fraction * trace / min(target.shape)
    coefficient = x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(x.shape[0]), y)
    fitted = x @ coefficient
    singular = np.linalg.svd(fitted, compute_uv=False)
    numerical_rank = int(np.sum(singular > rank_relative_tolerance * singular[0])) if singular.size and singular[0] > 0 else 0
    requested = basis.shape[1]
    return HookTransport(
        target_factors=coefficient[:, :numerical_rank],
        source_factors=basis[:, :numerical_rank],
        full_singular_values=singular,
        requested_rank=requested,
        effective_rank=min(requested, numerical_rank),
        rank_boundary_relative_gap=None,
        ridge=ridge,
        status="OK" if numerical_rank >= requested else "RANK_DEFICIENT",
    )


def transport_prefix(transport: HookTransport, rank: int) -> HookTransport:
    """Return a nested prefix of a basis-constrained transport."""

    if rank <= 0 or rank > transport.requested_rank:
        raise ValueError("rank is outside the fitted transport family")
    keep = min(rank, transport.effective_rank)
    return HookTransport(
        target_factors=transport.target_factors[:, :keep],
        source_factors=transport.source_factors[:, :keep],
        full_singular_values=transport.full_singular_values,
        requested_rank=rank,
        effective_rank=keep,
        rank_boundary_relative_gap=None,
        ridge=transport.ridge,
        status="OK" if keep == rank else "RANK_DEFICIENT",
    )


def fit_nuisance_projector(
    process: np.ndarray,
    weights: np.ndarray,
    *,
    explained_variance_threshold: float = 0.9,
    maximum_rank: int = 64,
) -> NuisanceProjector:
    """Freeze the unique smallest global hook subspace reaching a variance target."""

    values = _matrix(process, "process")
    if not 0 < explained_variance_threshold < 1:
        raise ValueError("explained variance threshold must lie strictly between zero and one")
    if maximum_rank <= 0 or maximum_rank > min(values.shape):
        raise ValueError("maximum rank exceeds available process dimensions")
    sample_weights = _weights(weights, values.shape[0])
    weighted = np.sqrt(sample_weights)[:, None] * values
    _, singular, right_t = np.linalg.svd(weighted, full_matrices=False)
    eigenvalues = singular * singular
    total = float(np.sum(eigenvalues))
    if total <= np.finfo(np.float64).eps:
        return NuisanceProjector(np.empty((values.shape[1], 0)), eigenvalues, 0, 0.0, "INACTIVE")
    cumulative = np.cumsum(eigenvalues) / total
    candidates = np.flatnonzero(cumulative[:maximum_rank] >= explained_variance_threshold)
    if candidates.size == 0:
        rank = maximum_rank; status = "THRESHOLD_NOT_REACHED"
    else:
        rank = int(candidates[0]) + 1; status = "OK"
    return NuisanceProjector(right_t[:rank].T, eigenvalues, rank, float(cumulative[rank - 1]), status)


def residualize_hook_process(process: np.ndarray, nuisance: NuisanceProjector | np.ndarray) -> np.ndarray:
    """Remove the frozen shared nuisance subspace from a hook process."""

    values = _matrix(process, "process")
    basis = nuisance.basis if isinstance(nuisance, NuisanceProjector) else _matrix(nuisance, "nuisance_basis")
    if basis.shape[0] != values.shape[1]:
        raise ValueError("nuisance basis and hook dimensions differ")
    if basis.shape[1] == 0:
        return values.copy()
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8, rtol=1e-8):
        raise ValueError("nuisance basis must have orthonormal columns")
    return values - (values @ basis) @ basis.T


def transport_metrics(
    source_process: np.ndarray,
    transported_process: np.ndarray,
    weights: np.ndarray,
    *,
    energy_epsilon: float = 1e-12,
) -> TransportMetrics:
    source = _matrix(source_process, "source_process")
    transported = _matrix(transported_process, "transported_process")
    if source.shape != transported.shape:
        raise ValueError("source and transported processes must have identical shape")
    sample_weights = _weights(weights, source.shape[0])
    source_energy = float(np.sum(sample_weights[:, None] * source * source))
    transported_energy = float(np.sum(sample_weights[:, None] * transported * transported))
    cross = float(np.sum(sample_weights[:, None] * source * transported))
    denominator = source_energy + transported_energy
    bcc = 2.0 * cross / denominator if denominator > energy_epsilon else None
    residual = max(0.0, denominator - 2.0 * cross)
    normalized = residual / source_energy if source_energy > energy_epsilon else None
    return TransportMetrics(source_energy, transported_energy, cross, bcc, normalized)


def transport_subspace_overlap(left: HookTransport, right: HookTransport) -> float:
    """Normalized overlap of two target-side transport subspaces."""

    if left.target_factors.shape[0] != right.target_factors.shape[0]:
        raise ValueError("target hook dimensions differ")
    if left.effective_rank == 0 or right.effective_rank == 0:
        return 0.0
    q_left, _ = np.linalg.qr(left.target_factors)
    q_right, _ = np.linalg.qr(right.target_factors)
    overlap = float(np.sum((q_left.T @ q_right) ** 2))
    return overlap / min(left.effective_rank, right.effective_rank)


def decide_transport_gate(
    positive: TransportMetrics,
    negative: TransportMetrics,
    *,
    rank_boundary_relative_gap: float | None,
    collision_improvement_over_global: float,
    raw_control_specificity: float,
    global_control_specificity: float,
    minimum_bcc: float = 0.8,
    maximum_normalized_residual: float = 0.2,
    minimum_specificity: float = 0.0,
    minimum_control_advantage: float = 0.05,
    minimum_collision_improvement: float = 0.05,
    minimum_rank_gap: float = 0.001,
) -> TransportGate:
    """Apply the frozen meaningful-transfer and strong-control refusal gate."""

    if positive.bcc is None or positive.normalized_residual is None:
        return TransportGate("UNRESOLVED_RELATION", "POSITIVE_PROCESS_INACTIVE", None, None)
    negative_bcc = 0.0 if negative.bcc is None else negative.bcc
    specificity = positive.bcc - negative_bcc
    best_control = max(raw_control_specificity, global_control_specificity)
    if positive.bcc < minimum_bcc or positive.normalized_residual > maximum_normalized_residual:
        return TransportGate("UNRESOLVED_RELATION", "MEANINGFUL_TRANSFER_FLOOR", specificity, best_control)
    if specificity <= minimum_specificity:
        return TransportGate("UNRESOLVED_RELATION", "HARD_NEGATIVE_CONTRAST", specificity, best_control)
    if collision_improvement_over_global < minimum_collision_improvement:
        return TransportGate("UNRESOLVED_RELATION", "COLLISION_CONTROL", specificity, best_control)
    if rank_boundary_relative_gap is None or rank_boundary_relative_gap < minimum_rank_gap:
        return TransportGate("UNRESOLVED_RELATION", "RANK_BOUNDARY", specificity, best_control)
    if specificity - best_control < minimum_control_advantage:
        return TransportGate("UNRESOLVED_RELATION", "RAW_OR_GLOBAL_CONTROL_NOT_BEATEN", specificity, best_control)
    return TransportGate("FOUND_RELATION", None, specificity, best_control)


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite rank-2 array")
    return matrix


def _weights(value: np.ndarray, rows: int) -> np.ndarray:
    weights = np.asarray(value, dtype=np.float64).reshape(-1)
    if weights.size != rows or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite, nonnegative, and match observations")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive mass")
    return weights / total
