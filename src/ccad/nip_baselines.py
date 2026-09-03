"""Truth-free baseline primitives for prospective M1-NIP completion.

Continuous coefficients may define rankings or non-native references, but native
outputs are always unweighted target-atom supports evaluated against the common
discovery/mean feasibility rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


IMPLEMENTED_NATIVE_LANES = {
    "CONTRIBUTION_NEAREST_ATOM",
    "PW_MCC_HUNGARIAN",
    "GREEDY_DECODER_COSINE",
    "BINARY_FORWARD_OMP",
    "RANDOM_MATCHED_GROUP",
}
IMPLEMENTED_CONTINUOUS_REFERENCES = {
    "SIGNED_CONTINUOUS_REGRESSION",
    "NONNEGATIVE_CONTINUOUS_REGRESSION",
}


@dataclass(frozen=True)
class NativeBaselineResult:
    lane: str
    status: str
    identification: str
    multiplicity: str | None
    supports: tuple[tuple[int, ...], ...]
    ranking: tuple[int, ...]
    ranking_scores: tuple[float, ...]
    evaluated_support_count: int
    terminal_reason: str | None
    diagnostics: dict


@dataclass(frozen=True)
class ContinuousReferenceResult:
    lane: str
    coefficients: tuple[float, ...]
    discovery_residual_sq: float
    iterations: int
    converged: bool


def _validate_inputs(source: np.ndarray, targets: np.ndarray, source_mean: np.ndarray, target_means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(source, dtype=np.float64)
    x_atoms = np.asarray(targets, dtype=np.float64)
    mu_s = np.asarray(source_mean, dtype=np.float64).reshape(-1)
    mu_t = np.asarray(target_means, dtype=np.float64)
    if y.ndim != 2 or x_atoms.ndim != 3 or y.shape[0] != x_atoms.shape[0] or y.shape[1] != x_atoms.shape[2]:
        raise ValueError("source and target contributions must share observations and hook dimension")
    if mu_t.shape != (y.shape[1], x_atoms.shape[1]) or mu_s.shape != (y.shape[1],):
        raise ValueError("mean contribution shapes are inconsistent")
    if not all(np.all(np.isfinite(value)) for value in (y, x_atoms, mu_s, mu_t)):
        raise ValueError("baseline inputs must be finite")
    return y, x_atoms, mu_s, mu_t


def _score_support(y: np.ndarray, x_atoms: np.ndarray, mu_s: np.ndarray, mu_t: np.ndarray, support: tuple[int, ...], *, epsilon: float) -> tuple[float, float]:
    prediction = np.sum(x_atoms[:, np.asarray(support), :], axis=1)
    residual = y - prediction
    d_ctr = float(np.mean(np.sum(residual * residual, axis=1))) / (float(np.mean(np.sum(y * y, axis=1))) + epsilon)
    mean_delta = mu_s - np.sum(mu_t[:, np.asarray(support)], axis=1)
    d_mu = float(mean_delta @ mean_delta) / (float(mu_s @ mu_s) + epsilon)
    return d_ctr, d_mu


def _result_from_ranking(
    lane: str,
    ranking: tuple[int, ...],
    scores: tuple[float, ...],
    y: np.ndarray,
    x_atoms: np.ndarray,
    mu_s: np.ndarray,
    mu_t: np.ndarray,
    *,
    g_max: int,
    tau_ctr: float,
    tau_mu: float,
    epsilon: float,
    tie_tolerance: float,
    refuse_membership_ties: bool,
    diagnostics: dict | None = None,
) -> NativeBaselineResult:
    evaluated = 0
    limit = min(g_max, len(ranking))
    for size in range(1, limit + 1):
        if refuse_membership_ties and size < len(ranking) and abs(scores[size - 1] - scores[size]) <= tie_tolerance:
            return NativeBaselineResult(lane, "BUDGET_REFUSAL", "UNRESOLVED", None, (), ranking, scores, evaluated, "BOUNDARY_TIE", diagnostics or {})
        support = tuple(sorted(ranking[:size]))
        d_ctr, d_mu = _score_support(y, x_atoms, mu_s, mu_t, support, epsilon=epsilon)
        evaluated += 1
        if d_ctr <= tau_ctr and d_mu <= tau_mu:
            return NativeBaselineResult(lane, "OK", "FOUND", "UNIQUE", (support,), ranking, scores, evaluated, None, diagnostics or {})
    return NativeBaselineResult(lane, "OK", "UNRESOLVED", None, (), ranking, scores, evaluated, "NO_ACCEPTED_PREFIX", diagnostics or {})


def singleton_d_ctr(source: np.ndarray, targets: np.ndarray, *, epsilon: float) -> np.ndarray:
    y = np.asarray(source, dtype=np.float64)
    x_atoms = np.asarray(targets, dtype=np.float64)
    denominator = float(np.mean(np.sum(y * y, axis=1))) + epsilon
    return np.asarray([float(np.mean(np.sum((y - x_atoms[:, atom, :]) ** 2, axis=1))) / denominator for atom in range(x_atoms.shape[1])])


def infer_decoder_directions(contributions: np.ndarray) -> np.ndarray:
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("contributions must be observation x atom x hook")
    directions = np.zeros((values.shape[2], values.shape[1]), dtype=np.float64)
    for atom in range(values.shape[1]):
        _, singular, vh = np.linalg.svd(values[:, atom, :], full_matrices=False)
        if singular.size and singular[0] > 0.0:
            directions[:, atom] = vh[0]
    return directions


def run_native_baseline(
    lane: str,
    source: np.ndarray,
    targets: np.ndarray,
    source_mean: np.ndarray,
    target_means: np.ndarray,
    *,
    g_max: int,
    tau_ctr: float,
    tau_mu: float,
    epsilon: float,
    tie_tolerance: float,
    solver_seed: int,
) -> NativeBaselineResult:
    if lane not in IMPLEMENTED_NATIVE_LANES:
        raise NotImplementedError(f"native lane is not implemented: {lane}")
    y, x_atoms, mu_s, mu_t = _validate_inputs(source, targets, source_mean, target_means)
    atom_count = x_atoms.shape[1]
    if g_max < 1 or epsilon <= 0.0 or tie_tolerance < 0.0:
        raise ValueError("invalid support or numerical parameters")

    if lane == "CONTRIBUTION_NEAREST_ATOM":
        residuals = singleton_d_ctr(y, x_atoms, epsilon=epsilon)
        order = tuple(sorted(range(atom_count), key=lambda atom: (residuals[atom], atom)))
        best = residuals[order[0]]
        tied = tuple(atom for atom in order if abs(residuals[atom] - best) <= tie_tolerance)
        feasible = tuple((atom,) for atom in tied if max(_score_support(y, x_atoms, mu_s, mu_t, (atom,), epsilon=epsilon)[0] / max(tau_ctr, epsilon), _score_support(y, x_atoms, mu_s, mu_t, (atom,), epsilon=epsilon)[1] / max(tau_mu, epsilon)) <= 1.0)
        if feasible:
            return NativeBaselineResult(lane, "OK", "FOUND", "UNIQUE" if len(feasible) == 1 else "AMBIGUOUS", feasible, order, tuple(float(residuals[a]) for a in order), len(tied), None, {})
        return NativeBaselineResult(lane, "OK", "UNRESOLVED", None, (), order, tuple(float(residuals[a]) for a in order), len(tied), "NO_FEASIBLE_SINGLETON", {})

    if lane in {"PW_MCC_HUNGARIAN", "GREEDY_DECODER_COSINE"}:
        source_direction = infer_decoder_directions(y[:, None, :])[:, 0]
        target_directions = infer_decoder_directions(x_atoms)
        denominator = np.linalg.norm(source_direction) * np.linalg.norm(target_directions, axis=0)
        similarities = np.full(atom_count, -np.inf)
        active = denominator > 0.0
        similarities[active] = np.abs(source_direction @ target_directions[:, active] / denominator[active])
        order = tuple(sorted(range(atom_count), key=lambda atom: (-similarities[atom], atom)))
        scores = tuple(float(similarities[atom]) for atom in order)
        if lane == "PW_MCC_HUNGARIAN":
            if not np.isfinite(scores[0]):
                return NativeBaselineResult(lane, "OK", "UNRESOLVED", None, (), order, scores, 0, "ZERO_NORM_INELIGIBLE", {"scope": "DEGENERATE_SINGLETON_BASELINE"})
            best = scores[0]
            tied = tuple(atom for atom in order if abs(similarities[atom] - best) <= tie_tolerance)
            feasible = tuple((atom,) for atom in tied if all(value <= threshold for value, threshold in zip(_score_support(y, x_atoms, mu_s, mu_t, (atom,), epsilon=epsilon), (tau_ctr, tau_mu))))
            return NativeBaselineResult(lane, "OK", "FOUND" if feasible else "UNRESOLVED", "UNIQUE" if len(feasible) == 1 else "AMBIGUOUS" if feasible else None, feasible, order, scores, len(tied), None if feasible else "NO_FEASIBLE_ASSIGNED_SINGLETON", {"scope": "DEGENERATE_SINGLETON_BASELINE"})
        return _result_from_ranking(lane, order, scores, y, x_atoms, mu_s, mu_t, g_max=g_max, tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=epsilon, tie_tolerance=tie_tolerance, refuse_membership_ties=True)

    if lane == "BINARY_FORWARD_OMP":
        design = x_atoms.transpose(0, 2, 1).reshape(-1, atom_count)
        response = y.reshape(-1)
        norms = np.linalg.norm(design, axis=0)
        normalized = np.zeros_like(design)
        active = norms > 0.0
        normalized[:, active] = design[:, active] / norms[active]
        selected: list[int] = []
        coefficients = np.asarray([], dtype=np.float64)
        residual = response.copy()
        ranking_scores: list[float] = []
        for _ in range(min(g_max, atom_count)):
            correlations = np.abs(normalized.T @ residual)
            correlations[np.asarray(selected, dtype=int)] = -np.inf
            best = float(np.max(correlations))
            tied = np.flatnonzero(np.abs(correlations - best) <= tie_tolerance)
            if len(tied) != 1:
                return NativeBaselineResult(lane, "BUDGET_REFUSAL", "UNRESOLVED", None, (), tuple(selected), tuple(ranking_scores), len(selected), "SELECTION_TIE", {"continuous_coefficients": tuple(float(value) for value in coefficients)})
            selected.append(int(tied[0]))
            ranking_scores.append(best)
            coefficients, *_ = np.linalg.lstsq(design[:, selected], response, rcond=1e-12)
            residual = response - design[:, selected] @ coefficients
            support = tuple(sorted(selected))
            d_ctr, d_mu = _score_support(y, x_atoms, mu_s, mu_t, support, epsilon=epsilon)
            if d_ctr <= tau_ctr and d_mu <= tau_mu:
                return NativeBaselineResult(lane, "OK", "FOUND", "UNIQUE", (support,), tuple(selected), tuple(ranking_scores), len(selected), None, {"continuous_coefficients": tuple(float(value) for value in coefficients), "coefficients_used_for_native_endpoint": False})
        return NativeBaselineResult(lane, "OK", "UNRESOLVED", None, (), tuple(selected), tuple(ranking_scores), len(selected), "NO_ACCEPTED_PREFIX", {"continuous_coefficients": tuple(float(value) for value in coefficients), "coefficients_used_for_native_endpoint": False})

    rng = np.random.default_rng(solver_seed)
    order = tuple(int(value) for value in rng.permutation(atom_count))
    scores = tuple(float(atom_count - index) for index in range(atom_count))
    return _result_from_ranking(lane, order, scores, y, x_atoms, mu_s, mu_t, g_max=g_max, tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=epsilon, tie_tolerance=tie_tolerance, refuse_membership_ties=False, diagnostics={"primary_replicate_index": 0, "diagnostic_replicates_not_run": 32})


def run_continuous_reference(lane: str, source: np.ndarray, targets: np.ndarray, *, tolerance: float = 1e-10, objective_tolerance: float = 1e-12, max_iterations: int = 10000) -> ContinuousReferenceResult:
    if lane not in IMPLEMENTED_CONTINUOUS_REFERENCES:
        raise NotImplementedError(f"continuous lane is not implemented: {lane}")
    y = np.asarray(source, dtype=np.float64).reshape(-1)
    x_atoms = np.asarray(targets, dtype=np.float64)
    if x_atoms.ndim != 3 or x_atoms.shape[0] * x_atoms.shape[2] != y.size:
        raise ValueError("continuous reference tensors are incompatible")
    design = x_atoms.transpose(0, 2, 1).reshape(y.size, x_atoms.shape[1])
    if lane == "SIGNED_CONTINUOUS_REGRESSION":
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=1e-12)
        residual = y - design @ coefficients
        return ContinuousReferenceResult(lane, tuple(float(value) for value in coefficients), float(residual @ residual), 1, True)

    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    spectral = float(np.linalg.norm(design, ord=2))
    if spectral == 0.0:
        return ContinuousReferenceResult(lane, tuple(float(value) for value in coefficients), float(y @ y), 0, True)
    step = 1.0 / (spectral * spectral)
    previous_objective = float(y @ y)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        residual = design @ coefficients - y
        updated = np.maximum(0.0, coefficients - step * (design.T @ residual))
        new_residual = design @ updated - y
        objective = float(new_residual @ new_residual)
        if np.max(np.abs(updated - coefficients)) <= tolerance or abs(previous_objective - objective) <= objective_tolerance:
            coefficients = updated
            converged = True
            break
        coefficients = updated
        previous_objective = objective
    residual = y - design @ coefficients
    return ContinuousReferenceResult(lane, tuple(float(value) for value in coefficients), float(residual @ residual), iterations, converged)
