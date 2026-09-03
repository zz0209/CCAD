"""Truth-free baseline primitives for prospective M1-NIP completion.

Continuous coefficients may define rankings or non-native references, but native
outputs are always unweighted target-atom supports evaluated against the common
discovery/mean feasibility rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccad.proposal import li15_spectral_proposal


IMPLEMENTED_NATIVE_LANES = {
    "CONTRIBUTION_NEAREST_ATOM",
    "PW_MCC_HUNGARIAN",
    "GREEDY_DECODER_COSINE",
    "DUSTBIN_SINKHORN",
    "BINARY_FORWARD_OMP",
    "OT_MASS_NATIVE_SUPPORT",
    "SPECTRAL_LOCAL_SVD_NATIVE_SUPPORT",
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


def infer_rank_one_codes(contributions: np.ndarray, *, relative_tolerance: float = 1e-12) -> tuple[np.ndarray, tuple[float, ...]]:
    """Factor per-atom contribution matrices into codes, failing closed on rank > 1."""
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 3 or relative_tolerance < 0.0:
        raise ValueError("contributions must be observation x atom x hook and tolerance nonnegative")
    codes = np.zeros((values.shape[0], values.shape[1]), dtype=np.float64)
    residuals: list[float] = []
    for atom in range(values.shape[1]):
        u, singular, _ = np.linalg.svd(values[:, atom, :], full_matrices=False)
        total_sq = float(singular @ singular)
        tail_sq = float(singular[1:] @ singular[1:]) if singular.size > 1 else 0.0
        residual = np.sqrt(tail_sq / total_sq) if total_sq > 0.0 else 0.0
        if residual > relative_tolerance:
            raise ValueError(f"atom {atom} contribution process is not rank one: relative residual {residual:.6g}")
        if singular.size:
            codes[:, atom] = u[:, 0] * singular[0]
        residuals.append(float(residual))
    return codes, tuple(residuals)


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.where(finite, values - maximum, -np.inf)
    result = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) + maximum
    return np.squeeze(result, axis=axis)


def _balanced_log_sinkhorn(cost: np.ndarray, a: np.ndarray, b: np.ndarray, *, regularization: float, tolerance: float, max_iterations: int) -> tuple[np.ndarray, int, bool, float]:
    if regularization <= 0.0 or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid Sinkhorn numerical parameters")
    matrix = np.asarray(cost, dtype=np.float64)
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if matrix.shape != (left.size, right.size) or np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("Sinkhorn cost and positive marginals are incompatible")
    if not np.isclose(np.sum(left), np.sum(right), atol=tolerance, rtol=0.0):
        raise ValueError("balanced Sinkhorn marginals must have equal mass")
    log_kernel = -matrix / regularization
    log_u = np.zeros(left.size, dtype=np.float64)
    log_v = np.zeros(right.size, dtype=np.float64)
    log_a = np.log(left)
    log_b = np.log(right)
    converged = False
    marginal_error = np.inf
    for iteration in range(1, max_iterations + 1):
        log_u = log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_kernel + log_u[:, None], axis=0)
        plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
        marginal_error = max(float(np.max(np.abs(np.sum(plan, axis=1) - left))), float(np.max(np.abs(np.sum(plan, axis=0) - right))))
        if marginal_error <= tolerance:
            converged = True
            break
    return plan, iteration, converged, marginal_error


def _unbalanced_log_sinkhorn(cost: np.ndarray, a: np.ndarray, b: np.ndarray, *, regularization: float, marginal_relaxation: float, tolerance: float, max_iterations: int) -> tuple[np.ndarray, int, bool, float]:
    if regularization <= 0.0 or marginal_relaxation <= 0.0 or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid unbalanced Sinkhorn numerical parameters")
    matrix = np.asarray(cost, dtype=np.float64)
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if matrix.shape != (left.size, right.size) or np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("unbalanced Sinkhorn cost and positive marginals are incompatible")
    power = marginal_relaxation / (marginal_relaxation + regularization)
    log_kernel = -matrix / regularization
    log_u = np.zeros(left.size, dtype=np.float64)
    log_v = np.zeros(right.size, dtype=np.float64)
    log_a = np.log(left)
    log_b = np.log(right)
    converged = False
    scaling_change = np.inf
    for iteration in range(1, max_iterations + 1):
        previous_u = log_u.copy()
        previous_v = log_v.copy()
        log_u = power * (log_a - _logsumexp(log_kernel + log_v[None, :], axis=1))
        log_v = power * (log_b - _logsumexp(log_kernel + log_u[:, None], axis=0))
        scaling_change = max(float(np.max(np.abs(log_u - previous_u))), float(np.max(np.abs(log_v - previous_v))))
        if scaling_change <= tolerance:
            converged = True
            break
    plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return plan, iteration, converged, scaling_change


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

    if lane in {"DUSTBIN_SINKHORN", "OT_MASS_NATIVE_SUPPORT"}:
        costs = np.clip(singleton_d_ctr(y, x_atoms, epsilon=epsilon), 0.0, 1.0)
        if lane == "DUSTBIN_SINKHORN":
            augmented = np.full((2, atom_count + 1), tau_ctr, dtype=np.float64)
            augmented[0, :atom_count] = costs
            augmented[1, atom_count] = 0.0
            plan, iterations, converged, error = _balanced_log_sinkhorn(
                augmented, np.full(2, 0.5), np.full(atom_count + 1, 1.0 / (atom_count + 1)),
                regularization=0.05, tolerance=1e-9, max_iterations=1000,
            )
            masses = plan[0, :atom_count]
            diagnostic_name = "marginal_error"
        else:
            plan, iterations, converged, error = _unbalanced_log_sinkhorn(
                costs[None, :], np.ones(1), np.full(atom_count, 1.0 / atom_count),
                regularization=0.05, marginal_relaxation=1.0, tolerance=1e-9, max_iterations=1000,
            )
            masses = plan[0]
            diagnostic_name = "scaling_change"
        order = tuple(sorted(range(atom_count), key=lambda atom: (-masses[atom], atom)))
        scores = tuple(float(masses[atom]) for atom in order)
        diagnostics = {
            "scope": "DEGENERATE_SINGLE_QUERY",
            "solver": "BALANCED_LOG_SINKHORN" if lane == "DUSTBIN_SINKHORN" else "UNBALANCED_LOG_SINKHORN",
            "iterations": iterations,
            "converged": converged,
            diagnostic_name: error,
            "transport_mass": float(np.sum(plan)),
        }
        if not converged:
            return NativeBaselineResult(lane, "BUDGET_REFUSAL", "UNRESOLVED", None, (), order, scores, 0, "SINKHORN_DID_NOT_CONVERGE", diagnostics)
        return _result_from_ranking(lane, order, scores, y, x_atoms, mu_s, mu_t, g_max=g_max, tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=epsilon, tie_tolerance=tie_tolerance, refuse_membership_ties=True, diagnostics=diagnostics)

    if lane == "SPECTRAL_LOCAL_SVD_NATIVE_SUPPORT":
        source_codes, source_residuals = infer_rank_one_codes(y[:, None, :], relative_tolerance=1e-12)
        target_codes, target_residuals = infer_rank_one_codes(x_atoms, relative_tolerance=1e-12)
        spectral = li15_spectral_proposal(
            source_codes, target_codes, correlation_threshold=0.2, max_clusters=8,
            kmeans_seed=solver_seed, max_neighborhood_atoms=atom_count + 1,
        )
        residuals = singleton_d_ctr(y, x_atoms, epsilon=epsilon)
        best = min(range(atom_count), key=lambda atom: (residuals[atom], atom))
        neighborhood = next((item for item in spectral.proposal.neighborhoods if item.anchor_left == 0 and item.anchor_right == best), None)
        diagnostics = {
            "cluster_count": spectral.cluster_count,
            "eigenvalues": spectral.eigenvalues,
            "mixed_cluster_count": spectral.mixed_cluster_count,
            "best_contribution_singleton": best,
            "source_factorization_residuals": source_residuals,
            "target_factorization_residuals": target_residuals,
        }
        if neighborhood is None or neighborhood.status != "OK" or not neighborhood.right_ids:
            reason = "NO_MIXED_CLUSTER" if neighborhood is None else neighborhood.refusal_reason or "NO_MIXED_CLUSTER"
            return NativeBaselineResult(lane, "BUDGET_REFUSAL" if neighborhood is not None and neighborhood.status != "OK" else "OK", "UNRESOLVED", None, (), (), (), 0, reason, diagnostics)
        order = tuple(sorted(neighborhood.right_ids, key=lambda atom: (residuals[atom], atom)))
        scores = tuple(float(residuals[atom]) for atom in order)
        return _result_from_ranking(lane, order, scores, y, x_atoms, mu_s, mu_t, g_max=min(g_max, 4), tau_ctr=tau_ctr, tau_mu=tau_mu, epsilon=epsilon, tie_tolerance=tie_tolerance, refuse_membership_ties=True, diagnostics=diagnostics)

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
