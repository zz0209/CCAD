"""Raw, JSON-ready metric surface for frozen native-support candidates.

The adapter consumes observed tensors and an already frozen support.  It has no
truth, label, family-rule, proposal, or selection input.  Algorithmic fields that
cannot be measured from tensors alone remain explicitly not applicable until the
post-closure scorer supplies them.
"""

from __future__ import annotations

import numpy as np


def _kish(weights: np.ndarray) -> float:
    denominator = float(weights @ weights)
    return 0.0 if denominator == 0.0 else float(np.sum(weights) ** 2 / denominator)


def _direction_matrix(contributions: np.ndarray) -> np.ndarray:
    directions = []
    for atom in range(contributions.shape[1]):
        _, singular, vh = np.linalg.svd(contributions[:, atom, :], full_matrices=False)
        if singular.size and singular[0] > 0.0:
            directions.append(vh[0])
    if not directions:
        return np.zeros((contributions.shape[2], 0), dtype=np.float64)
    return np.asarray(directions, dtype=np.float64).T


def _psc(source_directions: np.ndarray, target_directions: np.ndarray) -> dict:
    def basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if matrix.shape[1] == 0:
            return np.zeros((matrix.shape[0], 0)), np.asarray([], dtype=np.float64)
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        rank = 0 if not singular.size or singular[0] == 0.0 else int(np.sum(singular > 1e-12 * singular[0]))
        return u[:, :rank], singular[:rank]

    left, _ = basis(source_directions)
    right, right_singular = basis(target_directions)
    rank_left, rank_right = left.shape[1], right.shape[1]
    if rank_left == 0 or rank_right == 0:
        return {
            "psc_value": None, "psc_rank_source": rank_left, "psc_rank_target": rank_right,
            "psc_projector_distance_sq": None, "psc_principal_angles_radians": [],
            "effective_rank": rank_right, "condition_number": None, "psc_status": "DEGENERATE",
        }
    projector_distance = float(np.sum((left @ left.T - right @ right.T) ** 2))
    cosine = np.linalg.svd(left.T @ right, compute_uv=False)
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    condition = None if not right_singular.size or right_singular[-1] == 0.0 else float(right_singular[0] / right_singular[-1])
    return {
        "psc_value": 1.0 - projector_distance / (rank_left + rank_right),
        "psc_rank_source": rank_left,
        "psc_rank_target": rank_right,
        "psc_projector_distance_sq": projector_distance,
        "psc_principal_angles_radians": [float(value) for value in angles],
        "effective_rank": rank_right,
        "condition_number": condition,
        "psc_status": "OK",
    }


def native_support_metric_surface(
    source_contributions: np.ndarray,
    target_contributions: np.ndarray,
    source_mean_contributions: np.ndarray,
    target_mean_contributions: np.ndarray,
    document_ids: np.ndarray,
    *,
    source_atom_id: int,
    target_ids: tuple[int, ...],
    epsilon: float,
    algorithm_diagnostics: dict | None = None,
) -> dict:
    """Measure a frozen support without using truth or selecting a candidate."""
    source_all = np.asarray(source_contributions, dtype=np.float64)
    target_all = np.asarray(target_contributions, dtype=np.float64)
    source_means = np.asarray(source_mean_contributions, dtype=np.float64)
    target_means = np.asarray(target_mean_contributions, dtype=np.float64)
    documents = np.asarray(document_ids).reshape(-1)
    ids = tuple(int(value) for value in target_ids)
    if source_all.ndim != 3 or target_all.ndim != 3 or source_all.shape[0] != target_all.shape[0] or source_all.shape[2] != target_all.shape[2]:
        raise ValueError("contributions must be paired observation x atom x hook tensors")
    if source_all.shape[0] != documents.size or not ids or tuple(sorted(set(ids))) != ids:
        raise ValueError("documents must align and target_ids must be nonempty, sorted, and unique")
    if not 0 <= source_atom_id < source_all.shape[1] or ids[-1] >= target_all.shape[1] or epsilon <= 0.0:
        raise ValueError("atom IDs and epsilon are invalid")
    if source_means.shape != (source_all.shape[2], source_all.shape[1]) or target_means.shape != (target_all.shape[2], target_all.shape[1]):
        raise ValueError("mean contribution shapes do not match hook and atom dimensions")

    source = source_all[:, source_atom_id, :]
    selected = target_all[:, np.asarray(ids), :]
    target = np.sum(selected, axis=1)
    source_energy = float(np.mean(np.sum(source * source, axis=1)))
    target_energy = float(np.mean(np.sum(target * target, axis=1)))
    cross_inner = float(np.mean(np.sum(source * target, axis=1)))
    centered_numerator = max(0.0, source_energy + target_energy - 2.0 * cross_inner)
    centered_denominator = source_energy + epsilon
    bcc_denominator = source_energy + target_energy
    bcc_value = None if bcc_denominator <= 1e-15 else 2.0 * cross_inner / bcc_denominator
    bcc_residual = None if bcc_denominator <= 1e-15 else centered_numerator / bcc_denominator

    source_mean = source_means[:, source_atom_id]
    target_mean = np.sum(target_means[:, np.asarray(ids)], axis=1)
    mean_delta = source_mean - target_mean
    mean_numerator = float(mean_delta @ mean_delta)
    mean_denominator = float(source_mean @ source_mean) + epsilon

    individual_energy = np.mean(np.sum(selected * selected, axis=2), axis=0)
    cancellation = None if target_energy <= 1e-15 else float(np.sum(individual_energy) / target_energy)
    leverage = [] if target_energy <= 1e-15 else [float(value / target_energy) for value in individual_energy]
    token_energy = np.sum(source * source, axis=1)
    unique_documents = np.unique(documents)
    document_energy = np.asarray([np.sum(token_energy[documents == doc]) for doc in unique_documents], dtype=np.float64)

    psc = _psc(
        _direction_matrix(source_all[:, [source_atom_id], :]),
        _direction_matrix(target_all[:, np.asarray(ids), :]),
    )
    not_applicable = {"status": "NOT_APPLICABLE_PRELABEL", "value": None}
    surface = {
        "schema_version": "metric_surface.v2-nip",
        "centered_residual_numerator": centered_numerator,
        "centered_source_energy_denominator": centered_denominator,
        "d_ctr": centered_numerator / centered_denominator,
        "mean_residual_numerator": mean_numerator,
        "mean_source_energy_denominator": mean_denominator,
        "d_mu": mean_numerator / mean_denominator,
        "bcc_value": bcc_value,
        "bcc_cross_inner": cross_inner,
        "bcc_source_energy": source_energy,
        "bcc_target_energy": target_energy,
        "bcc_normalized_residual": bcc_residual,
        **psc,
        "source_mean_contribution": [float(value) for value in source_mean],
        "target_mean_contribution": [float(value) for value in target_mean],
        "mean_difference_norm": float(np.linalg.norm(mean_delta)),
        "support_size": len(ids),
        "unmatched_energy": centered_numerator,
        "cancellation_ratio": cancellation,
        "leave_one_out_leverage": leverage,
        "occupancy": int(np.sum(token_energy > 1e-15)),
        "active_document_count": int(np.sum(document_energy > 1e-15)),
        "document_ess": _kish(document_energy),
        "multiplicity": not_applicable,
        "tie_set": not_applicable,
        "nearest_competitor_margin": not_applicable,
        "solver_gap": not_applicable,
        "proposal_stability": not_applicable,
        "proposal_recall": not_applicable,
        "conditional_solver_correctness": not_applicable,
        "end_to_end_recovery": not_applicable,
        "coverage": not_applicable,
        "terminal_reason": not_applicable,
    }
    if algorithm_diagnostics:
        unknown = set(algorithm_diagnostics) - set(surface)
        if unknown:
            raise ValueError(f"unknown algorithm diagnostic fields: {sorted(unknown)}")
        surface.update(algorithm_diagnostics)
    return surface
