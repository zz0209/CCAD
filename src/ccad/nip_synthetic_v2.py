"""Prospective cap-identifiable NIP construction.

This module is an implementation probe for a future protocol suffix.  It does
not alter the locked v1 generator and contains no label/evaluation API.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ccad.nip_synthetic import NIPObservedInstance, generate_nip_observed


TARGET_ATOM_COUNT = 20
DECOY_ORTHOGONAL_ENERGY = 0.08
REGISTERED_CAPS = (4, 8, 12, 16, 20)

_BASE_TARGET_COUNTS = {
    "N01_structured_split": 3, "N02_structured_merge_refactorization": 3,
    "N03_tied_native_supports": 4, "N04_absent_target": 2, "N05_bloated_decoy": 4,
    "N06_exact_dense_orthogonal_rotation": 2, "N07_margin_separated_approximate_rotation": 2,
    "N08_continuous_only_representation": 2, "N09_cancellation": 2,
    "N10_rare_occupancy": 3, "N11_downstream_cliff": 1, "N12_mean_mismatch": 1,
}

# Construction-only rank pressure. Values mean the first registered cap at
# which every planted atom needed for the family construction is proposed.
CAP_PRESSURE = {
    "N11_downstream_cliff": 4,
    "N01_structured_split": 8,
    "N02_structured_merge_refactorization": 12,
    "N03_tied_native_supports": 16,
    "N05_bloated_decoy": 20,
    "N09_cancellation": 12,
    "N10_rare_occupancy": 16,
}

_CONSTRUCTION_REQUIRED_IDS = {
    "N01_structured_split": (0, 1),
    "N02_structured_merge_refactorization": (0, 1, 2),
    "N03_tied_native_supports": (0, 1, 2, 3),
    "N05_bloated_decoy": (0, 1),
    "N09_cancellation": (0, 1),
    "N10_rare_occupancy": (0, 1),
    "N11_downstream_cliff": (0,),
}


def _orthogonal_decoy_basis(instance: NIPObservedInstance, count: int, seed: int) -> np.ndarray:
    n, _, hook_dim = instance.target_contributions.shape
    all_atoms = np.concatenate((instance.source_contributions, instance.target_contributions), axis=1)
    codes = []
    source_direction = None
    for atom_id, atom in enumerate(all_atoms.transpose(1, 0, 2)):
        u, singular, vh = np.linalg.svd(atom, full_matrices=False)
        total_sq = float(singular @ singular)
        tail_sq = float(singular[1:] @ singular[1:]) if singular.size > 1 else 0.0
        residual = np.sqrt(tail_sq / total_sq) if total_sq > 0.0 else 0.0
        if residual > 1e-12:
            raise RuntimeError(f"base atom {atom_id} is not a native rank-one contribution")
        codes.append(u[:, 0] * singular[0] if singular.size else np.zeros(n))
        if atom_id == instance.source_atom_id:
            source_direction = vh[0]
    if source_direction is None or np.linalg.norm(source_direction) == 0.0:
        raise RuntimeError("source atom has no decoder direction")
    source_direction = source_direction / np.linalg.norm(source_direction)
    # A code vector orthogonal to every existing atom code gives a flattened
    # rank-one contribution orthogonal to every existing contribution,
    # regardless of decoder direction.  Orthogonality to the constant vector
    # also gives zero sample mean in every hook coordinate.
    forbidden_matrix = np.stack([*codes, np.ones(n, dtype=np.float64)], axis=1)
    u_forbidden, singular_forbidden, _ = np.linalg.svd(forbidden_matrix, full_matrices=False)
    forbidden_rank = 0 if not singular_forbidden.size or singular_forbidden[0] == 0.0 else int(np.sum(singular_forbidden > 1e-12 * singular_forbidden[0]))
    q_forbidden = u_forbidden[:, :forbidden_rank]
    rng = np.random.default_rng(seed)
    candidates = rng.standard_normal((n, count))
    candidates -= q_forbidden @ (q_forbidden.T @ candidates)
    code_basis, triangular = np.linalg.qr(candidates, mode="reduced")
    if code_basis.shape[1] != count or np.min(np.abs(np.diag(triangular))) < 1e-10:
        raise RuntimeError("insufficient orthogonal complement for decoy construction")
    return np.stack([(code_basis[:, column, None] * source_direction[None, :]).reshape(-1) for column in range(count)], axis=1)


def _base_singleton_residuals(instance: NIPObservedInstance) -> np.ndarray:
    source = instance.source_contributions[:, instance.source_atom_id, :].reshape(-1)
    targets = instance.target_contributions.transpose(1, 0, 2).reshape(instance.target_contributions.shape[1], -1)
    denominator = float(source @ source)
    if denominator <= 0.0:
        raise ValueError("source contribution must have positive energy")
    return np.sum((targets - source[None, :]) ** 2, axis=1) / denominator


def _decoy_residual_schedule(instance: NIPObservedInstance, decoy_count: int) -> np.ndarray:
    required = _CONSTRUCTION_REQUIRED_IDS.get(instance.family_id)
    if required is None or CAP_PRESSURE.get(instance.family_id) == 4:
        return np.linspace(1.25, 2.0, decoy_count, dtype=np.float64)
    base_scores = _base_singleton_residuals(instance)
    critical = float(np.max(base_scores[np.asarray(required, dtype=int)]))
    previous_cap = REGISTERED_CAPS[REGISTERED_CAPS.index(CAP_PRESSURE[instance.family_id]) - 1]
    base_at_or_ahead = int(np.sum(base_scores <= critical + 1e-12))
    low_count = previous_cap + 1 - base_at_or_ahead
    if not 0 < low_count <= decoy_count or critical <= DECOY_ORTHOGONAL_ENERGY + 0.04:
        raise RuntimeError("requested cap-pressure band is infeasible for this construction")
    low = np.linspace(DECOY_ORTHOGONAL_ENERGY + 0.01, critical - 0.02, low_count, dtype=np.float64)
    high_count = decoy_count - low_count
    high = np.linspace(critical + 0.20, critical + 0.80, high_count, dtype=np.float64) if high_count else np.empty(0)
    return np.concatenate((low, high))


def generate_cap_identifiable_observed(
    family_id: str, *, structural_seed: int, sample_seed: int, n: int = 512
) -> NIPObservedInstance:
    """Return a 20-target observed instance with construction-safe decoys."""
    base = generate_nip_observed(family_id, structural_seed=structural_seed, sample_seed=sample_seed, n=n)
    decoy_count = TARGET_ATOM_COUNT - base.target_contributions.shape[1]
    if decoy_count <= 0:
        raise ValueError("base construction must have fewer than 20 target atoms")
    basis = _orthogonal_decoy_basis(base, decoy_count, structural_seed ^ 0xC023C023)
    residuals = _decoy_residual_schedule(base, decoy_count)
    beta = np.sqrt(DECOY_ORTHOGONAL_ENERGY)
    source = base.source_contributions[:, base.source_atom_id, :].reshape(-1)
    source_norm = float(np.linalg.norm(source))
    decoys = []
    for column, residual in enumerate(residuals):
        alpha = 1.0 - np.sqrt(float(residual) - DECOY_ORTHOGONAL_ENERGY)
        flat = alpha * source + beta * source_norm * basis[:, column]
        decoys.append(flat.reshape(base.source_contributions.shape[0], base.source_contributions.shape[2]))
    target = np.concatenate((base.target_contributions, np.stack(decoys, axis=1)), axis=1)
    target_mean = np.concatenate(
        (base.target_mean_contributions, np.zeros((base.target_mean_contributions.shape[0], decoy_count))), axis=1
    )
    return replace(base, target_contributions=target, target_mean_contributions=target_mean)


def construction_certificate(instance: NIPObservedInstance) -> dict[str, object]:
    """Return construction-only invariants; no method prediction or truth label."""
    if instance.target_contributions.shape[1] != TARGET_ATOM_COUNT:
        raise ValueError("certificate requires the 20-atom construction")
    base_count = _BASE_TARGET_COUNTS[instance.family_id]
    n, _, hook_dim = instance.target_contributions.shape
    source = instance.source_contributions[:, instance.source_atom_id, :].reshape(-1)
    base_targets = instance.target_contributions[:, :base_count, :].transpose(1, 0, 2).reshape(base_count, -1)
    decoys = instance.target_contributions[:, base_count:, :].transpose(1, 0, 2).reshape(TARGET_ATOM_COUNT - base_count, -1)
    forbidden = [source, *base_targets]
    for hook_index in range(hook_dim):
        constant = np.zeros((n, hook_dim), dtype=np.float64)
        constant[:, hook_index] = 1.0
        forbidden.append(constant.reshape(-1))
    forbidden_matrix = np.stack(forbidden, axis=1)
    u_forbidden, singular_forbidden, _ = np.linalg.svd(forbidden_matrix, full_matrices=False)
    forbidden_rank = 0 if not singular_forbidden.size or singular_forbidden[0] == 0.0 else int(np.sum(singular_forbidden > 1e-12 * singular_forbidden[0]))
    q_forbidden = u_forbidden[:, :forbidden_rank]
    residual_vectors = decoys - (decoys @ q_forbidden) @ q_forbidden.T
    source_energy = float(source @ source)
    residual_energies = np.sum(residual_vectors * residual_vectors, axis=1) / source_energy
    normalized = residual_vectors / np.linalg.norm(residual_vectors, axis=1, keepdims=True)
    gram = normalized @ normalized.T
    off_diagonal = gram - np.eye(gram.shape[0])

    expected_cap = CAP_PRESSURE.get(instance.family_id)
    observed_cap = None
    if expected_cap is not None:
        scores = np.sum((instance.target_contributions.transpose(1, 0, 2).reshape(TARGET_ATOM_COUNT, -1) - source[None, :]) ** 2, axis=1) / source_energy
        ranking = tuple(sorted(range(TARGET_ATOM_COUNT), key=lambda atom_id: (float(scores[atom_id]), atom_id)))
        worst_rank = max(ranking.index(atom_id) + 1 for atom_id in _CONSTRUCTION_REQUIRED_IDS[instance.family_id])
        observed_cap = next(cap for cap in REGISTERED_CAPS if worst_rank <= cap)
    payload = {
        "schema_version": "nip_v2_construction_certificate.v1",
        "family_id": instance.family_id,
        "target_atom_count": TARGET_ATOM_COUNT,
        "base_atom_count": base_count,
        "decoy_count": TARGET_ATOM_COUNT - base_count,
        "minimum_decoy_orthogonal_residual": float(np.min(residual_energies)),
        "maximum_decoy_orthogonality_error": float(np.max(np.abs(off_diagonal))),
        "declared_first_cap": expected_cap,
        "observed_first_cap": observed_cap,
        "cap_contract_pass": None if expected_cap is None else expected_cap == observed_cap,
    }
    return payload
