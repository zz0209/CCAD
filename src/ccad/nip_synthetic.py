"""D0-scale synthetic registry for native intervention portability.

This module contains observed tensors only. Scoring truth lives in the
separate ``ccad.nip_truth`` module and must not be imported by a D0 runner.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


FAMILIES = (
    "N01_structured_split", "N02_structured_merge_refactorization",
    "N03_tied_native_supports", "N04_absent_target", "N05_bloated_decoy",
    "N06_exact_dense_orthogonal_rotation", "N07_margin_separated_approximate_rotation",
    "N08_continuous_only_representation", "N09_cancellation", "N10_rare_occupancy",
    "N11_downstream_cliff", "N12_mean_mismatch",
)


@dataclass(frozen=True)
class NIPObservedInstance:
    family_id: str
    source_contributions: np.ndarray  # observations x source atoms x hook dim
    target_contributions: np.ndarray  # observations x target atoms x hook dim
    source_mean_contributions: np.ndarray  # hook dim x source atoms
    target_mean_contributions: np.ndarray  # hook dim x target atoms
    document_ids: np.ndarray
    source_atom_id: int = 0


def _stack(*atoms: np.ndarray) -> np.ndarray:
    return np.stack(atoms, axis=1)


def generate_nip_observed(family_id: str, *, structural_seed: int, sample_seed: int, n: int = 512) -> NIPObservedInstance:
    """Generate observed tensors only; callers enforce phase-specific seed policy."""
    if family_id not in FAMILIES or n < 32:
        raise ValueError("unknown family or insufficient observations")
    structural_rng = np.random.default_rng(structural_seed)
    rng = np.random.default_rng(sample_seed)
    x = structural_rng.uniform(0.8, 1.2) * rng.standard_normal((n, 1))
    y = structural_rng.uniform(0.8, 1.2) * rng.standard_normal((n, 1))
    docs = np.repeat(np.arange((n + 7) // 8), 8)[:n]
    source = _stack(x)
    target_mean = None
    source_mean = None
    if family_id == "N01_structured_split":
        target = _stack(0.4 * x, 0.6 * x, y)
    elif family_id == "N02_structured_merge_refactorization":
        target = _stack(0.5 * x + y, 0.3 * x - y, 0.2 * x)
    elif family_id == "N03_tied_native_supports":
        target = _stack(0.25 * x, 0.75 * x, 0.4 * x, 0.6 * x)
    elif family_id == "N04_absent_target":
        target = _stack(2.0 * x, y)
    elif family_id == "N05_bloated_decoy":
        target = _stack(0.4 * x, 0.6 * x, y, -y)
    elif family_id in {"N06_exact_dense_orthogonal_rotation", "N07_margin_separated_approximate_rotation"}:
        x2 = rng.standard_normal((n, 2))
        angle = structural_rng.uniform(0.55, 1.0) if family_id.startswith("N06") else structural_rng.uniform(0.28, 0.38)
        q = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        source = _stack(x2[:, [0]] * np.asarray([1.0, 0.0]))
        coords = x2 @ q
        target = _stack(*(coords[:, [j]] * q[:, j] for j in range(2)))
    elif family_id == "N08_continuous_only_representation":
        target = _stack(2.0 * x, y)
    elif family_id == "N09_cancellation":
        target = _stack(x + 10.0 * y, -10.0 * y)
    elif family_id == "N10_rare_occupancy":
        rare = np.zeros_like(x)
        rare[:16] = x[:16]
        source = _stack(rare)
        target = _stack(0.4 * rare, 0.6 * rare, y)
    elif family_id == "N11_downstream_cliff":
        target = _stack(x)
    else:  # N12_mean_mismatch
        target = _stack(x)
        source_mean = np.asarray([[1.0]])
        target_mean = np.asarray([[2.0]])

    hook_dim = source.shape[2]
    if source_mean is None:
        source_mean = np.zeros((hook_dim, source.shape[1]))
    if target_mean is None:
        target_mean = np.zeros((hook_dim, target.shape[1]))
    observed = NIPObservedInstance(
        family_id=family_id,
        source_contributions=source,
        target_contributions=target,
        source_mean_contributions=source_mean,
        target_mean_contributions=target_mean,
        document_ids=docs,
    )
    return observed


def observed_kernels(instance: NIPObservedInstance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = instance.source_contributions.transpose(1, 0, 2).reshape(instance.source_contributions.shape[1], -1)
    target = instance.target_contributions.transpose(1, 0, 2).reshape(instance.target_contributions.shape[1], -1)
    n = instance.source_contributions.shape[0]
    return source @ source.T / n, source @ target.T / n, target @ target.T / n


def assert_observed_schema_truth_free() -> None:
    forbidden = {"truth", "label", "planted_support", "minimum_supports", "identification", "causal_outcome"}
    names = {field.name for field in fields(NIPObservedInstance)}
    if names & forbidden:
        raise RuntimeError(f"observed schema leaks truth fields: {sorted(names & forbidden)}")
