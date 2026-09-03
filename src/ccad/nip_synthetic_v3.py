"""Prospective v3 NIP construction with an observable causal endpoint.

V2 remains immutable history.  This module changes only N11: its native
target atom is close, but not pointwise equal, to the source contribution.
The shared-hook endpoint is fully observed and evaluated without consulting
``ccad.nip_truth``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ccad.nip_synthetic import NIPObservedInstance, generate_nip_observed
from ccad.nip_synthetic_v2 import DECOY_ORTHOGONAL_ENERGY, TARGET_ATOM_COUNT
from ccad.nip_synthetic_v2 import _decoy_residual_schedule, _orthogonal_decoy_basis
from ccad.nip_synthetic_v2 import construction_certificate as v2_construction_certificate
from ccad.nip_synthetic_v2 import generate_cap_identifiable_observed


N11_CENTERED_RESIDUAL = 0.01
N11_FEASIBILITY_THRESHOLD = 0.05
N11_MINIMUM_THRESHOLD_MARGIN = 0.03
N11_CLIFF_NORMALIZED_MARGIN = 0.05
N11_MINIMUM_CLIFF_GAP = 1.0
N11_MAXIMUM_SMOOTH_RMSE = 0.11


@dataclass(frozen=True)
class SharedHookEndpoint:
    """Observed deterministic downstream endpoint at a shared hook."""

    base_hook_states: np.ndarray  # observations x hook dim
    cliff_direction: np.ndarray  # hook dim
    cliff_threshold: float
    smooth_scale: float


@dataclass(frozen=True)
class NIPV3ObservedInstance:
    family_id: str
    source_contributions: np.ndarray
    target_contributions: np.ndarray
    source_mean_contributions: np.ndarray
    target_mean_contributions: np.ndarray
    document_ids: np.ndarray
    source_atom_id: int = 0
    endpoint: SharedHookEndpoint | None = None


def _upgrade(instance: NIPObservedInstance, endpoint: SharedHookEndpoint | None = None) -> NIPV3ObservedInstance:
    return NIPV3ObservedInstance(
        family_id=instance.family_id,
        source_contributions=instance.source_contributions,
        target_contributions=instance.target_contributions,
        source_mean_contributions=instance.source_mean_contributions,
        target_mean_contributions=instance.target_mean_contributions,
        document_ids=instance.document_ids,
        source_atom_id=instance.source_atom_id,
        endpoint=endpoint,
    )


def generate_endpoint_observed(
    family_id: str, *, structural_seed: int, sample_seed: int, n: int = 512
) -> NIPV3ObservedInstance:
    """Return the v3 observed construction without truth or outcome labels."""
    if family_id != "N11_downstream_cliff":
        base = generate_cap_identifiable_observed(
            family_id, structural_seed=structural_seed, sample_seed=sample_seed, n=n
        )
        return _upgrade(base)
    base = generate_nip_observed(
        family_id, structural_seed=structural_seed, sample_seed=sample_seed, n=n
    )
    if n % 2:
        raise ValueError("N11 requires an even observation count for exact zero-mean perturbations")
    if base.source_contributions.shape[2] != 1:
        raise ValueError("N11 v3 endpoint is frozen for one-dimensional hook fixtures")

    source = base.source_contributions[:, base.source_atom_id, :]
    source_rms = float(np.sqrt(np.mean(source * source)))
    if not np.isfinite(source_rms) or source_rms <= 0.0:
        raise ValueError("N11 source contribution must have finite positive RMS")

    signs = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    rng = np.random.default_rng(structural_seed ^ 0xC024C024)
    signs = signs[rng.permutation(n)].reshape(n, 1)
    amplitude = np.sqrt(N11_CENTERED_RESIDUAL) * source_rms
    delta = amplitude * signs

    target = np.array(base.target_contributions, copy=True)
    target[:, 0, :] = source + delta
    perturbed_base = replace(base, target_contributions=target)

    # Generate v2-style decoys only after the N11 target is perturbed.  Their
    # orthogonal complement must be defined against the actual v3 span.
    decoy_count = TARGET_ATOM_COUNT - target.shape[1]
    basis = _orthogonal_decoy_basis(perturbed_base, decoy_count, structural_seed ^ 0xC023C023)
    residuals = _decoy_residual_schedule(perturbed_base, decoy_count)
    beta = np.sqrt(DECOY_ORTHOGONAL_ENERGY)
    source_flat = source.reshape(-1)
    source_norm = float(np.linalg.norm(source_flat))
    decoys = []
    for column, residual in enumerate(residuals):
        alpha = 1.0 - np.sqrt(float(residual) - DECOY_ORTHOGONAL_ENERGY)
        flat = alpha * source_flat + beta * source_norm * basis[:, column]
        decoys.append(flat.reshape(n, source.shape[1]))
    target = np.concatenate((target, np.stack(decoys, axis=1)), axis=1)
    target_mean = np.concatenate(
        (base.target_mean_contributions, np.zeros((base.target_mean_contributions.shape[0], decoy_count))),
        axis=1,
    )
    midpoint = source + 0.5 * delta
    endpoint = SharedHookEndpoint(
        base_hook_states=midpoint,
        cliff_direction=np.asarray([1.0], dtype=np.float64),
        cliff_threshold=0.0,
        smooth_scale=1.0 / source_rms,
    )
    changed = replace(base, target_contributions=target, target_mean_contributions=target_mean)
    return _upgrade(changed, endpoint)


def evaluate_shared_hook_endpoint(instance: NIPV3ObservedInstance, target_ids: tuple[int, ...]) -> dict[str, float]:
    """Measure the frozen endpoint using tensors only, never a family label."""
    endpoint = instance.endpoint
    if endpoint is None:
        raise ValueError("instance has no shared-hook endpoint")
    if not target_ids:
        raise ValueError("target support must be nonempty")
    source = instance.source_contributions[:, instance.source_atom_id, :]
    target = np.sum(instance.target_contributions[:, target_ids, :], axis=1)
    source_ablated = endpoint.base_hook_states - source
    target_ablated = endpoint.base_hook_states - target
    direction = endpoint.cliff_direction
    source_projection = source_ablated @ direction - endpoint.cliff_threshold
    target_projection = target_ablated @ direction - endpoint.cliff_threshold
    if np.any(source_projection == 0.0) or np.any(target_projection == 0.0):
        raise RuntimeError("cliff endpoint encountered a boundary tie")
    source_cliff = (source_projection >= 0.0).astype(np.float64)
    target_cliff = (target_projection >= 0.0).astype(np.float64)
    cliff_difference = source_cliff - target_cliff
    smooth_difference = endpoint.smooth_scale * (source_projection - target_projection)
    source_rms = float(np.sqrt(np.mean(source * source)))
    raw_delta = source - target
    return {
        "target_support_size": float(len(target_ids)),
        "source_rms": source_rms,
        "raw_delta_rmse": float(np.sqrt(np.mean(raw_delta * raw_delta))),
        "minimum_raw_cliff_margin": float(
            min(np.min(np.abs(source_projection)), np.min(np.abs(target_projection)))
        ),
        "cliff_disagreement_rate": float(np.mean(source_cliff != target_cliff)),
        "cliff_effect_rmse": float(np.sqrt(np.mean(cliff_difference * cliff_difference))),
        "minimum_normalized_cliff_margin": float(
            min(np.min(np.abs(source_projection)), np.min(np.abs(target_projection))) / source_rms
        ),
        "smooth_effect_rmse": float(np.sqrt(np.mean(smooth_difference * smooth_difference))),
    }


def construction_certificate(instance: NIPV3ObservedInstance) -> dict[str, object]:
    """Return construction invariants and measured endpoint values, no truth."""
    legacy = NIPObservedInstance(
        family_id=instance.family_id,
        source_contributions=instance.source_contributions,
        target_contributions=instance.target_contributions,
        source_mean_contributions=instance.source_mean_contributions,
        target_mean_contributions=instance.target_mean_contributions,
        document_ids=instance.document_ids,
        source_atom_id=instance.source_atom_id,
    )
    payload = dict(v2_construction_certificate(legacy))
    payload["schema_version"] = "nip_v3_construction_certificate.v1"
    payload["endpoint_present"] = instance.endpoint is not None
    if instance.endpoint is None:
        payload["n11_centered_residual"] = None
        payload["n11_endpoint"] = None
        return payload

    source = instance.source_contributions[:, instance.source_atom_id, :]
    target = instance.target_contributions[:, 0, :]
    denominator = float(np.sum(source * source))
    centered_residual = float(np.sum((source - target) ** 2) / denominator)
    endpoint_metrics = evaluate_shared_hook_endpoint(instance, (0,))
    payload["n11_centered_residual"] = centered_residual
    payload["n11_sample_mean_delta_norm"] = float(np.linalg.norm(np.mean(target - source, axis=0)))
    payload["n11_endpoint"] = endpoint_metrics
    return payload
