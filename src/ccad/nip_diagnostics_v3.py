"""Truth-free orthogonal diagnostics for frozen native-support predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccad.nip_synthetic_v3 import NIPV3ObservedInstance, evaluate_shared_hook_endpoint


@dataclass(frozen=True)
class NIPOrthogonalDiagnostics:
    target_ids: tuple[int, ...]
    cancellation_energy_ratio: float | None
    aggregate_target_energy: float
    source_active_token_count: int
    source_active_document_count: int
    source_token_energy_kish_ess: float
    source_document_energy_kish_ess: float
    d_mu: float
    endpoint: dict[str, float] | None


def _kish(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights * weights))
    return 0.0 if denominator == 0.0 else float(np.sum(weights) ** 2 / denominator)


def evaluate_orthogonal_diagnostics(
    instance: NIPV3ObservedInstance,
    target_ids: tuple[int, ...],
    *,
    epsilon: float = 1e-12,
    activity_atol: float = 1e-15,
) -> NIPOrthogonalDiagnostics:
    """Recompute safety/evidence/mean/endpoint metrics from observed tensors.

    The caller must supply the content-addressed predicted support (or a
    separately declared diagnostic candidate).  This function never selects a
    support and never reads a truth registry or family-specific outcome label.
    """
    ids = tuple(int(value) for value in target_ids)
    if not ids or tuple(sorted(set(ids))) != ids:
        raise ValueError("target_ids must be nonempty, sorted, and unique")
    if epsilon <= 0.0 or activity_atol < 0.0:
        raise ValueError("epsilon must be positive and activity_atol non-negative")
    target_count = instance.target_contributions.shape[1]
    if ids[0] < 0 or ids[-1] >= target_count:
        raise ValueError("target support is outside the observed universe")

    selected = instance.target_contributions[:, ids, :]
    individual_energy = np.mean(np.sum(selected * selected, axis=2), axis=0)
    aggregate = np.sum(selected, axis=1)
    aggregate_energy = float(np.mean(np.sum(aggregate * aggregate, axis=1)))
    cancellation = None
    if aggregate_energy > activity_atol:
        cancellation = float(np.sum(individual_energy) / aggregate_energy)

    source = instance.source_contributions[:, instance.source_atom_id, :]
    token_energy = np.sum(source * source, axis=1)
    documents = np.asarray(instance.document_ids).reshape(-1)
    if documents.size != token_energy.size:
        raise ValueError("document ids must align with contribution observations")
    unique_documents = np.unique(documents)
    document_energy = np.asarray([np.sum(token_energy[documents == doc]) for doc in unique_documents])

    source_mean = instance.source_mean_contributions[:, instance.source_atom_id]
    target_mean = np.sum(instance.target_mean_contributions[:, ids], axis=1)
    mean_delta = source_mean - target_mean
    d_mu = float(mean_delta @ mean_delta) / (float(source_mean @ source_mean) + epsilon)
    endpoint = evaluate_shared_hook_endpoint(instance, ids) if instance.endpoint is not None else None
    return NIPOrthogonalDiagnostics(
        target_ids=ids,
        cancellation_energy_ratio=cancellation,
        aggregate_target_energy=aggregate_energy,
        source_active_token_count=int(np.sum(token_energy > activity_atol)),
        source_active_document_count=int(np.sum(document_energy > activity_atol)),
        source_token_energy_kish_ess=_kish(token_energy),
        source_document_energy_kish_ess=_kish(document_energy),
        d_mu=d_mu,
        endpoint=endpoint,
    )
