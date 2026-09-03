"""Truth-free orthogonal diagnostics for frozen native-support predictions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

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


@dataclass(frozen=True)
class CenteredOnlyCandidate:
    target_ids: tuple[int, ...]
    d_ctr: float
    evaluated_count: int
    candidate_hash: str


def _kish(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights * weights))
    return 0.0 if denominator == 0.0 else float(np.sum(weights) ** 2 / denominator)


def freeze_centered_only_candidate(
    k_source_source: np.ndarray,
    k_source_target: np.ndarray,
    k_target_target: np.ndarray,
    *,
    source_atom_id: int,
    proposed_target_ids: tuple[int, ...],
    g_max: int,
    epsilon: float,
    candidate_budget: int,
) -> CenteredOnlyCandidate:
    """Freeze the best contribution-only candidate before any mean/truth check."""
    import hashlib
    import json

    k_ss = np.asarray(k_source_source, dtype=np.float64)
    k_st = np.asarray(k_source_target, dtype=np.float64)
    k_tt = np.asarray(k_target_target, dtype=np.float64)
    ids = tuple(int(value) for value in proposed_target_ids)
    if tuple(sorted(set(ids))) != ids or not ids:
        raise ValueError("proposed_target_ids must be nonempty, sorted, and unique")
    if g_max < 1 or epsilon <= 0.0 or candidate_budget < 1:
        raise ValueError("g_max, epsilon, and candidate_budget must be positive")
    source_count, target_count = k_st.shape
    if k_ss.shape != (source_count, source_count) or k_tt.shape != (target_count, target_count):
        raise ValueError("kernel shapes are inconsistent")
    if not 0 <= source_atom_id < source_count or ids[0] < 0 or ids[-1] >= target_count:
        raise ValueError("source or target id is outside the kernel universe")
    planned = sum(comb(len(ids), size) for size in range(1, min(g_max, len(ids)) + 1))
    if planned > candidate_budget:
        raise ValueError("centered-only candidate family exceeds budget")
    source_energy = float(k_ss[source_atom_id, source_atom_id])
    scored: list[tuple[float, int, tuple[int, ...]]] = []
    for size in range(1, min(g_max, len(ids)) + 1):
        for support in combinations(ids, size):
            selected = np.asarray(support, dtype=int)
            target_energy = float(np.sum(k_tt[np.ix_(selected, selected)]))
            cross = float(np.sum(k_st[source_atom_id, selected]))
            numerator = source_energy + target_energy - 2.0 * cross
            scale = max(1.0, abs(source_energy), abs(target_energy), 2.0 * abs(cross))
            if numerator < -1e-10 * scale:
                raise ValueError("kernels imply a materially negative centered residual")
            scored.append((max(0.0, numerator) / (source_energy + epsilon), size, support))
    d_ctr, _, target_ids = min(scored, key=lambda row: (row[0], row[1], row[2]))
    payload = {
        "schema_version": "centered_only_candidate.v1",
        "source_atom_id": source_atom_id,
        "proposed_target_ids": ids,
        "g_max": g_max,
        "candidate_budget": candidate_budget,
        "target_ids": target_ids,
        "d_ctr": d_ctr,
        "evaluated_count": planned,
    }
    candidate_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()
    return CenteredOnlyCandidate(target_ids, d_ctr, planned, candidate_hash)


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
