"""One-sided Minimum-Support Contribution Correspondence (MSCC).

This module scores a source atom against a finite, discovery-frozen family of
unweighted target-native supports.  It deliberately has no truth or held-out
evaluation input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from math import comb
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class NativeSupportCandidate:
    target_ids: tuple[int, ...]
    d_ctr: float
    d_mu: float


@dataclass(frozen=True)
class MSCCResult:
    status: str
    identification: str
    multiplicity: str | None
    minimum_support_size: int | None
    supports: tuple[NativeSupportCandidate, ...]
    best_candidate: NativeSupportCandidate | None
    nearest_competitor: NativeSupportCandidate | None
    planned_candidate_count: int
    evaluated_count: int
    candidate_budget: int
    complete_universe: bool
    unresolved_reason: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class SourceConditionedProposal:
    status: str
    source_atom_id: int
    proposed_target_ids: tuple[int, ...]
    singleton_d_ctr: tuple[float, ...]
    boundary_margin: float | None
    full_dictionary_comparisons: int
    planned_support_count: int
    proposal_hash: str
    refusal_reason: str | None


@dataclass(frozen=True)
class FrozenMSCCPrediction:
    schema_version: str
    protocol_hash: str
    proposal_hash: str
    discovery_fingerprint: str
    source_atom_id: int
    search_status: str
    identification: str
    multiplicity: str | None
    supports: tuple[NativeSupportCandidate, ...]
    prediction_hash: str


def _support_count(atom_count: int, g_max: int) -> int:
    return sum(comb(atom_count, size) for size in range(1, min(atom_count, g_max) + 1))


def source_conditioned_topk_proposal(
    k_source_source: np.ndarray,
    k_source_target: np.ndarray,
    k_target_target: np.ndarray,
    *,
    source_atom_id: int,
    atom_cap: int,
    g_max: int,
    epsilon: float,
    candidate_budget: int,
    boundary_tie_tolerance: float,
) -> SourceConditionedProposal:
    """Freeze a source-conditioned target atom family using discovery kernels only."""
    k_ss = np.asarray(k_source_source, dtype=np.float64)
    k_st = np.asarray(k_source_target, dtype=np.float64)
    k_tt = np.asarray(k_target_target, dtype=np.float64)
    if k_ss.ndim != 2 or k_st.ndim != 2 or k_tt.ndim != 2:
        raise ValueError("kernels must be rank-2")
    source_count, target_count = k_st.shape
    if k_ss.shape != (source_count, source_count) or k_tt.shape != (target_count, target_count):
        raise ValueError("kernel shapes are inconsistent")
    if not all(np.all(np.isfinite(value)) for value in (k_ss, k_st, k_tt)):
        raise ValueError("kernels must be finite")
    if not 0 <= source_atom_id < source_count:
        raise ValueError("source_atom_id is outside the source universe")
    if atom_cap < 1 or g_max < 1 or epsilon <= 0.0 or candidate_budget < 1 or boundary_tie_tolerance < 0.0:
        raise ValueError("caps/budget/epsilon must be positive and tie tolerance non-negative")
    source_energy = float(k_ss[source_atom_id, source_atom_id])
    if source_energy < 0.0:
        raise ValueError("source self-kernel energy must be non-negative")
    singleton_residuals: list[float] = []
    for target_id in range(target_count):
        target_energy = float(k_tt[target_id, target_id])
        numerator = source_energy + target_energy - 2.0 * float(k_st[source_atom_id, target_id])
        scale = max(1.0, abs(source_energy), abs(target_energy), 2.0 * abs(float(k_st[source_atom_id, target_id])))
        if numerator < -1e-10 * scale:
            raise ValueError("kernels imply a materially negative singleton squared residual")
        singleton_residuals.append(max(0.0, numerator) / (source_energy + epsilon))
    ranking = tuple(sorted(range(target_count), key=lambda target_id: (singleton_residuals[target_id], target_id)))
    selected_count = min(atom_cap, target_count)
    selected = tuple(sorted(ranking[:selected_count]))
    boundary_margin = None
    refusal_reason = None
    if target_count > atom_cap:
        boundary_margin = singleton_residuals[ranking[atom_cap]] - singleton_residuals[ranking[atom_cap - 1]]
        if boundary_margin <= boundary_tie_tolerance:
            selected = ()
            refusal_reason = "BOUNDARY_TIE"
    planned = _support_count(len(selected), g_max)
    if refusal_reason is None and planned > candidate_budget:
        selected = ()
        refusal_reason = "PLANNED_SUPPORT_COUNT_EXCEEDS_BUDGET"
    payload = {
        "schema_version": "source_conditioned_proposal.v1",
        "source_atom_id": source_atom_id,
        "atom_cap": atom_cap,
        "g_max": g_max,
        "candidate_budget": candidate_budget,
        "boundary_tie_tolerance": boundary_tie_tolerance,
        "ranking": ranking,
        "singleton_d_ctr": singleton_residuals,
        "proposed_target_ids": selected,
        "boundary_margin": boundary_margin,
        "planned_support_count": planned,
        "refusal_reason": refusal_reason,
    }
    proposal_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return SourceConditionedProposal(
        status="OK" if refusal_reason is None else "BUDGET_REFUSAL",
        source_atom_id=source_atom_id,
        proposed_target_ids=selected,
        singleton_d_ctr=tuple(singleton_residuals),
        boundary_margin=boundary_margin,
        full_dictionary_comparisons=target_count,
        planned_support_count=planned,
        proposal_hash=proposal_hash,
        refusal_reason=refusal_reason,
    )


def _frozen_prediction_payload(
    result: MSCCResult,
    *,
    protocol_hash: str,
    proposal_hash: str,
    discovery_fingerprint: str,
    source_atom_id: int,
) -> dict:
    return {
        "schema_version": "frozen_mscc_prediction.v1",
        "protocol_hash": protocol_hash,
        "proposal_hash": proposal_hash,
        "discovery_fingerprint": discovery_fingerprint,
        "source_atom_id": source_atom_id,
        "search_status": result.status,
        "identification": result.identification,
        "multiplicity": result.multiplicity,
        "supports": [(item.target_ids, item.d_ctr, item.d_mu) for item in result.supports],
        "planned_candidate_count": result.planned_candidate_count,
        "evaluated_count": result.evaluated_count,
        "complete_universe": result.complete_universe,
        "unresolved_reason": result.unresolved_reason,
    }


def freeze_mscc_prediction(
    result: MSCCResult,
    *,
    protocol_hash: str,
    proposal_hash: str,
    discovery_fingerprint: str,
    source_atom_id: int,
) -> FrozenMSCCPrediction:
    """Content-address an MSCC discovery output before any truth/eval access."""
    if not protocol_hash or not proposal_hash or not discovery_fingerprint:
        raise ValueError("protocol, proposal and discovery hashes are required")
    payload = _frozen_prediction_payload(
        result, protocol_hash=protocol_hash, proposal_hash=proposal_hash,
        discovery_fingerprint=discovery_fingerprint, source_atom_id=source_atom_id,
    )
    prediction_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return FrozenMSCCPrediction(
        schema_version="frozen_mscc_prediction.v1",
        protocol_hash=protocol_hash,
        proposal_hash=proposal_hash,
        discovery_fingerprint=discovery_fingerprint,
        source_atom_id=source_atom_id,
        search_status=result.status,
        identification=result.identification,
        multiplicity=result.multiplicity,
        supports=result.supports,
        prediction_hash=prediction_hash,
    )


def verify_frozen_mscc_prediction(frozen: FrozenMSCCPrediction, result: MSCCResult) -> bool:
    if frozen.schema_version != "frozen_mscc_prediction.v1":
        return False
    payload = _frozen_prediction_payload(
        result, protocol_hash=frozen.protocol_hash, proposal_hash=frozen.proposal_hash,
        discovery_fingerprint=frozen.discovery_fingerprint, source_atom_id=frozen.source_atom_id,
    )
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return (
        expected == frozen.prediction_hash
        and result.status == frozen.search_status
        and result.identification == frozen.identification
        and result.multiplicity == frozen.multiplicity
        and result.supports == frozen.supports
    )


def minimum_support_contribution_correspondence(
    k_source_source: np.ndarray,
    k_source_target: np.ndarray,
    k_target_target: np.ndarray,
    source_mean_contributions: np.ndarray,
    target_mean_contributions: np.ndarray,
    *,
    source_atom_id: int,
    proposed_target_ids: tuple[int, ...],
    g_max: int,
    tau_ctr: float,
    tau_mu: float,
    epsilon: float,
    candidate_budget: int,
    complete_universe: bool = False,
) -> MSCCResult:
    """Find every cardinality-minimal feasible native target support.

    Kernels and means must come from splits permitted by the caller's frozen
    protocol.  ``complete_universe`` is checked mechanically and controls
    whether an empty feasible set may be called ``CERTIFIED_ABSENT``. That
    certificate is global only when ``g_max`` covers the entire target atom
    universe; enumerating every atom but only bounded-size supports is not a
    complete support-universe search.
    """
    started = perf_counter()
    k_ss = np.asarray(k_source_source, dtype=np.float64)
    k_st = np.asarray(k_source_target, dtype=np.float64)
    k_tt = np.asarray(k_target_target, dtype=np.float64)
    mu_s = np.asarray(source_mean_contributions, dtype=np.float64)
    mu_t = np.asarray(target_mean_contributions, dtype=np.float64)
    if k_ss.ndim != 2 or k_st.ndim != 2 or k_tt.ndim != 2:
        raise ValueError("kernels must be rank-2")
    source_count, target_count = k_st.shape
    if k_ss.shape != (source_count, source_count) or k_tt.shape != (target_count, target_count):
        raise ValueError("kernel shapes are inconsistent")
    if not all(np.all(np.isfinite(value)) for value in (k_ss, k_st, k_tt, mu_s, mu_t)):
        raise ValueError("kernels and mean contributions must be finite")
    if not np.allclose(k_ss, k_ss.T, atol=1e-10, rtol=1e-10) or not np.allclose(k_tt, k_tt.T, atol=1e-10, rtol=1e-10):
        raise ValueError("self-kernels must be symmetric")
    if mu_s.ndim != 2 or mu_t.ndim != 2 or mu_s.shape[0] != mu_t.shape[0]:
        raise ValueError("mean contribution matrices must be rank-2 with a shared hook dimension")
    if mu_s.shape[1] != source_count or mu_t.shape[1] != target_count:
        raise ValueError("mean contribution feature counts must match kernels")
    if not 0 <= source_atom_id < source_count:
        raise ValueError("source_atom_id is outside the source universe")
    if g_max < 1 or tau_ctr < 0.0 or tau_mu < 0.0 or epsilon <= 0.0 or candidate_budget < 1:
        raise ValueError("g_max/budget/epsilon must be positive and thresholds non-negative")
    target_ids = tuple(int(value) for value in proposed_target_ids)
    if tuple(sorted(set(target_ids))) != target_ids:
        raise ValueError("proposed_target_ids must be sorted and unique")
    if target_ids and (target_ids[0] < 0 or target_ids[-1] >= target_count):
        raise ValueError("proposed target ID is outside the target universe")
    is_complete = target_ids == tuple(range(target_count))
    if complete_universe and not is_complete:
        raise ValueError("complete_universe requires every target atom exactly once")
    if complete_universe and g_max < target_count:
        raise ValueError("complete_universe requires g_max to cover the target atom universe")

    planned = _support_count(len(target_ids), g_max)
    if planned > candidate_budget:
        return MSCCResult(
            status="BUDGET_REFUSAL",
            identification="UNRESOLVED",
            multiplicity=None,
            minimum_support_size=None,
            supports=(),
            best_candidate=None,
            nearest_competitor=None,
            planned_candidate_count=planned,
            evaluated_count=0,
            candidate_budget=candidate_budget,
            complete_universe=complete_universe,
            unresolved_reason="BUDGET_REFUSAL",
            elapsed_seconds=perf_counter() - started,
        )

    source_energy = float(k_ss[source_atom_id, source_atom_id])
    if source_energy < 0.0:
        raise ValueError("source self-kernel energy must be non-negative")
    source_mean = mu_s[:, source_atom_id]
    ctr_denom = source_energy + epsilon
    mu_denom = float(source_mean @ source_mean) + epsilon
    scored: list[NativeSupportCandidate] = []
    feasible: list[NativeSupportCandidate] = []
    for size in range(1, min(len(target_ids), g_max) + 1):
        for support in combinations(target_ids, size):
            ids = np.asarray(support, dtype=int)
            target_energy = float(np.sum(k_tt[np.ix_(ids, ids)]))
            cross = float(np.sum(k_st[source_atom_id, ids]))
            if target_energy < -1e-10 * max(1.0, abs(target_energy)):
                raise ValueError("target self-kernel energy must be non-negative")
            raw_ctr_numerator = source_energy + target_energy - 2.0 * cross
            numerical_scale = max(1.0, abs(source_energy), abs(target_energy), 2.0 * abs(cross))
            if raw_ctr_numerator < -1e-10 * numerical_scale:
                raise ValueError("kernels imply a materially negative squared residual")
            ctr_numerator = max(0.0, raw_ctr_numerator)
            target_mean = np.sum(mu_t[:, ids], axis=1)
            mean_delta = source_mean - target_mean
            candidate = NativeSupportCandidate(
                target_ids=support,
                d_ctr=ctr_numerator / ctr_denom,
                d_mu=float(mean_delta @ mean_delta) / mu_denom,
            )
            scored.append(candidate)
            if candidate.d_ctr <= tau_ctr and candidate.d_mu <= tau_mu:
                feasible.append(candidate)

    ranked = sorted(scored, key=lambda item: (max(item.d_ctr / max(tau_ctr, epsilon), item.d_mu / max(tau_mu, epsilon)), len(item.target_ids), item.target_ids))
    best = ranked[0] if ranked else None
    if feasible:
        minimum_size = min(len(item.target_ids) for item in feasible)
        selected = tuple(sorted((item for item in feasible if len(item.target_ids) == minimum_size), key=lambda item: item.target_ids))
        selected_ids = {item.target_ids for item in selected}
        nearest_competitor = next((item for item in ranked if item.target_ids not in selected_ids), None)
        return MSCCResult(
            status="OK",
            identification="FOUND",
            multiplicity="UNIQUE" if len(selected) == 1 else "AMBIGUOUS",
            minimum_support_size=minimum_size,
            supports=selected,
            best_candidate=best,
            nearest_competitor=nearest_competitor,
            planned_candidate_count=planned,
            evaluated_count=planned,
            candidate_budget=candidate_budget,
            complete_universe=complete_universe,
            unresolved_reason=None,
            elapsed_seconds=perf_counter() - started,
        )

    return MSCCResult(
        status="OK",
        identification="CERTIFIED_ABSENT" if complete_universe else "UNRESOLVED",
        multiplicity=None,
        minimum_support_size=None,
        supports=(),
        best_candidate=best,
        nearest_competitor=best,
        planned_candidate_count=planned,
        evaluated_count=planned,
        candidate_budget=candidate_budget,
        complete_universe=complete_universe,
        unresolved_reason=None if complete_universe else "NO_ACCEPTED_IN_FROZEN_FAMILY",
        elapsed_seconds=perf_counter() - started,
    )
