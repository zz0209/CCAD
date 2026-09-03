"""Small exact oracles for truth-known CBSM neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from math import comb
from time import perf_counter

import numpy as np

from .metrics import bcc_from_kernels


@dataclass(frozen=True)
class BalancedCandidate:
    left_ids: tuple[int, ...]
    right_ids: tuple[int, ...]
    normalized_residual: float


@dataclass(frozen=True)
class ExhaustiveSearchResult:
    all_candidates: tuple[BalancedCandidate, ...]
    passing_candidates: tuple[BalancedCandidate, ...]
    support_minimal_candidates: tuple[BalancedCandidate, ...]
    best_residual: float | None
    second_best_residual: float | None
    solver_gap: float | None
    tie_set: tuple[BalancedCandidate, ...]
    evaluated_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ExactCoverResult:
    maximum_cardinality: int
    maximum_covers: tuple[tuple[BalancedCandidate, ...], ...]
    exact_cover_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FullUniverseSearchResult:
    status: str
    planned_candidate_count: int
    candidate_budget: int
    search: ExhaustiveSearchResult | None
    refusal_reason: str | None


@dataclass(frozen=True)
class CandidateFamilySearchResult:
    status: str
    candidate_family: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    passing_candidates: tuple[BalancedCandidate, ...]
    support_minimal_candidates: tuple[BalancedCandidate, ...]
    best_residual: float | None
    second_best_residual: float | None
    solver_gap: float | None
    tie_set: tuple[BalancedCandidate, ...]
    evaluated_count: int
    candidate_budget: int
    refusal_reason: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class FrozenDiscoveryPrediction:
    schema_version: str
    proposal_source: str
    proposal_hash: str
    discovery_fingerprint: str
    search_status: str
    candidate_family: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    predictions: tuple[BalancedCandidate, ...]
    prediction_hash: str


@dataclass(frozen=True)
class HeldOutHyperedgeEvaluation:
    proposal_recall: float
    precision: float
    recall: float
    f1: float
    eval_normalized_residuals: tuple[float | None, ...]
    failure_attribution: str | None


def _nonempty_subsets(ids: tuple[int, ...], max_size: int | None) -> list[tuple[int, ...]]:
    if max_size is not None and max_size < 1:
        raise ValueError("max_group_size must be positive when provided")
    limit = len(ids) if max_size is None else min(len(ids), max_size)
    return [subset for size in range(1, limit + 1) for subset in combinations(ids, size)]


def _subset_count(size: int, max_group_size: int | None) -> int:
    if max_group_size is not None and max_group_size < 1:
        raise ValueError("max_group_size must be positive when provided")
    limit = size if max_group_size is None else min(size, max_group_size)
    return sum(comb(size, group_size) for group_size in range(1, limit + 1))


def _validate_kernel_shapes(k_ll: np.ndarray, k_lr: np.ndarray, k_rr: np.ndarray) -> None:
    if k_ll.ndim != 2 or k_lr.ndim != 2 or k_rr.ndim != 2:
        raise ValueError("kernels must be rank-2")
    left_count, right_count = k_lr.shape
    if k_ll.shape != (left_count, left_count) or k_rr.shape != (right_count, right_count):
        raise ValueError("kernel shapes do not define one complete bipartite feature universe")


def exhaustive_balanced_pairs(
    k_ll: np.ndarray,
    k_lr: np.ndarray,
    k_rr: np.ndarray,
    left_pool: tuple[int, ...],
    right_pool: tuple[int, ...],
    *,
    residual_tolerance: float,
    max_group_size: int | None = None,
    support_minimal_only: bool = True,
) -> list[BalancedCandidate]:
    """Enumerate a bounded neighborhood; intended only as a correctness oracle."""
    result = exhaustive_balanced_search(
        k_ll,
        k_lr,
        k_rr,
        left_pool,
        right_pool,
        residual_tolerance=residual_tolerance,
        tie_tolerance=0.0,
        max_group_size=max_group_size,
    )
    selected = result.support_minimal_candidates if support_minimal_only else result.passing_candidates
    return list(selected)


def exhaustive_balanced_search(
    k_ll: np.ndarray,
    k_lr: np.ndarray,
    k_rr: np.ndarray,
    left_pool: tuple[int, ...],
    right_pool: tuple[int, ...],
    *,
    residual_tolerance: float,
    tie_tolerance: float,
    max_group_size: int | None = None,
) -> ExhaustiveSearchResult:
    """Return the complete finite candidate family and auditable solver diagnostics."""
    if residual_tolerance < 0.0 or tie_tolerance < 0.0:
        raise ValueError("residual and tie tolerances must be non-negative")
    started = perf_counter()
    evaluated: list[BalancedCandidate] = []
    attempted_count = 0
    for left in _nonempty_subsets(left_pool, max_group_size):
        for right in _nonempty_subsets(right_pool, max_group_size):
            attempted_count += 1
            result = bcc_from_kernels(
                k_ll,
                k_lr,
                k_rr,
                np.asarray(left, dtype=int),
                np.asarray(right, dtype=int),
            )
            if result.status == "OK" and result.normalized_residual is not None:
                evaluated.append(BalancedCandidate(left, right, result.normalized_residual))
    ranked = sorted(evaluated, key=lambda item: (item.normalized_residual, len(item.left_ids) + len(item.right_ids), item.left_ids, item.right_ids))
    passing = [candidate for candidate in ranked if candidate.normalized_residual <= residual_tolerance]
    minimal: list[BalancedCandidate] = []
    for candidate in passing:
        lset = set(candidate.left_ids)
        rset = set(candidate.right_ids)
        has_proper = any(
            set(other.left_ids).issubset(lset)
            and set(other.right_ids).issubset(rset)
            and (other.left_ids != candidate.left_ids or other.right_ids != candidate.right_ids)
            for other in passing
        )
        if not has_proper:
            minimal.append(candidate)
    minimal.sort(key=lambda item: (len(item.left_ids) + len(item.right_ids), item.left_ids, item.right_ids))
    best = ranked[0].normalized_residual if ranked else None
    second = ranked[1].normalized_residual if len(ranked) > 1 else None
    ties = tuple(candidate for candidate in ranked if best is not None and candidate.normalized_residual <= best + tie_tolerance)
    return ExhaustiveSearchResult(
        all_candidates=tuple(ranked),
        passing_candidates=tuple(passing),
        support_minimal_candidates=tuple(minimal),
        best_residual=best,
        second_best_residual=second,
        solver_gap=None if best is None or second is None else second - best,
        tie_set=ties,
        evaluated_count=attempted_count,
        elapsed_seconds=perf_counter() - started,
    )


def full_universe_balanced_search(
    k_ll: np.ndarray,
    k_lr: np.ndarray,
    k_rr: np.ndarray,
    *,
    residual_tolerance: float,
    tie_tolerance: float,
    max_group_size: int | None,
    candidate_budget: int,
) -> FullUniverseSearchResult:
    """Truth-blind exact reference over all feature IDs, with fail-closed budgeting."""
    _validate_kernel_shapes(k_ll, k_lr, k_rr)
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    left_count, right_count = k_lr.shape
    planned = _subset_count(left_count, max_group_size) * _subset_count(right_count, max_group_size)
    if planned > candidate_budget:
        return FullUniverseSearchResult(
            status="BUDGET_REFUSAL",
            planned_candidate_count=planned,
            candidate_budget=candidate_budget,
            search=None,
            refusal_reason="PLANNED_CANDIDATE_COUNT_EXCEEDS_BUDGET",
        )
    search = exhaustive_balanced_search(
        k_ll,
        k_lr,
        k_rr,
        tuple(range(left_count)),
        tuple(range(right_count)),
        residual_tolerance=residual_tolerance,
        tie_tolerance=tie_tolerance,
        max_group_size=max_group_size,
    )
    if search.evaluated_count != planned:
        raise RuntimeError("full-universe search did not evaluate the declared candidate family")
    return FullUniverseSearchResult(
        status="OK",
        planned_candidate_count=planned,
        candidate_budget=candidate_budget,
        search=search,
        refusal_reason=None,
    )


def search_candidate_family(
    k_ll: np.ndarray,
    k_lr: np.ndarray,
    k_rr: np.ndarray,
    candidate_family: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    *,
    residual_tolerance: float,
    tie_tolerance: float,
    candidate_budget: int,
) -> CandidateFamilySearchResult:
    """Score a pre-proposed finite family without access to truth or held-out data."""
    _validate_kernel_shapes(k_ll, k_lr, k_rr)
    if residual_tolerance < 0.0 or tie_tolerance < 0.0 or candidate_budget < 1:
        raise ValueError("tolerances must be non-negative and candidate_budget positive")
    canonical = tuple(sorted(set(candidate_family), key=lambda item: (len(item[0]) + len(item[1]), item[0], item[1])))
    left_count, right_count = k_lr.shape
    for left_ids, right_ids in canonical:
        if not left_ids or not right_ids:
            raise ValueError("candidate groups must be nonempty")
        if tuple(sorted(set(left_ids))) != left_ids or tuple(sorted(set(right_ids))) != right_ids:
            raise ValueError("candidate IDs must be sorted and unique")
        if min(left_ids) < 0 or max(left_ids) >= left_count or min(right_ids) < 0 or max(right_ids) >= right_count:
            raise ValueError("candidate ID is outside the kernel universe")
    if len(canonical) > candidate_budget:
        return CandidateFamilySearchResult(
            status="BUDGET_REFUSAL",
            candidate_family=canonical,
            passing_candidates=(),
            support_minimal_candidates=(),
            best_residual=None,
            second_best_residual=None,
            solver_gap=None,
            tie_set=(),
            evaluated_count=0,
            candidate_budget=candidate_budget,
            refusal_reason="CANDIDATE_FAMILY_EXCEEDS_BUDGET",
            elapsed_seconds=0.0,
        )
    started = perf_counter()
    scored: list[BalancedCandidate] = []
    for left_ids, right_ids in canonical:
        result = bcc_from_kernels(
            k_ll, k_lr, k_rr,
            np.asarray(left_ids, dtype=int), np.asarray(right_ids, dtype=int),
        )
        if result.status == "OK" and result.normalized_residual is not None:
            scored.append(BalancedCandidate(left_ids, right_ids, result.normalized_residual))
    ranked = sorted(scored, key=lambda item: (item.normalized_residual, len(item.left_ids) + len(item.right_ids), item.left_ids, item.right_ids))
    passing = tuple(item for item in ranked if item.normalized_residual <= residual_tolerance)
    minimal = tuple(item for item in passing if not any(
        set(other.left_ids).issubset(item.left_ids)
        and set(other.right_ids).issubset(item.right_ids)
        and (other.left_ids != item.left_ids or other.right_ids != item.right_ids)
        for other in passing
    ))
    best = ranked[0].normalized_residual if ranked else None
    second = ranked[1].normalized_residual if len(ranked) > 1 else None
    ties = tuple(item for item in ranked if best is not None and item.normalized_residual <= best + tie_tolerance)
    return CandidateFamilySearchResult(
        status="OK",
        candidate_family=canonical,
        passing_candidates=passing,
        support_minimal_candidates=minimal,
        best_residual=best,
        second_best_residual=second,
        solver_gap=None if best is None or second is None else second - best,
        tie_set=ties,
        evaluated_count=len(canonical),
        candidate_budget=candidate_budget,
        refusal_reason=None,
        elapsed_seconds=perf_counter() - started,
    )


def _prediction_payload(
    *,
    proposal_source: str,
    proposal_hash: str,
    discovery_fingerprint: str,
    search_status: str,
    candidate_family: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    predictions: tuple[BalancedCandidate, ...],
) -> dict:
    return {
        "schema_version": "frozen_discovery_prediction.v1",
        "proposal_source": proposal_source,
        "proposal_hash": proposal_hash,
        "discovery_fingerprint": discovery_fingerprint,
        "search_status": search_status,
        "candidate_family": candidate_family,
        "predictions": [(item.left_ids, item.right_ids, item.normalized_residual) for item in predictions],
    }


def freeze_discovery_prediction(
    search: CandidateFamilySearchResult,
    *,
    proposal_source: str,
    proposal_hash: str,
    discovery_fingerprint: str,
) -> FrozenDiscoveryPrediction:
    """Create a content-addressed record that must exist before held-out evaluation."""
    if not proposal_source or not proposal_hash or not discovery_fingerprint:
        raise ValueError("proposal source/hash and discovery fingerprint are required")
    payload = _prediction_payload(
        proposal_source=proposal_source,
        proposal_hash=proposal_hash,
        discovery_fingerprint=discovery_fingerprint,
        search_status=search.status,
        candidate_family=search.candidate_family,
        predictions=search.support_minimal_candidates,
    )
    prediction_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return FrozenDiscoveryPrediction(
        schema_version="frozen_discovery_prediction.v1",
        proposal_source=proposal_source,
        proposal_hash=proposal_hash,
        discovery_fingerprint=discovery_fingerprint,
        search_status=search.status,
        candidate_family=search.candidate_family,
        predictions=search.support_minimal_candidates,
        prediction_hash=prediction_hash,
    )


def verify_frozen_discovery_prediction(frozen: FrozenDiscoveryPrediction) -> bool:
    payload = _prediction_payload(
        proposal_source=frozen.proposal_source,
        proposal_hash=frozen.proposal_hash,
        discovery_fingerprint=frozen.discovery_fingerprint,
        search_status=frozen.search_status,
        candidate_family=frozen.candidate_family,
        predictions=frozen.predictions,
    )
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return frozen.schema_version == "frozen_discovery_prediction.v1" and frozen.prediction_hash == expected


def evaluate_frozen_hyperedges(
    frozen: FrozenDiscoveryPrediction,
    k_ll_eval: np.ndarray,
    k_lr_eval: np.ndarray,
    k_rr_eval: np.ndarray,
    planted_hyperedges: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> HeldOutHyperedgeEvaluation:
    """Evaluate only a valid content-addressed prediction against held-out synthetic truth."""
    if not verify_frozen_discovery_prediction(frozen):
        raise ValueError("held-out evaluation requires a valid frozen discovery prediction")
    _validate_kernel_shapes(k_ll_eval, k_lr_eval, k_rr_eval)
    truth = set(planted_hyperedges)
    candidate_set = set(frozen.candidate_family)
    predicted = {(item.left_ids, item.right_ids) for item in frozen.predictions}
    proposal_recall = 1.0 if not truth else len(candidate_set & truth) / len(truth)
    true_positive = len(predicted & truth)
    precision = 1.0 if not predicted and not truth else (true_positive / len(predicted) if predicted else 0.0)
    recall = 1.0 if not truth else true_positive / len(truth)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    eval_residuals = []
    for item in frozen.predictions:
        result = bcc_from_kernels(
            k_ll_eval, k_lr_eval, k_rr_eval,
            np.asarray(item.left_ids, dtype=int), np.asarray(item.right_ids, dtype=int),
        )
        eval_residuals.append(result.normalized_residual)
    if frozen.search_status == "BUDGET_REFUSAL":
        attribution = "BUDGET_REFUSAL"
    elif proposal_recall < 1.0:
        attribution = "PROPOSAL_MISS"
    elif predicted != truth:
        attribution = "SOLVER_MISS"
    else:
        attribution = None
    return HeldOutHyperedgeEvaluation(
        proposal_recall=proposal_recall,
        precision=precision,
        recall=recall,
        f1=f1,
        eval_normalized_residuals=tuple(eval_residuals),
        failure_attribution=attribution,
    )


def forced_partition_projection(candidates: tuple[BalancedCandidate, ...]) -> tuple[BalancedCandidate, ...]:
    """Negative-control projection that greedily deletes any overlapping hyperedge."""
    ranked = sorted(candidates, key=lambda item: (item.normalized_residual, item.left_ids, item.right_ids))
    selected: list[BalancedCandidate] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for candidate in ranked:
        if used_left.isdisjoint(candidate.left_ids) and used_right.isdisjoint(candidate.right_ids):
            selected.append(candidate)
            used_left.update(candidate.left_ids)
            used_right.update(candidate.right_ids)
    return tuple(selected)


def enumerate_maximum_exact_covers(
    candidates: tuple[BalancedCandidate, ...],
    left_universe: tuple[int, ...],
    right_universe: tuple[int, ...],
    *,
    candidate_limit: int = 24,
) -> ExactCoverResult:
    """Enumerate all exact covers for a small truth-known hypergraph and retain every maximum cover."""
    if len(candidates) > candidate_limit:
        raise ValueError("exact-cover oracle candidate limit exceeded")
    started = perf_counter()
    target_left = set(left_universe)
    target_right = set(right_universe)
    ordered = tuple(sorted(candidates, key=lambda item: (item.left_ids, item.right_ids, item.normalized_residual)))
    exact_covers: list[tuple[BalancedCandidate, ...]] = []
    for size in range(1, len(ordered) + 1):
        for subset in combinations(ordered, size):
            left_sets = [set(item.left_ids) for item in subset]
            right_sets = [set(item.right_ids) for item in subset]
            if any(left_sets[i] & left_sets[j] or right_sets[i] & right_sets[j]
                   for i in range(size) for j in range(i + 1, size)):
                continue
            if set().union(*left_sets) == target_left and set().union(*right_sets) == target_right:
                exact_covers.append(subset)
    maximum = max((len(cover) for cover in exact_covers), default=0)
    maximum_covers = tuple(cover for cover in exact_covers if len(cover) == maximum)
    return ExactCoverResult(
        maximum_cardinality=maximum,
        maximum_covers=maximum_covers,
        exact_cover_count=len(exact_covers),
        elapsed_seconds=perf_counter() - started,
    )
