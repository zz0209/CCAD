"""Truth-blind proposal primitives for corrective M1 discovery tests."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


REQUIRED_SPLIT_SEEDS = (
    "structural_seed_a",
    "structural_seed_b",
    "mean_sample_seed",
    "discovery_sample_seed",
    "eval_sample_seed",
)


@dataclass(frozen=True)
class ProposalNeighborhood:
    anchor_left: int
    anchor_right: int
    left_ids: tuple[int, ...]
    right_ids: tuple[int, ...]
    status: str
    refusal_reason: str | None


@dataclass(frozen=True)
class BipartiteProposal:
    score_source: str
    top_k: int
    edges: tuple[tuple[int, int], ...]
    neighborhoods: tuple[ProposalNeighborhood, ...]
    left_degrees: tuple[int, ...]
    right_degrees: tuple[int, ...]


@dataclass(frozen=True)
class SpectralProposalResult:
    proposal: BipartiteProposal
    cluster_count: int
    eigenvalues: tuple[float, ...]
    correlation_threshold: float
    mixed_cluster_count: int


def decoder_cosine_affinity(d_left: np.ndarray, d_right: np.ndarray) -> np.ndarray:
    left = np.asarray(d_left, dtype=np.float64)
    right = np.asarray(d_right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("decoders must be rank-2 and share hook dimension")
    norms = np.linalg.norm(left, axis=0)[:, None] * np.linalg.norm(right, axis=0)[None, :]
    scores = np.full((left.shape[1], right.shape[1]), -np.inf)
    active = norms > 0.0
    scores[active] = np.abs((left.T @ right)[active] / norms[active])
    return scores


def absolute_code_correlation_affinity(z_left: np.ndarray, z_right: np.ndarray) -> np.ndarray:
    left = np.asarray(z_left, dtype=np.float64)
    right = np.asarray(z_right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0] or left.shape[0] == 0:
        raise ValueError("codes must be nonempty rank-2 arrays with paired observations")
    left = left - np.mean(left, axis=0, keepdims=True)
    right = right - np.mean(right, axis=0, keepdims=True)
    norms = np.linalg.norm(left, axis=0)[:, None] * np.linalg.norm(right, axis=0)[None, :]
    scores = np.full((left.shape[1], right.shape[1]), -np.inf)
    active = norms > 0.0
    scores[active] = np.abs((left.T @ right)[active] / norms[active])
    return scores


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    denominator = norms[:, None] * norms[None, :]
    result = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    active = denominator > 0.0
    gram = centered.T @ centered
    result[active] = gram[active] / denominator[active]
    np.fill_diagonal(result, (norms > 0.0).astype(np.float64))
    return result


def _deterministic_kmeans(points: np.ndarray, cluster_count: int, *, seed: int, restarts: int = 10) -> np.ndarray:
    if not 1 <= cluster_count <= points.shape[0]:
        raise ValueError("cluster_count must be within the number of points")
    best_labels = None
    best_inertia = np.inf
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        centers = [int(rng.integers(points.shape[0]))]
        while len(centers) < cluster_count:
            distances = np.min(np.sum((points[:, None, :] - points[np.asarray(centers)][None, :, :]) ** 2, axis=2), axis=1)
            distances[np.asarray(centers)] = 0.0
            total = float(np.sum(distances))
            if total == 0.0:
                centers.append(next(index for index in range(points.shape[0]) if index not in centers))
            else:
                centers.append(int(rng.choice(points.shape[0], p=distances / total)))
        centroid = points[np.asarray(centers)].copy()
        labels = np.zeros(points.shape[0], dtype=int)
        for _ in range(100):
            distances = np.sum((points[:, None, :] - centroid[None, :, :]) ** 2, axis=2)
            updated = np.argmin(distances, axis=1)
            if np.array_equal(updated, labels):
                break
            labels = updated
            for cluster in range(cluster_count):
                members = points[labels == cluster]
                if len(members):
                    centroid[cluster] = np.mean(members, axis=0)
        inertia = float(np.sum((points - centroid[labels]) ** 2))
        key = tuple(int(value) for value in labels)
        if inertia < best_inertia - 1e-15 or (abs(inertia - best_inertia) <= 1e-15 and (best_labels is None or key < tuple(best_labels))):
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


def li15_spectral_proposal(
    z_left: np.ndarray,
    z_right: np.ndarray,
    *,
    correlation_threshold: float,
    max_clusters: int,
    kmeans_seed: int,
    max_neighborhood_atoms: int,
) -> SpectralProposalResult:
    """Independent implementation of Li et al.'s joint-correlation spectral baseline."""
    left = np.asarray(z_left, dtype=np.float64)
    right = np.asarray(z_right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0] or left.shape[0] == 0:
        raise ValueError("codes must be nonempty rank-2 arrays with paired observations")
    if not 0.0 <= correlation_threshold < 1.0:
        raise ValueError("correlation_threshold must be in [0, 1)")
    combined = np.concatenate([left, right], axis=1)
    correlation = np.abs(_correlation_matrix(combined))
    adjacency = np.where(correlation > correlation_threshold, correlation, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    vertex_count = combined.shape[1]
    upper = min(max_clusters, vertex_count - 1)
    if upper < 2:
        cluster_count = 1
    else:
        candidate_counts = np.arange(2, upper + 1)
        gaps = np.asarray([eigenvalues[count] - eigenvalues[count - 1] for count in candidate_counts])
        cluster_count = int(candidate_counts[int(np.argmax(gaps))])
    labels = _deterministic_kmeans(eigenvectors[:, :cluster_count], cluster_count, seed=kmeans_seed)
    left_count = left.shape[1]
    edges: set[tuple[int, int]] = set()
    mixed_clusters = 0
    for cluster in range(cluster_count):
        members = np.flatnonzero(labels == cluster)
        left_ids = [int(index) for index in members if index < left_count]
        right_ids = [int(index - left_count) for index in members if index >= left_count]
        if left_ids and right_ids:
            mixed_clusters += 1
            edges.update((left_id, right_id) for left_id in left_ids for right_id in right_ids)
    proposal = _proposal_from_edges(
        tuple(sorted(edges)),
        left_count=left.shape[1],
        right_count=right.shape[1],
        top_k=0,
        score_source="LI15-SPECTRAL",
        max_neighborhood_atoms=max_neighborhood_atoms,
    )
    return SpectralProposalResult(
        proposal=proposal,
        cluster_count=cluster_count,
        eigenvalues=tuple(float(value) for value in eigenvalues),
        correlation_threshold=correlation_threshold,
        mixed_cluster_count=mixed_clusters,
    )


def validate_independent_split_seeds(seed_provenance: dict[str, int]) -> None:
    """Fail closed unless structure, mean, discovery and eval streams are distinct."""
    missing = [name for name in REQUIRED_SPLIT_SEEDS if name not in seed_provenance]
    if missing:
        raise ValueError(f"missing required split seeds: {missing}")
    values = [int(seed_provenance[name]) for name in REQUIRED_SPLIT_SEEDS]
    if len(set(values)) != len(values):
        raise ValueError("structural, mean, discovery and eval seeds must be pairwise distinct")


def singleton_contribution_affinity(
    k_ll: np.ndarray,
    k_lr: np.ndarray,
    k_rr: np.ndarray,
    *,
    inactive_atol: float = 1e-15,
) -> np.ndarray:
    """Return negative singleton normalized residual using discovery-only kernels."""
    if k_ll.ndim != 2 or k_lr.ndim != 2 or k_rr.ndim != 2:
        raise ValueError("kernels must be rank-2")
    left_count, right_count = k_lr.shape
    if k_ll.shape != (left_count, left_count) or k_rr.shape != (right_count, right_count):
        raise ValueError("kernel shapes are inconsistent")
    energy = np.diag(k_ll)[:, None] + np.diag(k_rr)[None, :]
    residual = np.full_like(k_lr, np.inf, dtype=np.float64)
    active = energy > inactive_atol
    residual[active] = (energy[active] - 2.0 * k_lr[active]) / energy[active]
    return -residual


def symmetric_topk_proposal(
    score_matrix: np.ndarray,
    *,
    top_k: int,
    score_source: str,
    max_neighborhood_atoms: int,
) -> BipartiteProposal:
    """Build deterministic overlapping anchor neighborhoods from a full score matrix."""
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("score_matrix must be a nonempty rank-2 matrix")
    if not np.all(np.isfinite(scores) | np.isneginf(scores)):
        raise ValueError("scores must be finite or negative infinity")
    if not score_source:
        raise ValueError("score_source must be nonempty")
    if top_k < 1 or max_neighborhood_atoms < 2:
        raise ValueError("top_k must be positive and max_neighborhood_atoms at least two")
    left_count, right_count = scores.shape
    k_left = min(top_k, right_count)
    k_right = min(top_k, left_count)
    edges: set[tuple[int, int]] = set()
    for left_id in range(left_count):
        order = sorted(range(right_count), key=lambda right_id: (-scores[left_id, right_id], right_id))
        edges.update((left_id, right_id) for right_id in order[:k_left] if np.isfinite(scores[left_id, right_id]))
    for right_id in range(right_count):
        order = sorted(range(left_count), key=lambda left_id: (-scores[left_id, right_id], left_id))
        edges.update((left_id, right_id) for left_id in order[:k_right] if np.isfinite(scores[left_id, right_id]))
    return _proposal_from_edges(
        tuple(sorted(edges)),
        left_count=left_count,
        right_count=right_count,
        top_k=top_k,
        score_source=score_source,
        max_neighborhood_atoms=max_neighborhood_atoms,
    )


def _proposal_from_edges(
    ordered_edges: tuple[tuple[int, int], ...],
    *,
    left_count: int,
    right_count: int,
    top_k: int,
    score_source: str,
    max_neighborhood_atoms: int,
) -> BipartiteProposal:
    left_neighbors = {left_id: set() for left_id in range(left_count)}
    right_neighbors = {right_id: set() for right_id in range(right_count)}
    for left_id, right_id in ordered_edges:
        left_neighbors[left_id].add(right_id)
        right_neighbors[right_id].add(left_id)
    neighborhoods: list[ProposalNeighborhood] = []
    for anchor_left, anchor_right in ordered_edges:
        left_ids = tuple(sorted(right_neighbors[anchor_right]))
        right_ids = tuple(sorted(left_neighbors[anchor_left]))
        over_budget = len(left_ids) + len(right_ids) > max_neighborhood_atoms
        neighborhoods.append(ProposalNeighborhood(
            anchor_left=anchor_left,
            anchor_right=anchor_right,
            left_ids=left_ids,
            right_ids=right_ids,
            status="BUDGET_REFUSAL" if over_budget else "OK",
            refusal_reason="NEIGHBORHOOD_ATOM_CAP_EXCEEDED" if over_budget else None,
        ))
    return BipartiteProposal(
        score_source=score_source,
        top_k=top_k,
        edges=ordered_edges,
        neighborhoods=tuple(neighborhoods),
        left_degrees=tuple(len(left_neighbors[left_id]) for left_id in range(left_count)),
        right_degrees=tuple(len(right_neighbors[right_id]) for right_id in range(right_count)),
    )


def degree_matched_random_proposal(
    reference: BipartiteProposal,
    *,
    seed: int,
    max_neighborhood_atoms: int,
    swap_attempt_multiplier: int = 20,
) -> BipartiteProposal:
    """Randomize a proposal by simple bipartite edge swaps that preserve exact degrees."""
    if swap_attempt_multiplier < 1:
        raise ValueError("swap_attempt_multiplier must be positive")
    left_count = len(reference.left_degrees)
    right_count = len(reference.right_degrees)
    rng = np.random.default_rng(seed)
    edges = set(reference.edges)
    edge_list = sorted(edges)
    accepted = 0
    for _ in range(max(1, swap_attempt_multiplier * len(edge_list))):
        if len(edge_list) < 2:
            break
        first, second = rng.choice(len(edge_list), size=2, replace=False)
        (left_a, right_a), (left_b, right_b) = edge_list[first], edge_list[second]
        if left_a == left_b or right_a == right_b:
            continue
        swapped_a = (left_a, right_b)
        swapped_b = (left_b, right_a)
        if swapped_a in edges or swapped_b in edges:
            continue
        edges.remove((left_a, right_a))
        edges.remove((left_b, right_b))
        edges.add(swapped_a)
        edges.add(swapped_b)
        edge_list = sorted(edges)
        accepted += 1
    proposal = _proposal_from_edges(
        tuple(edge_list),
        left_count=left_count,
        right_count=right_count,
        top_k=reference.top_k,
        score_source="RANDOM-MATCHED",
        max_neighborhood_atoms=max_neighborhood_atoms,
    )
    if proposal.left_degrees != reference.left_degrees or proposal.right_degrees != reference.right_degrees:
        raise RuntimeError("degree-matched randomization changed the degree sequence")
    return proposal


def proposal_candidate_family(
    proposal: BipartiteProposal,
    *,
    max_group_size: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return the deduplicated finite subset-pair family from non-refused neighborhoods."""
    if max_group_size < 1:
        raise ValueError("max_group_size must be positive")
    family: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for neighborhood in proposal.neighborhoods:
        if neighborhood.status != "OK":
            continue
        left_subsets = [
            subset
            for size in range(1, min(len(neighborhood.left_ids), max_group_size) + 1)
            for subset in combinations(neighborhood.left_ids, size)
        ]
        right_subsets = [
            subset
            for size in range(1, min(len(neighborhood.right_ids), max_group_size) + 1)
            for subset in combinations(neighborhood.right_ids, size)
        ]
        family.update((left, right) for left in left_subsets for right in right_subsets)
    return tuple(sorted(family, key=lambda item: (len(item[0]) + len(item[1]), item[0], item[1])))
