"""Truth-known synthetic constructions tied to the CBSM proof boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticPair:
    family_id: str
    d_left: np.ndarray
    d_right: np.ndarray
    z_left_mean: np.ndarray
    z_right_mean: np.ndarray
    z_left_eval: np.ndarray
    z_right_eval: np.ndarray
    planted_hyperedges: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    split_weights: tuple[float, ...] = ()
    seed_provenance: tuple[tuple[str, int], ...] = ()
    expected_decision: str = "UNSPECIFIED"
    expected_covers: tuple[
        tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], ...
    ] = ()
    diagnostic_values: tuple[tuple[str, float], ...] = ()
    mean_document_ids: tuple[int, ...] = ()
    eval_document_ids: tuple[int, ...] = ()
    hook_mean: np.ndarray | None = None
    hook_eval: np.ndarray | None = None


def sylvester_hadamard(q: int) -> np.ndarray:
    if q < 1 or q & (q - 1):
        raise ValueError("q must be a positive power of two")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < q:
        h = np.block([[h, h], [h, -h]])
    return h


def _signed_relu_codes(coordinates: np.ndarray) -> np.ndarray:
    return np.concatenate([np.maximum(coordinates, 0.0), np.maximum(-coordinates, 0.0)], axis=1)


def hadamard_gauge_instance(q: int, n_mean: int, n_eval: int, seed: int) -> SyntheticPair:
    rng = np.random.default_rng(seed)
    x_mean = rng.standard_normal((n_mean, q))
    x_eval = rng.standard_normal((n_eval, q))
    identity = np.eye(q)
    rotation = sylvester_hadamard(q) / np.sqrt(q)
    d_left = np.concatenate([identity, -identity], axis=1)
    d_right = np.concatenate([rotation, -rotation], axis=1)
    return SyntheticPair(
        family_id="F01_hadamard_gauge",
        d_left=d_left,
        d_right=d_right,
        z_left_mean=_signed_relu_codes(x_mean),
        z_right_mean=_signed_relu_codes(x_mean @ rotation),
        z_left_eval=_signed_relu_codes(x_eval),
        z_right_eval=_signed_relu_codes(x_eval @ rotation),
    )


def _signed_permutation(q: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = np.eye(q, dtype=np.float64)[:, rng.permutation(q)]
    return matrix * rng.choice(np.array([-1.0, 1.0]), size=q)


def hadamard_gauge_seeded(
    q: int,
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """F01 with independent structural and paired-sample RNG streams."""
    left_basis = _signed_permutation(q, structural_seed_a)
    right_gauge = _signed_permutation(q, structural_seed_b)
    right_basis = left_basis @ (sylvester_hadamard(q) / np.sqrt(q)) @ right_gauge
    x_mean = np.random.default_rng(mean_sample_seed).standard_normal((n_mean, q))
    x_eval = np.random.default_rng(eval_sample_seed).standard_normal((n_eval, q))
    return SyntheticPair(
        family_id="F01_hadamard_gauge",
        d_left=np.concatenate([left_basis, -left_basis], axis=1),
        d_right=np.concatenate([right_basis, -right_basis], axis=1),
        z_left_mean=_signed_relu_codes(x_mean @ left_basis),
        z_right_mean=_signed_relu_codes(x_mean @ right_basis),
        z_left_eval=_signed_relu_codes(x_eval @ left_basis),
        z_right_eval=_signed_relu_codes(x_eval @ right_basis),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
    )


def same_span_different_computation(n_mean: int, n_eval: int, seed: int) -> SyntheticPair:
    rng = np.random.default_rng(seed)
    x_mean = rng.standard_normal((n_mean, 2))
    x_eval = rng.standard_normal((n_eval, 2))
    sign_flip = np.array([1.0, -1.0])
    return SyntheticPair(
        family_id="F10_same_span_different_computation",
        d_left=np.eye(2),
        d_right=np.eye(2),
        z_left_mean=x_mean,
        z_right_mean=x_mean * sign_flip,
        z_left_eval=x_eval,
        z_right_eval=x_eval * sign_flip,
    )


def same_span_different_computation_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """F10 risk group shares its span but applies an orthogonal function change."""
    qa, _ = np.linalg.qr(np.random.default_rng(structural_seed_a).standard_normal((2, 2)))
    qb, _ = np.linalg.qr(np.random.default_rng(structural_seed_b).standard_normal((2, 2)))
    orientation = 1.0 if structural_seed_b % 2 else -1.0
    rotation = orientation * np.array([[0.0, -1.0], [1.0, 0.0]])
    scales = 0.75 + np.random.default_rng(structural_seed_a + structural_seed_b).random(2)
    embed = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def samples(seed: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((count, 2))
        clean = rng.standard_normal((count, 1))
        return x @ qa, (x @ rotation) @ qb, clean / scales[0], clean / scales[1]

    zl_mean, zr_mean, clean_l_mean, clean_r_mean = samples(mean_sample_seed, n_mean)
    zl_eval, zr_eval, clean_l_eval, clean_r_eval = samples(eval_sample_seed, n_eval)
    return SyntheticPair(
        family_id="F10_same_span_different_computation",
        d_left=np.column_stack([embed @ qa, np.array([0.0, 0.0, scales[0]])]),
        d_right=np.column_stack([embed @ qb, np.array([0.0, 0.0, scales[1]])]),
        z_left_mean=np.column_stack([zl_mean, clean_l_mean]),
        z_right_mean=np.column_stack([zr_mean, clean_r_mean]),
        z_left_eval=np.column_stack([zl_eval, clean_l_eval]),
        z_right_eval=np.column_stack([zr_eval, clean_r_eval]),
        planted_hyperedges=(((0, 1), (0, 1)), ((2,), (2,))),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="REFUSE_FUNCTION_MISMATCH_WITH_CLEAN_CONTROL_PASS",
        diagnostic_values=(("rotation_orientation", orientation),),
    )


def same_sum_bloated_span(n_mean: int, n_eval: int, seed: int) -> SyntheticPair:
    rng = np.random.default_rng(seed)
    x_mean = rng.standard_normal((n_mean, 1))
    x_eval = rng.standard_normal((n_eval, 1))
    zeros_mean = np.zeros_like(x_mean)
    zeros_eval = np.zeros_like(x_eval)
    return SyntheticPair(
        family_id="F11_same_sum_bloated_span",
        d_left=np.array([[1.0], [0.0]]),
        d_right=np.eye(2),
        z_left_mean=x_mean,
        z_right_mean=np.concatenate([x_mean, zeros_mean], axis=1),
        z_left_eval=x_eval,
        z_right_eval=np.concatenate([x_eval, zeros_eval], axis=1),
    )


def same_sum_bloated_span_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """F11 matches summed contributions while the right decoder span is bloated."""
    rng_a = np.random.default_rng(structural_seed_a)
    angle = rng_a.uniform(-np.pi, np.pi)
    u = np.array([np.cos(angle), np.sin(angle), 0.0])
    orientation = 1.0 if structural_seed_b % 2 else -1.0
    v = orientation * np.array([-np.sin(angle), np.cos(angle), 0.0])
    rng_b = np.random.default_rng(structural_seed_b)
    alpha = rng_b.uniform(0.5, 1.5)
    scales = rng_b.uniform(0.75, 1.25, size=4)

    d_left = np.column_stack([scales[0] * u, scales[3] * np.array([0.0, 0.0, 1.0])])
    d_right = np.column_stack([
        scales[1] * (u + alpha * v),
        scales[2] * (u - alpha * v),
        np.array([0.0, 0.0, 1.0]),
    ])

    def samples(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((count, 1))
        clean = rng.standard_normal((count, 1))
        left = np.column_stack([x[:, 0] / scales[0], clean[:, 0] / scales[3]])
        right = np.column_stack([
            x[:, 0] / (2.0 * scales[1]),
            x[:, 0] / (2.0 * scales[2]),
            clean[:, 0],
        ])
        return left, right

    zl_mean, zr_mean = samples(mean_sample_seed, n_mean)
    zl_eval, zr_eval = samples(eval_sample_seed, n_eval)
    return SyntheticPair(
        family_id="F11_same_sum_bloated_span",
        d_left=d_left,
        d_right=d_right,
        z_left_mean=zl_mean,
        z_right_mean=zr_mean,
        z_left_eval=zl_eval,
        z_right_eval=zr_eval,
        planted_hyperedges=(((0,), (0, 1)), ((1,), (2,))),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="REFUSE_SPAN_BLOAT_WITH_CLEAN_CONTROL_PASS",
        diagnostic_values=(("span_bloat_alpha", float(alpha)),),
    )


def non_lipschitz_downstream_cliff_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """F12: arbitrarily close contributions straddle a discontinuous readout."""
    if n_mean < 4 or n_eval < 4 or n_mean % 2 or n_eval % 2:
        raise ValueError("F12 requires even mean/eval sizes of at least four")
    angle = np.random.default_rng(structural_seed_a).uniform(-np.pi, np.pi)
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-np.sin(angle), np.cos(angle)])
    rng_b = np.random.default_rng(structural_seed_b)
    delta = float(10.0 ** rng_b.uniform(-4.0, -2.0))
    scales = rng_b.uniform(0.75, 1.25, size=2)

    def samples(seed: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        signs = np.concatenate([np.ones(count // 2), -np.ones(count // 2)])
        rng.shuffle(signs)
        common = rng.standard_normal(count)
        common -= np.mean(common)
        common -= signs * (np.dot(common, signs) / np.dot(signs, signs))
        common /= np.sqrt(np.mean(common ** 2))
        left = ((common - delta * signs) / scales[0])[:, None]
        right = ((common + delta * signs) / scales[1])[:, None]
        hook = common[:, None] * u[None, :] + signs[:, None] * v[None, :]
        return left, right, hook

    zl_mean, zr_mean, hook_mean = samples(mean_sample_seed, n_mean)
    zl_eval, zr_eval, hook_eval = samples(eval_sample_seed, n_eval)
    return SyntheticPair(
        family_id="F12_non_lipschitz_downstream_cliff",
        d_left=(scales[0] * u)[:, None],
        d_right=(scales[1] * u)[:, None],
        z_left_mean=zl_mean,
        z_right_mean=zr_mean,
        z_left_eval=zl_eval,
        z_right_eval=zr_eval,
        planted_hyperedges=(((0,), (0,)),),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="NONCAUSAL_UNDER_UNCERTIFIED_READOUT_WITH_SMOOTH_CONTROL_PASS",
        diagnostic_values=(("delta", delta), ("direction_angle", float(angle))),
        hook_mean=hook_mean,
        hook_eval=hook_eval,
    )


def local_block_rotations(
    block_ranks: tuple[int, ...],
    n_mean: int,
    n_eval: int,
    seed: int,
) -> SyntheticPair:
    """Orthogonal hook subspaces with independent within-block basis rotations."""
    return local_block_rotations_seeded(
        block_ranks,
        n_mean,
        n_eval,
        structural_seed_a=seed,
        structural_seed_b=seed + 1,
        mean_sample_seed=seed + 2,
        eval_sample_seed=seed + 3,
    )


def local_block_rotations_seeded(
    block_ranks: tuple[int, ...],
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """F02 blocks with independent left/right bases and paired sample seeds."""
    if not block_ranks or any(rank < 2 for rank in block_ranks):
        raise ValueError("local rotation blocks must all have rank at least two")
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    hook_dim = sum(block_ranks)
    x_mean = np.random.default_rng(mean_sample_seed).standard_normal((n_mean, hook_dim))
    x_eval = np.random.default_rng(eval_sample_seed).standard_normal((n_eval, hook_dim))
    d_left_parts: list[np.ndarray] = []
    d_right_parts: list[np.ndarray] = []
    z_left_mean_parts: list[np.ndarray] = []
    z_right_mean_parts: list[np.ndarray] = []
    z_left_eval_parts: list[np.ndarray] = []
    z_right_eval_parts: list[np.ndarray] = []
    planted: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    hook_offset = 0
    feature_offset = 0
    for rank in block_ranks:
        left_q, _ = np.linalg.qr(rng_a.standard_normal((rank, rank)))
        right_q, _ = np.linalg.qr(rng_b.standard_normal((rank, rank)))
        embed = np.zeros((hook_dim, rank))
        embed[hook_offset : hook_offset + rank, :] = np.eye(rank)
        left_rotated = embed @ left_q
        right_rotated = embed @ right_q
        d_left_local = np.concatenate([left_rotated, -left_rotated], axis=1)
        d_right_local = np.concatenate([right_rotated, -right_rotated], axis=1)
        xm = x_mean[:, hook_offset : hook_offset + rank]
        xe = x_eval[:, hook_offset : hook_offset + rank]
        d_left_parts.append(d_left_local)
        d_right_parts.append(d_right_local)
        z_left_mean_parts.append(_signed_relu_codes(xm @ left_q))
        z_right_mean_parts.append(_signed_relu_codes(xm @ right_q))
        z_left_eval_parts.append(_signed_relu_codes(xe @ left_q))
        z_right_eval_parts.append(_signed_relu_codes(xe @ right_q))
        ids = tuple(range(feature_offset, feature_offset + 2 * rank))
        planted.append((ids, ids))
        hook_offset += rank
        feature_offset += 2 * rank
    return SyntheticPair(
        family_id="F02_local_block_rotations",
        d_left=np.concatenate(d_left_parts, axis=1),
        d_right=np.concatenate(d_right_parts, axis=1),
        z_left_mean=np.concatenate(z_left_mean_parts, axis=1),
        z_right_mean=np.concatenate(z_right_mean_parts, axis=1),
        z_left_eval=np.concatenate(z_left_eval_parts, axis=1),
        z_right_eval=np.concatenate(z_right_eval_parts, axis=1),
        planted_hyperedges=tuple(planted),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="UNIQUE_PARTITION",
    )


def unequal_split_merge(block_count: int, n_mean: int, n_eval: int, seed: int) -> SyntheticPair:
    """Alternating 1-to-2 and 2-to-1 exact contribution-preserving blocks."""
    return unequal_split_merge_seeded(
        block_count,
        n_mean,
        n_eval,
        structural_seed_a=seed,
        structural_seed_b=seed + 1,
        mean_sample_seed=seed + 2,
        eval_sample_seed=seed + 3,
    )


def unequal_split_merge_seeded(
    block_count: int,
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """F03 instance with independently recorded structural and paired-sample seeds."""
    if block_count < 2:
        raise ValueError("unequal split/merge requires at least two blocks")
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    x_mean = np.random.default_rng(mean_sample_seed).standard_normal((n_mean, block_count))
    x_eval = np.random.default_rng(eval_sample_seed).standard_normal((n_eval, block_count))
    d_left_parts: list[np.ndarray] = []
    d_right_parts: list[np.ndarray] = []
    z_left_mean_parts: list[np.ndarray] = []
    z_right_mean_parts: list[np.ndarray] = []
    z_left_eval_parts: list[np.ndarray] = []
    z_right_eval_parts: list[np.ndarray] = []
    planted: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    weights: list[float] = []
    left_offset = 0
    right_offset = 0
    for block in range(block_count):
        alpha = float((rng_b if block % 2 == 0 else rng_a).uniform(0.2, 0.8))
        weights.append(alpha)
        basis = np.zeros((block_count, 1))
        basis[block, 0] = 1.0
        xm = x_mean[:, block : block + 1]
        xe = x_eval[:, block : block + 1]
        split_mean = np.concatenate([alpha * xm, (1.0 - alpha) * xm], axis=1)
        split_eval = np.concatenate([alpha * xe, (1.0 - alpha) * xe], axis=1)
        split_decoder = np.concatenate([basis, basis], axis=1)
        if block % 2 == 0:
            d_left_parts.append(basis)
            d_right_parts.append(split_decoder)
            z_left_mean_parts.append(xm)
            z_right_mean_parts.append(split_mean)
            z_left_eval_parts.append(xe)
            z_right_eval_parts.append(split_eval)
            planted.append(((left_offset,), (right_offset, right_offset + 1)))
            left_offset += 1
            right_offset += 2
        else:
            d_left_parts.append(split_decoder)
            d_right_parts.append(basis)
            z_left_mean_parts.append(split_mean)
            z_right_mean_parts.append(xm)
            z_left_eval_parts.append(split_eval)
            z_right_eval_parts.append(xe)
            planted.append(((left_offset, left_offset + 1), (right_offset,)))
            left_offset += 2
            right_offset += 1
    return SyntheticPair(
        family_id="F03_unequal_split_merge",
        d_left=np.concatenate(d_left_parts, axis=1),
        d_right=np.concatenate(d_right_parts, axis=1),
        z_left_mean=np.concatenate(z_left_mean_parts, axis=1),
        z_right_mean=np.concatenate(z_right_mean_parts, axis=1),
        z_left_eval=np.concatenate(z_left_eval_parts, axis=1),
        z_right_eval=np.concatenate(z_right_eval_parts, axis=1),
        planted_hyperedges=tuple(planted),
        split_weights=tuple(weights),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="UNIQUE",
    )


def partial_overlap_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """Two exact hyperedges share atom 0; no disjoint partition can retain both."""
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    left_scales = rng_a.uniform(0.5, 1.5, size=3)
    right_scales = rng_b.uniform(0.5, 1.5, size=3)
    delta_scale = float(0.5 * (rng_a.uniform(0.5, 1.5) + rng_b.uniform(0.5, 1.5)))

    def build_targets(sample_seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        latent = np.random.default_rng(sample_seed).standard_normal((count, 4))
        left = latent[:, :3]
        delta = delta_scale * latent[:, 3:4]
        right = np.concatenate([
            left[:, 0:1] + delta,
            left[:, 1:2] - delta,
            left[:, 2:3] - delta,
        ], axis=1)
        return left, right

    left_mean_targets, right_mean_targets = build_targets(mean_sample_seed, n_mean)
    left_eval_targets, right_eval_targets = build_targets(eval_sample_seed, n_eval)
    d_left = np.zeros((2, 3))
    d_right = np.zeros((2, 3))
    d_left[0, :] = left_scales
    d_right[0, :] = right_scales
    return SyntheticPair(
        family_id="F04_partial_overlap",
        d_left=d_left,
        d_right=d_right,
        z_left_mean=left_mean_targets / left_scales,
        z_right_mean=right_mean_targets / right_scales,
        z_left_eval=left_eval_targets / left_scales,
        z_right_eval=right_eval_targets / right_scales,
        planted_hyperedges=(((0, 1), (0, 1)), ((0, 2), (0, 2))),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="OVERLAPPING_HYPERGRAPH",
    )


def cooccurrence_confounding_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """Perfectly correlated codes with deliberately misaligned decoder directions."""
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    theta = float(rng_a.uniform(-np.pi, np.pi))
    decoder_cosine = float(rng_b.uniform(0.0, 0.2))
    decoder_sign = -1.0 if rng_b.integers(0, 2) == 0 else 1.0
    left_direction = np.array([[np.cos(theta)], [np.sin(theta)]])
    perpendicular = np.array([[-np.sin(theta)], [np.cos(theta)]])
    right_direction = decoder_sign * decoder_cosine * left_direction + np.sqrt(1.0 - decoder_cosine ** 2) * perpendicular
    left_scale = float(rng_a.uniform(0.5, 1.5))
    right_scale = float(rng_b.uniform(0.5, 1.5))
    x_mean = np.random.default_rng(mean_sample_seed).standard_normal((n_mean, 1))
    x_eval = np.random.default_rng(eval_sample_seed).standard_normal((n_eval, 1))
    return SyntheticPair(
        family_id="F06_cooccurrence_confounding",
        d_left=left_scale * left_direction,
        d_right=right_scale * right_direction,
        z_left_mean=x_mean / left_scale,
        z_right_mean=x_mean / right_scale,
        z_left_eval=x_eval / left_scale,
        z_right_eval=x_eval / right_scale,
        planted_hyperedges=(),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="REFUSE_CONTRIBUTION_MISMATCH",
    )


def competing_covers_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """Two duplicate atoms per seed induce two indistinguishable maximum exact covers."""
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    left_scales = rng_a.uniform(0.5, 1.5, size=2)
    right_scales = rng_b.uniform(0.5, 1.5, size=2)
    x_mean = np.random.default_rng(mean_sample_seed).standard_normal((n_mean, 1))
    x_eval = np.random.default_rng(eval_sample_seed).standard_normal((n_eval, 1))
    d_left = np.zeros((2, 2))
    d_right = np.zeros((2, 2))
    d_left[0, :] = left_scales
    d_right[0, :] = right_scales
    return SyntheticPair(
        family_id="F08_competing_covers",
        d_left=d_left,
        d_right=d_right,
        z_left_mean=np.repeat(x_mean, 2, axis=1) / left_scales,
        z_right_mean=np.repeat(x_mean, 2, axis=1) / right_scales,
        z_left_eval=np.repeat(x_eval, 2, axis=1) / left_scales,
        z_right_eval=np.repeat(x_eval, 2, axis=1) / right_scales,
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="AMBIGUOUS",
        expected_covers=(
            (((0,), (0,)), ((1,), (1,))),
            (((0,), (1,)), ((1,), (0,))),
        ),
    )


def whole_dictionary_only_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """Only the full 3-by-3 dictionaries balance; no proper local subset pair does."""
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    left_scales = rng_a.uniform(0.5, 1.5, size=3)
    right_scales = rng_b.uniform(0.5, 1.5, size=3)
    delta_scale = float(0.5 * (rng_a.uniform(0.5, 1.5) + rng_b.uniform(0.5, 1.5)))

    def build_targets(sample_seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        latent = np.random.default_rng(sample_seed).standard_normal((count, 5))
        left = latent[:, :3]
        delta_1 = delta_scale * latent[:, 3:4]
        delta_2 = delta_scale * latent[:, 4:5]
        right = np.concatenate([
            left[:, 0:1] + delta_1,
            left[:, 1:2] + delta_2,
            left[:, 2:3] - delta_1 - delta_2,
        ], axis=1)
        return left, right

    left_mean, right_mean = build_targets(mean_sample_seed, n_mean)
    left_eval, right_eval = build_targets(eval_sample_seed, n_eval)
    d_left = np.zeros((2, 3))
    d_right = np.zeros((2, 3))
    d_left[0, :] = left_scales
    d_right[0, :] = right_scales
    full_edge = ((0, 1, 2), (0, 1, 2))
    return SyntheticPair(
        family_id="F09_whole_dictionary_only",
        d_left=d_left,
        d_right=d_right,
        z_left_mean=left_mean / left_scales,
        z_right_mean=right_mean / right_scales,
        z_left_eval=left_eval / left_scales,
        z_right_eval=right_eval / right_scales,
        planted_hyperedges=(full_edge,),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="REFUSE_GLOBAL_ONLY",
    )


def cancellation_seeded(
    n_mean: int,
    n_eval: int,
    *,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """An exact group match hides large within-seed cancellation; a clean singleton is the control."""
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    cancellation_scale_a = float(rng_a.uniform(8.0, 12.0))
    cancellation_scale_b = float(rng_b.uniform(8.0, 12.0))
    left_scales = rng_a.uniform(0.5, 1.5, size=3)
    right_scales = rng_b.uniform(0.5, 1.5, size=3)

    def build_targets(sample_seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        latent = np.random.default_rng(sample_seed).standard_normal((count, 4))
        signal = latent[:, 0:1]
        cancel_a = cancellation_scale_a * latent[:, 1:2]
        cancel_b = cancellation_scale_b * latent[:, 2:3]
        clean = latent[:, 3:4]
        left = np.concatenate([signal + cancel_a, -cancel_a, clean], axis=1)
        right = np.concatenate([signal + cancel_b, -cancel_b, clean], axis=1)
        return left, right

    left_mean, right_mean = build_targets(mean_sample_seed, n_mean)
    left_eval, right_eval = build_targets(eval_sample_seed, n_eval)
    d_left = np.zeros((2, 3))
    d_right = np.zeros((2, 3))
    d_left[0, :2] = left_scales[:2]
    d_right[0, :2] = right_scales[:2]
    d_left[1, 2] = left_scales[2]
    d_right[1, 2] = right_scales[2]
    return SyntheticPair(
        family_id="F05_cancellation",
        d_left=d_left,
        d_right=d_right,
        z_left_mean=left_mean / left_scales,
        z_right_mean=right_mean / right_scales,
        z_left_eval=left_eval / left_scales,
        z_right_eval=right_eval / right_scales,
        planted_hyperedges=(((0, 1), (0, 1)), ((2,), (2,))),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="REFUSE_CANCELLATION_RISK_WITH_CLEAN_CONTROL_PASS",
        diagnostic_values=(
            ("cancellation_scale_a", cancellation_scale_a),
            ("cancellation_scale_b", cancellation_scale_b),
        ),
    )


def rare_occupancy_seeded(
    n_mean: int,
    n_eval: int,
    *,
    tokens_per_document: int,
    active_document_count: int,
    structural_seed_a: int,
    structural_seed_b: int,
    mean_sample_seed: int,
    eval_sample_seed: int,
) -> SyntheticPair:
    """A rare, clustered approximate match and a dense exact singleton control."""
    if n_mean % tokens_per_document or n_eval % tokens_per_document:
        raise ValueError("mean/eval sample counts must be divisible by tokens_per_document")
    mean_document_count = n_mean // tokens_per_document
    eval_document_count = n_eval // tokens_per_document
    if active_document_count < 1 or active_document_count >= min(mean_document_count, eval_document_count):
        raise ValueError("active_document_count must be positive and smaller than each split's document count")
    rng_a = np.random.default_rng(structural_seed_a)
    rng_b = np.random.default_rng(structural_seed_b)
    left_scales = rng_a.uniform(0.5, 1.5, size=2)
    right_scales = rng_b.uniform(0.5, 1.5, size=2)

    def build_targets(sample_seed: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(sample_seed)
        document_count = count // tokens_per_document
        documents = np.repeat(np.arange(document_count), tokens_per_document)
        active_documents = np.sort(rng.choice(document_count, size=active_document_count, replace=False))
        risk_left = np.zeros((count, 1))
        risk_right = np.zeros((count, 1))
        for ordinal, document in enumerate(active_documents):
            positions = np.flatnonzero(documents == document)[:2]
            amplitude = float(rng.uniform(0.8, 1.2))
            risk_left[positions, 0] = (amplitude, -amplitude)
            right_multiplier = 0.5 if ordinal % 2 == 0 else 1.5
            risk_right[positions, 0] = right_multiplier * risk_left[positions, 0]
        clean = rng.standard_normal((count, 1))
        clean -= np.mean(clean, axis=0, keepdims=True)
        left = np.concatenate([risk_left, clean], axis=1)
        right = np.concatenate([risk_right, clean], axis=1)
        return left, right, documents

    left_mean, right_mean, mean_documents = build_targets(mean_sample_seed, n_mean)
    left_eval, right_eval, eval_documents = build_targets(eval_sample_seed, n_eval)
    d_left = np.zeros((2, 2))
    d_right = np.zeros((2, 2))
    d_left[0, 0] = left_scales[0]
    d_right[0, 0] = right_scales[0]
    d_left[1, 1] = left_scales[1]
    d_right[1, 1] = right_scales[1]
    return SyntheticPair(
        family_id="F07_rare_occupancy",
        d_left=d_left,
        d_right=d_right,
        z_left_mean=left_mean / left_scales,
        z_right_mean=right_mean / right_scales,
        z_left_eval=left_eval / left_scales,
        z_right_eval=right_eval / right_scales,
        planted_hyperedges=(((0,), (0,)), ((1,), (1,))),
        seed_provenance=(
            ("structural_seed_a", structural_seed_a),
            ("structural_seed_b", structural_seed_b),
            ("mean_sample_seed", mean_sample_seed),
            ("eval_sample_seed", eval_sample_seed),
        ),
        expected_decision="REFUSE_LOW_N_EFF_WITH_DENSE_CONTROL_PASS",
        diagnostic_values=(
            ("tokens_per_document", float(tokens_per_document)),
            ("active_document_count", float(active_document_count)),
        ),
        mean_document_ids=tuple(int(value) for value in mean_documents),
        eval_document_ids=tuple(int(value) for value in eval_documents),
    )
