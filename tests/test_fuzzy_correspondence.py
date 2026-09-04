from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from ccad.fuzzy_correspondence import (
    evaluate_fixed_correspondence,
    fit_fuzzy_correspondence,
    fit_fuzzy_correspondence_from_kernels,
    fit_probe_metric,
    metric_factor,
    sparse_contribution_kernels,
    soft_membership_overlap,
)


class FuzzyCorrespondenceTests(unittest.TestCase):
    def test_probe_metric_recovers_output_sensitive_subspace(self) -> None:
        rng = np.random.default_rng(3)
        directions = rng.normal(size=(2000, 4))
        output_map = np.asarray([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        effects = directions @ output_map
        fitted = fit_probe_metric(directions, effects, ridge_fraction=1e-8)
        self.assertEqual(fitted.rank, 2)
        basis, _ = metric_factor(fitted.matrix)
        projector = np.linalg.qr(basis)[0] @ np.linalg.qr(basis)[0].T
        np.testing.assert_allclose(projector, np.diag([1.0, 1.0, 0.0, 0.0]), atol=1e-6)

    def test_split_merge_is_a_native_many_to_many_correspondence(self) -> None:
        rng = np.random.default_rng(4)
        latent = rng.normal(size=(1200, 2))
        source = np.zeros((len(latent), 2, 2))
        target = np.zeros((len(latent), 3, 2))
        source[:, 0, 0] = latent[:, 0]
        source[:, 1, 1] = latent[:, 1]
        target[:, 0, 0] = 0.35 * latent[:, 0]
        target[:, 1, 0] = 0.65 * latent[:, 0]
        target[:, 2, 1] = latent[:, 1]
        weights = np.full(len(latent), 1.0 / len(latent))
        result = fit_fuzzy_correspondence(source, target, weights, rank=2)
        self.assertGreater(result.canonical_values[-1], 0.999)
        self.assertGreater(result.source_effective_support, 1.7)
        self.assertGreater(result.target_effective_support, 2.4)
        self.assertGreater(result.target_membership[0], 0.05)
        self.assertGreater(result.target_membership[1], 0.05)
        self.assertGreater(result.target_membership[2], 0.05)

    def test_causal_seminorm_ignores_downstream_null_difference(self) -> None:
        rng = np.random.default_rng(5)
        signal = rng.normal(size=1000)
        nuisance = rng.normal(size=1000)
        source = np.zeros((1000, 1, 2))
        target = np.zeros((1000, 2, 2))
        source[:, 0, 0] = signal
        source[:, 0, 1] = nuisance
        target[:, 0, 0] = signal
        target[:, 1, 1] = -nuisance
        weights = np.full(1000, 0.001)
        euclidean = fit_fuzzy_correspondence(source, target, weights, rank=1)
        causal = fit_fuzzy_correspondence(
            source, target, weights, metric=np.diag([1.0, 0.0]), rank=1,
        )
        self.assertGreater(causal.target_membership[0], 0.999)
        self.assertGreater(causal.canonical_values[0], euclidean.canonical_values[0])

    def test_hard_negative_contrast_suppresses_shared_nuisance(self) -> None:
        rng = np.random.default_rng(6)
        rows = 2000
        concept = rng.normal(size=rows)
        nuisance = rng.normal(size=rows)
        source = np.zeros((rows, 2, 2))
        target = np.zeros((rows, 2, 2))
        source[:, 0, 0] = concept
        target[:, 0, 0] = concept
        source[:, 1, 1] = nuisance
        target[:, 1, 1] = nuisance
        positive = np.r_[np.full(rows // 2, 2.0 / rows), np.zeros(rows // 2)]
        negative = np.r_[np.zeros(rows // 2), np.full(rows // 2, 2.0 / rows)]
        source[rows // 2 :, 0, :] = 0.0
        target[rows // 2 :, 0, :] = 0.0
        plain = fit_fuzzy_correspondence(source, target, positive, rank=1)
        contrasted = fit_fuzzy_correspondence(
            source, target, positive, negative_weights=negative,
            rank=1, contrast_strength=1.0,
        )
        self.assertGreater(contrasted.source_membership[0], plain.source_membership[0])
        self.assertGreater(contrasted.target_membership[0], plain.target_membership[0])

    def test_soft_overlap_detects_cross_query_collision(self) -> None:
        self.assertAlmostEqual(soft_membership_overlap([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(soft_membership_overlap([1, 1], [2, 2]), 1.0)

    def test_fixed_relation_transfers_to_held_out_samples(self) -> None:
        rng = np.random.default_rng(12)
        train = rng.normal(size=1000)
        heldout = rng.normal(size=600)
        source_train = np.zeros((1000, 1, 2))
        target_train = np.zeros_like(source_train)
        source_train[:, 0, 0] = train
        target_train[:, 0, 0] = train
        fitted = fit_fuzzy_correspondence(
            source_train, target_train, np.full(1000, 0.001), rank=1,
        )
        source_test = np.zeros((600, 1, 2))
        target_test = np.zeros_like(source_test)
        source_test[:, 0, 0] = heldout
        target_test[:, 0, 0] = heldout
        metrics = evaluate_fixed_correspondence(
            source_test, target_test, np.full(600, 1.0 / 600),
            fitted.source_loadings, fitted.target_loadings,
        )
        self.assertAlmostEqual(metrics.bcc, 1.0, places=10)
        self.assertAlmostEqual(metrics.normalized_residual, 0.0, places=10)

    def test_rotation_preserves_the_paired_aggregate_relation(self) -> None:
        rng = np.random.default_rng(8)
        latent = rng.normal(size=(1500, 2))
        angle = 0.63
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        source = np.zeros((len(latent), 2, 2))
        target = np.zeros_like(source)
        source[:, 0, 0] = latent[:, 0]
        source[:, 1, 1] = latent[:, 1]
        rotated = latent @ rotation
        target[:, 0, :] = rotated[:, 0, None] * rotation[:, 0]
        target[:, 1, :] = rotated[:, 1, None] * rotation[:, 1]
        result = fit_fuzzy_correspondence(
            source, target, np.full(len(latent), 1.0 / len(latent)), rank=1,
        )
        self.assertGreater(result.canonical_values[0], 0.999)
        self.assertTrue(np.all(result.source_membership > 0.1))
        self.assertTrue(np.all(result.target_membership > 0.1))

    def test_equal_competing_rank_one_relations_have_no_boundary_gap(self) -> None:
        rng = np.random.default_rng(9)
        latent = rng.normal(size=(4000, 2))
        source = np.zeros((len(latent), 2, 2))
        target = np.zeros_like(source)
        source[:, 0, 0] = latent[:, 0]
        source[:, 1, 1] = latent[:, 1]
        target[:] = source
        result = fit_fuzzy_correspondence(
            source, target, np.full(len(latent), 1.0 / len(latent)), rank=1,
        )
        self.assertIsNotNone(result.rank_boundary_relative_gap)
        self.assertLess(result.rank_boundary_relative_gap, 0.03)

    def test_sparse_contribution_kernel_matches_dense_centered_bank(self) -> None:
        rng = np.random.default_rng(13)
        rows, source_features, target_features, hook = 300, 4, 5, 3
        source_codes = rng.binomial(1, 0.2, size=(rows, source_features)) * rng.random((rows, source_features))
        target_codes = rng.binomial(1, 0.25, size=(rows, target_features)) * rng.random((rows, target_features))
        source_decoders = rng.normal(size=(source_features, hook))
        target_decoders = rng.normal(size=(target_features, hook))
        source_mean = rng.random(source_features) * 0.1
        target_mean = rng.random(target_features) * 0.1
        weights = rng.random(rows)
        weights /= weights.sum()
        dense_source = (source_codes - source_mean[None, :])[:, :, None] * source_decoders[None, :, :]
        dense_target = (target_codes - target_mean[None, :])[:, :, None] * target_decoders[None, :, :]
        dense = fit_fuzzy_correspondence(dense_source, dense_target, weights, rank=3)
        kernels = sparse_contribution_kernels(
            csr_matrix(source_codes), csr_matrix(target_codes), source_decoders, target_decoders,
            weights, source_mean_codes=source_mean, target_mean_codes=target_mean,
        )
        sparse = fit_fuzzy_correspondence_from_kernels(kernels, rank=3)
        np.testing.assert_allclose(sparse.canonical_values, dense.canonical_values, atol=1e-10)
        np.testing.assert_allclose(sparse.coupling, dense.coupling, atol=1e-10)

    def test_sparse_contribution_kernel_matches_dense_contrastive_metric_fit(self) -> None:
        rng = np.random.default_rng(15)
        rows = 400
        source_codes = rng.normal(size=(rows, 3)) * rng.binomial(1, 0.3, size=(rows, 3))
        target_codes = rng.normal(size=(rows, 4)) * rng.binomial(1, 0.3, size=(rows, 4))
        source_decoders = rng.normal(size=(3, 3))
        target_decoders = rng.normal(size=(4, 3))
        source_mean = rng.normal(scale=0.02, size=3)
        target_mean = rng.normal(scale=0.02, size=4)
        positive = np.r_[np.ones(rows // 2), np.zeros(rows // 2)]
        negative = np.r_[np.zeros(rows // 2), np.ones(rows // 2)]
        metric = np.diag([2.0, 0.5, 0.0])
        factor, _ = metric_factor(metric)
        dense_source = np.einsum(
            "npd,dk->npk",
            (source_codes - source_mean[None, :])[:, :, None] * source_decoders[None, :, :],
            factor,
        )
        dense_target = np.einsum(
            "nqd,dk->nqk",
            (target_codes - target_mean[None, :])[:, :, None] * target_decoders[None, :, :],
            factor,
        )
        dense = fit_fuzzy_correspondence(
            dense_source, dense_target, positive, negative_weights=negative,
            rank=2, contrast_strength=0.4,
        )
        kernels = sparse_contribution_kernels(
            csr_matrix(source_codes), csr_matrix(target_codes), source_decoders, target_decoders,
            positive, source_mean_codes=source_mean, target_mean_codes=target_mean,
            negative_weights=negative, metric=metric,
        )
        sparse = fit_fuzzy_correspondence_from_kernels(kernels, rank=2, contrast_strength=0.4)
        np.testing.assert_allclose(sparse.canonical_values, dense.canonical_values, atol=1e-10)
        np.testing.assert_allclose(sparse.coupling, dense.coupling, atol=1e-10)

    def test_sparse_kernel_preserves_overlapping_query_relations(self) -> None:
        rng = np.random.default_rng(14)
        rows = 2000
        first, second = rng.normal(size=(2, rows))
        first[rows // 2 :] = 0.0
        second[: rows // 2] = 0.0
        source_codes = np.column_stack([first, second])
        target_codes = np.column_stack([0.5 * first, 0.5 * (first + second), 0.5 * second])
        source_decoders = np.ones((2, 1))
        target_decoders = np.ones((3, 1))
        zero_source = np.zeros(2)
        zero_target = np.zeros(3)
        first_weights = np.r_[np.ones(rows // 2), np.zeros(rows // 2)]
        second_weights = np.r_[np.zeros(rows // 2), np.ones(rows // 2)]
        first_relation = fit_fuzzy_correspondence_from_kernels(
            sparse_contribution_kernels(
                csr_matrix(source_codes), csr_matrix(target_codes), source_decoders, target_decoders,
                first_weights, source_mean_codes=zero_source, target_mean_codes=zero_target,
            ), rank=1,
        )
        second_relation = fit_fuzzy_correspondence_from_kernels(
            sparse_contribution_kernels(
                csr_matrix(source_codes), csr_matrix(target_codes), source_decoders, target_decoders,
                second_weights, source_mean_codes=zero_source, target_mean_codes=zero_target,
            ), rank=1,
        )
        self.assertGreater(first_relation.target_membership[1], 0.05)
        self.assertGreater(second_relation.target_membership[1], 0.05)
        self.assertGreater(soft_membership_overlap(first_relation.target_membership, second_relation.target_membership), 0.05)


if __name__ == "__main__":
    unittest.main()
