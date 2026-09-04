from __future__ import annotations

import unittest

import numpy as np

from ccad.fuzzy_correspondence import (
    fit_fuzzy_correspondence,
    fit_probe_metric,
    metric_factor,
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


if __name__ == "__main__":
    unittest.main()
