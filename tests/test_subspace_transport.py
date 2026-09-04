from __future__ import annotations

import unittest

import numpy as np

from ccad.subspace_transport import (
    fit_weighted_pca,
    fit_weighted_stitching,
    mean_transfer_metrics,
    projector_subspace_similarity,
    random_orthonormal_basis,
    select_weighted_support,
    stable_seed,
    subspace_ablation,
    transfer_metrics,
    weighted_mean,
)


class SubspaceTransportTests(unittest.TestCase):
    def test_source_only_weight_selection_is_deterministic_and_normalized(self) -> None:
        code = np.asarray([0.0, -2.0, 1.0, 2.0, 0.5])
        support = select_weighted_support(code, max_rows=2)
        self.assertEqual(support.indices.tolist(), [1, 3])
        np.testing.assert_allclose(support.weights, [0.5, 0.5])
        self.assertEqual(support.active_count, 4)
        self.assertAlmostEqual(support.effective_sample_size, 2.0)

    def test_rotated_factor_recovers_same_projector_and_exact_transfer(self) -> None:
        rng = np.random.default_rng(7)
        latent = rng.normal(size=(256, 2))
        load = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        angle = 0.7
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        left = latent @ load.T
        right = (latent @ rotation) @ (load @ rotation).T
        weights = np.full(len(latent), 1.0 / len(latent))
        mean = np.zeros(4)
        left_basis, _ = fit_weighted_pca(left, weights, mean, 2, random_seed=1, oversample=2)
        right_basis, _ = fit_weighted_pca(right, weights, mean, 2, random_seed=2, oversample=2)
        self.assertAlmostEqual(projector_subspace_similarity(left_basis, right_basis)["psc"], 1.0, places=10)
        metrics = transfer_metrics(left, right, weights, mean, mean, left_basis, right_basis)
        self.assertAlmostEqual(metrics.normalized_residual, 0.0, places=10)
        self.assertAlmostEqual(metrics.bcc, 1.0, places=10)

    def test_stitching_recovers_paired_subspaces(self) -> None:
        rng = np.random.default_rng(11)
        latent = rng.normal(size=(300, 2))
        source_load = np.linalg.qr(rng.normal(size=(6, 2)))[0]
        target_load = np.linalg.qr(rng.normal(size=(6, 2)))[0]
        source = latent @ source_load.T
        target = latent @ target_load.T
        weights = np.full(300, 1.0 / 300)
        left, right, singular = fit_weighted_stitching(
            source, target, weights, np.zeros(6), np.zeros(6), 2,
            random_seed=stable_seed("test", "stitching"), oversample=4,
        )
        self.assertEqual(singular.size, 2)
        self.assertAlmostEqual(projector_subspace_similarity(left, source_load)["psc"], 1.0, places=10)
        self.assertAlmostEqual(projector_subspace_similarity(right, target_load)["psc"], 1.0, places=10)

    def test_means_are_not_hidden_by_dynamic_centering(self) -> None:
        result = mean_transfer_metrics(np.asarray([2.0, 0.0]), np.asarray([1.0, 0.0]))
        self.assertAlmostEqual(result["normalized_mean_residual"], 0.25)
        self.assertAlmostEqual(result["mean_bcc"], 0.8)

    def test_random_basis_is_reproducible_and_ablation_uses_shared_hook_units(self) -> None:
        basis_a = random_orthonormal_basis(5, 2, stable_seed("q", 1))
        basis_b = random_orthonormal_basis(5, 2, stable_seed("q", 1))
        np.testing.assert_array_equal(basis_a, basis_b)
        hook = np.arange(20, dtype=np.float32).reshape(4, 5)
        contribution = hook * 0.25
        mean = weighted_mean(contribution, np.ones(4))
        ablated = subspace_ablation(hook, contribution, mean, basis_a)
        expected = hook - ((contribution - mean) @ basis_a @ basis_a.T).astype(np.float32)
        np.testing.assert_allclose(ablated, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
