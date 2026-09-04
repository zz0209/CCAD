from __future__ import annotations

import unittest

import numpy as np

from ccad.fuzzy_correspondence import (
    evaluate_fixed_correspondence,
    evaluate_fixed_correspondence_from_kernels,
    fit_crossed_probe_metric,
)


class CrossedProbeMetricTests(unittest.TestCase):
    def test_recovers_average_state_jacobian_gram(self) -> None:
        rng = np.random.default_rng(71)
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        directions = q.T
        jacobians = rng.normal(size=(3, 5, 8))
        effects = np.stack([directions @ jacobian.T for jacobian in jacobians])
        fitted = fit_crossed_probe_metric(
            directions, effects, ridge_fraction=0.0, relative_tolerance=1e-10,
        )
        expected = np.mean([jacobian.T @ jacobian for jacobian in jacobians], axis=0)
        expected *= expected.shape[0] / np.trace(expected)
        np.testing.assert_allclose(fitted.matrix, expected, atol=1e-10)

    def test_invariant_to_balanced_state_duplication(self) -> None:
        rng = np.random.default_rng(72)
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        directions = q.T
        effects = rng.normal(size=(4, 6, 3))
        original = fit_crossed_probe_metric(directions, effects, ridge_fraction=0.0)
        duplicated = fit_crossed_probe_metric(
            directions, np.repeat(effects, 2, axis=0), ridge_fraction=0.0,
        )
        np.testing.assert_allclose(original.matrix, duplicated.matrix, atol=1e-10)

    def test_kernel_fixed_evaluation_matches_dense_bank(self) -> None:
        rng = np.random.default_rng(73)
        source = rng.normal(size=(40, 3, 5))
        target = rng.normal(size=(40, 4, 5))
        weights = rng.random(40)
        weights /= weights.sum()
        left = rng.normal(size=(3, 2))
        right = rng.normal(size=(4, 2))
        dense = evaluate_fixed_correspondence(source, target, weights, left, right)
        source_matrix = (source * np.sqrt(weights)[:, None, None]).transpose(0, 2, 1).reshape(-1, 3)
        target_matrix = (target * np.sqrt(weights)[:, None, None]).transpose(0, 2, 1).reshape(-1, 4)
        kernel = evaluate_fixed_correspondence_from_kernels(
            source_matrix.T @ source_matrix,
            target_matrix.T @ target_matrix,
            source_matrix.T @ target_matrix,
            left,
            right,
        )
        self.assertAlmostEqual(dense.source_energy, kernel.source_energy)
        self.assertAlmostEqual(dense.target_energy, kernel.target_energy)
        self.assertAlmostEqual(dense.cross_energy, kernel.cross_energy)
        self.assertAlmostEqual(dense.normalized_residual, kernel.normalized_residual)
        self.assertAlmostEqual(dense.bcc, kernel.bcc)


if __name__ == "__main__":
    unittest.main()
