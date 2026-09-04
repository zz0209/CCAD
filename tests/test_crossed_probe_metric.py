from __future__ import annotations

import unittest

import numpy as np

from ccad.fuzzy_correspondence import fit_crossed_probe_metric


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


if __name__ == "__main__":
    unittest.main()
