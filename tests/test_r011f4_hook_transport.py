from __future__ import annotations

import unittest

import numpy as np

from ccad.hook_transport import (
    decide_transport_gate,
    fit_hook_space_transport,
    transport_metrics,
)


class HookTransportTests(unittest.TestCase):
    def test_rotation_transfers_on_held_out_rows(self) -> None:
        rng = np.random.default_rng(47)
        source_basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
        target_basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
        angle = 0.71
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        latent = rng.normal(size=(512, 2))
        cal = rng.normal(size=(512, 2))
        source = latent @ source_basis.T
        target = latent @ rotation @ target_basis.T
        fitted = fit_hook_space_transport(target, source, np.ones(len(latent)), rank=2, ridge_fraction=1e-6)
        metrics = transport_metrics(cal @ source_basis.T, fitted.predict(cal @ rotation @ target_basis.T), np.ones(len(cal)))
        self.assertEqual(fitted.status, "OK")
        self.assertGreater(metrics.bcc, 0.999)
        self.assertLess(metrics.normalized_residual, 0.002)

    def test_split_merge_transport(self) -> None:
        rng = np.random.default_rng(48)
        latent = rng.normal(size=(600, 2))
        cal = rng.normal(size=(600, 2))
        source_mix = np.array([[1, 0], [0, 1], [0.4, 0.6], [0, 0]], dtype=float)
        target_mix = np.array([[0.4, 0], [0.6, 0], [0, 1], [0.3, 0.7]], dtype=float)
        fitted = fit_hook_space_transport(latent @ target_mix.T, latent @ source_mix.T, np.ones(len(latent)), rank=2, ridge_fraction=1e-6)
        metrics = transport_metrics(cal @ source_mix.T, fitted.predict(cal @ target_mix.T), np.ones(len(cal)))
        self.assertGreater(metrics.bcc, 0.999)
        self.assertLess(metrics.normalized_residual, 0.002)

    def test_query_null_global_nuisance_refuses(self) -> None:
        rng = np.random.default_rng(49)
        discovery = np.column_stack([rng.normal(size=(500, 2)), np.zeros(500)])
        positive = np.column_stack([rng.normal(size=(500, 2)), np.zeros(500)])
        negative = np.column_stack([rng.normal(size=(500, 2)), np.zeros(500)])
        fitted = fit_hook_space_transport(discovery, discovery, np.ones(len(discovery)), rank=2)
        pos = transport_metrics(positive, fitted.predict(positive), np.ones(len(positive)))
        neg = transport_metrics(negative, fitted.predict(negative), np.ones(len(negative)))
        decision = decide_transport_gate(pos, neg, rank_boundary_relative_gap=0.5, collision_improvement_over_global=0.2, raw_control_specificity=0.0, global_control_specificity=0.0)
        self.assertEqual(decision.decision, "UNRESOLVED_RELATION")
        self.assertEqual(decision.reason, "HARD_NEGATIVE_CONTRAST")

    def test_raw_control_must_be_beaten(self) -> None:
        pos = transport_metrics(np.ones((20, 1)), np.full((20, 1), 0.9), np.ones(20))
        neg = transport_metrics(np.zeros((20, 1)), np.full((20, 1), 0.01), np.ones(20))
        decision = decide_transport_gate(pos, neg, rank_boundary_relative_gap=0.5, collision_improvement_over_global=0.2, raw_control_specificity=1.0, global_control_specificity=0.1)
        self.assertEqual(decision.reason, "RAW_OR_GLOBAL_CONTROL_NOT_BEATEN")

    def test_rank_deficiency_is_explicit(self) -> None:
        rng = np.random.default_rng(50)
        latent = rng.normal(size=(300, 1))
        target = np.column_stack([latent[:, 0], 2 * latent[:, 0], np.zeros(len(latent))])
        source = np.column_stack([3 * latent[:, 0], -latent[:, 0], np.zeros(len(latent))])
        fitted = fit_hook_space_transport(target, source, np.ones(len(latent)), rank=2)
        self.assertEqual(fitted.status, "RANK_DEFICIENT")
        self.assertEqual(fitted.effective_rank, 1)


if __name__ == "__main__":
    unittest.main()
