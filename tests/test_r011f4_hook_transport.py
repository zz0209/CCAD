from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from ccad.hook_transport import (
    decide_transport_gate,
    fit_basis_constrained_transport,
    fit_hook_space_transport,
    fit_nuisance_projector,
    residualize_hook_process,
    transport_prefix,
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

    def test_basis_constrained_prefixes_are_nested(self) -> None:
        rng = np.random.default_rng(52)
        target = rng.normal(size=(300, 5))
        basis, _ = np.linalg.qr(rng.normal(size=(6, 3)))
        coefficient = rng.normal(size=(5, 3))
        coordinates = target @ coefficient
        fitted = fit_basis_constrained_transport(target, coordinates, basis, np.ones(300), ridge_fraction=1e-6)
        rank_one = transport_prefix(fitted, 1)
        rank_two = transport_prefix(fitted, 2)
        np.testing.assert_allclose(rank_one.target_factors, rank_two.target_factors[:, :1])
        np.testing.assert_allclose(rank_one.source_factors, rank_two.source_factors[:, :1])
        metric = transport_metrics(coordinates[:, :2] @ basis[:, :2].T, rank_two.predict(target), np.ones(300))
        self.assertGreater(metric.bcc, .999)

    def test_nuisance_rank_is_smallest_variance_prefix(self) -> None:
        rng = np.random.default_rng(53)
        process = rng.normal(size=(4000, 3)) * np.sqrt([8.0, 2.0, .2])
        nuisance = fit_nuisance_projector(process, np.ones(len(process)), explained_variance_threshold=.9, maximum_rank=3)
        self.assertEqual(nuisance.status, "OK")
        self.assertEqual(nuisance.rank, 2)
        self.assertGreaterEqual(nuisance.explained_variance_fraction, .9)

    def test_residualization_preserves_orthogonal_query_signal(self) -> None:
        rng = np.random.default_rng(54)
        signal = rng.normal(size=(300, 1))
        process = np.column_stack([rng.normal(size=300), signal[:, 0], np.zeros(300)])
        nuisance_basis = np.array([[1.0], [0.0], [0.0]])
        residual = residualize_hook_process(process, nuisance_basis)
        np.testing.assert_allclose(residual[:, 1], signal[:, 0])
        np.testing.assert_allclose(residual[:, 0], 0.0)


class HookTransportProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.cfg = json.loads((root / "configs/r011f4_hook_transport_real_v2.json").read_text(encoding="utf-8"))

    def test_audit_and_execution_boundary(self) -> None:
        self.assertTrue(self.cfg["execution_enabled"])
        self.assertFalse(self.cfg["audit_opened"])
        self.assertEqual(self.cfg["forbidden_splits"], ["audit"])
        self.assertEqual(self.cfg["splits"], ["discovery", "calibration"])

    def test_replication_and_controls_are_frozen(self) -> None:
        self.assertEqual(self.cfg["anchor_units"], 160)
        self.assertEqual(self.cfg["candidate_ranks"], [1, 2, 4, 8])
        self.assertEqual(self.cfg["ridge_fraction"], .001)
        self.assertEqual(set(self.cfg["controls"]), {"query_conditioned_raw_hook_transport", "query_agnostic_whole_sae_global_transport"})

    def test_meaningful_transfer_gate_is_not_weakened(self) -> None:
        self.assertEqual(self.cfg["minimum_calibration_bcc"], .8)
        self.assertEqual(self.cfg["maximum_calibration_normalized_residual"], .2)
        self.assertEqual(self.cfg["minimum_control_specificity_advantage"], .05)


if __name__ == "__main__":
    unittest.main()
