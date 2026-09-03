from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from ccad.nip_metric_surface import native_support_metric_surface
from ccad.nip_synthetic_v3 import generate_endpoint_observed


class NIPMetricSurfaceTests(unittest.TestCase):
    def surface(self, family: str, support: tuple[int, ...]):
        observed = generate_endpoint_observed(family, structural_seed=123, sample_seed=456, n=512)
        return native_support_metric_surface(
            observed.source_contributions, observed.target_contributions,
            observed.source_mean_contributions, observed.target_mean_contributions,
            observed.document_ids, source_atom_id=0, target_ids=support, epsilon=1e-12,
        )

    def test_raw_bcc_and_residual_identities_are_persisted(self):
        metric = self.surface("N01_structured_split", (0, 1))
        self.assertAlmostEqual(
            metric["centered_residual_numerator"],
            metric["bcc_source_energy"] + metric["bcc_target_energy"] - 2.0 * metric["bcc_cross_inner"],
        )
        self.assertAlmostEqual(metric["d_ctr"], metric["centered_residual_numerator"] / metric["centered_source_energy_denominator"])
        self.assertAlmostEqual(metric["bcc_value"], 1.0)
        self.assertAlmostEqual(metric["bcc_normalized_residual"], 0.0)

    def test_mean_mismatch_is_separate_from_centered_match(self):
        metric = self.surface("N12_mean_mismatch", (0,))
        self.assertAlmostEqual(metric["d_ctr"], 0.0)
        self.assertAlmostEqual(metric["mean_residual_numerator"], 1.0)
        self.assertAlmostEqual(metric["d_mu"], 1.0, places=10)

    def test_cancellation_and_document_evidence_are_raw(self):
        cancellation = self.surface("N09_cancellation", (0, 1))
        rare = self.surface("N10_rare_occupancy", (0, 1))
        self.assertGreater(cancellation["cancellation_ratio"], 100.0)
        self.assertEqual(rare["active_document_count"], 2)
        self.assertLessEqual(rare["document_ess"], 2.01)

    def test_psc_uses_synthetic_atom_direction_spans(self):
        metric = self.surface("N06_exact_dense_orthogonal_rotation", (0, 1))
        self.assertEqual(metric["psc_status"], "OK")
        self.assertEqual(metric["psc_rank_source"], 1)
        self.assertEqual(metric["psc_rank_target"], 2)
        self.assertAlmostEqual(metric["psc_value"], 2.0 / 3.0, places=10)
        self.assertEqual(len(metric["psc_principal_angles_radians"]), 1)

    def test_n06_group_surface_separates_sum_metrics_from_atom_direction_psc(self):
        observed = generate_endpoint_observed(
            "N06_exact_dense_orthogonal_rotation", structural_seed=123,
            sample_seed=456, n=512,
        )
        metric = native_support_metric_surface(
            observed.source_contributions, observed.target_contributions,
            observed.source_mean_contributions, observed.target_mean_contributions,
            observed.document_ids, source_atom_id=None, source_atom_ids=(0, 1),
            target_ids=(0, 1), epsilon=1e-12,
        )
        self.assertAlmostEqual(metric["d_ctr"], 0.0)
        self.assertAlmostEqual(metric["bcc_value"], 1.0)
        self.assertEqual(metric["psc_rank_source"], 2)
        self.assertEqual(metric["psc_rank_target"], 2)
        self.assertAlmostEqual(metric["psc_value"], 1.0)

    def test_algorithm_fields_are_explicitly_not_applicable_until_scoring(self):
        metric = self.surface("N05_bloated_decoy", (0, 1))
        for key in ("proposal_recall", "conditional_solver_correctness", "solver_gap", "coverage"):
            self.assertEqual(metric[key]["status"], "NOT_APPLICABLE_PRELABEL")
            self.assertIsNone(metric[key]["value"])

    def test_surface_covers_every_frozen_mandatory_field(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/m1_nip_parent_completion_v1.json").read_text(encoding="utf-8"))
        metric = self.surface("N01_structured_split", (0, 1))
        self.assertFalse(set(config["mandatory_metric_fields"]) - set(metric))

    def test_invalid_support_fails_closed(self):
        observed = generate_endpoint_observed("N01_structured_split", structural_seed=1, sample_seed=2)
        with self.assertRaises(ValueError):
            native_support_metric_surface(
                observed.source_contributions, observed.target_contributions,
                observed.source_mean_contributions, observed.target_mean_contributions,
                observed.document_ids, source_atom_id=0, target_ids=(1, 1), epsilon=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
