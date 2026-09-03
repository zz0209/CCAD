from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from ccad.nip_diagnostics_v3 import evaluate_orthogonal_diagnostics
from ccad.nip_synthetic_v3 import generate_endpoint_observed


ROOT = Path(__file__).resolve().parents[1]


class NIPOrthogonalDiagnosticsV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "configs/m1_nip_orthogonal_diagnostics_v3.json").read_text(encoding="utf-8")
        )

    def test_n09_cancellation_is_measured_from_predicted_support_members(self):
        values = []
        for pair in range(20):
            observed = generate_endpoint_observed(
                "N09_cancellation", structural_seed=1000 + pair, sample_seed=2000 + pair
            )
            diagnostic = evaluate_orthogonal_diagnostics(observed, (0, 1))
            values.append(diagnostic.cancellation_energy_ratio)
        self.assertGreaterEqual(min(values), self.config["n09_minimum_unsafe_cancellation_energy_ratio"])

    def test_n10_document_level_evidence_is_insufficient(self):
        for pair in range(20):
            observed = generate_endpoint_observed(
                "N10_rare_occupancy", structural_seed=3000 + pair, sample_seed=4000 + pair
            )
            diagnostic = evaluate_orthogonal_diagnostics(observed, (0, 1))
            self.assertLess(
                diagnostic.source_active_document_count,
                self.config["n10_minimum_active_documents_for_sufficient_evidence"],
            )
            self.assertLessEqual(
                diagnostic.source_document_energy_kish_ess,
                self.config["n10_maximum_insufficient_document_energy_kish_ess"],
            )

    def test_n11_uses_supplied_support_and_persists_raw_scales(self):
        observed = generate_endpoint_observed(
            "N11_downstream_cliff", structural_seed=5001, sample_seed=5002
        )
        diagnostic = evaluate_orthogonal_diagnostics(observed, (0,))
        self.assertEqual(diagnostic.endpoint["target_support_size"], 1.0)
        self.assertGreater(diagnostic.endpoint["source_rms"], 0.0)
        self.assertGreater(diagnostic.endpoint["raw_delta_rmse"], 0.0)
        self.assertGreater(diagnostic.endpoint["minimum_raw_cliff_margin"], 0.0)
        wrong = evaluate_orthogonal_diagnostics(observed, (1,))
        self.assertNotEqual(wrong.endpoint["cliff_effect_rmse"], diagnostic.endpoint["cliff_effect_rmse"])

    def test_n12_mean_mismatch_is_recomputed_from_observed_means(self):
        observed = generate_endpoint_observed(
            "N12_mean_mismatch", structural_seed=6001, sample_seed=6002
        )
        diagnostic = evaluate_orthogonal_diagnostics(observed, (0,))
        self.assertGreater(diagnostic.d_mu, self.config["n12_mean_mismatch_tau"])
        repaired_means = replace(
            observed,
            target_mean_contributions=observed.source_mean_contributions.copy(),
        )
        self.assertAlmostEqual(evaluate_orthogonal_diagnostics(repaired_means, (0,)).d_mu, 0.0)

    def test_support_contract_fails_closed(self):
        observed = generate_endpoint_observed("N09_cancellation", structural_seed=7, sample_seed=8)
        for invalid in ((), (1, 0), (0, 0), (20,)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate_orthogonal_diagnostics(observed, invalid)


if __name__ == "__main__":
    unittest.main()
