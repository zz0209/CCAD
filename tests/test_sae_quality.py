from __future__ import annotations

import unittest

from ccad.sae_quality import budget_stability_checks, ce_recovered, select_k_bracket, select_k_extension


class SaeQualityTests(unittest.TestCase):
    def test_clean_reconstruction_recovers_all_damage(self) -> None:
        self.assertEqual(ce_recovered(2.0, 2.0, 4.0), 1.0)

    def test_zero_reconstruction_recovers_no_damage(self) -> None:
        self.assertEqual(ce_recovered(2.0, 4.0, 4.0), 0.0)

    def test_worse_than_zero_is_negative(self) -> None:
        self.assertEqual(ce_recovered(2.0, 5.0, 4.0), -0.5)

    def test_nonpositive_or_nonfinite_denominator_is_rejected(self) -> None:
        for values in ((2.0, 2.1, 2.0), (2.0, 2.1, 1.0), (float("nan"), 2.1, 4.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ce_recovered(*values)

    def test_budget_stability_boundary_is_inclusive(self) -> None:
        result = budget_stability_checks(
            {"fve": 0.60, "ce_recovered": 0.50, "alive_fraction": 0.40, "c_dec": 0.20},
            {"fve": 0.65, "ce_recovered": 0.45, "alive_fraction": 0.50, "c_dec": 0.22},
            {"fve_abs": 0.05, "ce_recovered_abs": 0.05, "alive_fraction_abs": 0.10, "c_dec_relative": 0.10},
        )
        self.assertTrue(result["pass"])

    def test_budget_stability_identifies_failed_component(self) -> None:
        result = budget_stability_checks(
            {"fve": 0.60, "ce_recovered": 0.50, "alive_fraction": 0.40, "c_dec": 0.20},
            {"fve": 0.66, "ce_recovered": 0.50, "alive_fraction": 0.40, "c_dec": 0.20},
            {"fve_abs": 0.05, "ce_recovered_abs": 0.05, "alive_fraction_abs": 0.10, "c_dec_relative": 0.10},
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["fve_abs"])

    def test_budget_stability_rejects_zero_reference_cdec(self) -> None:
        with self.assertRaises(ValueError):
            budget_stability_checks(
                {"fve": 0.60, "ce_recovered": 0.50, "alive_fraction": 0.40, "c_dec": 0.0},
                {"fve": 0.60, "ce_recovered": 0.50, "alive_fraction": 0.40, "c_dec": 0.0},
                {"fve_abs": 0.05, "ce_recovered_abs": 0.05, "alive_fraction_abs": 0.10, "c_dec_relative": 0.10},
            )

    def test_k_bracket_applies_joint_eligibility_and_geometry(self) -> None:
        result = select_k_bracket({
            8: {"fve": 0.90, "ce_recovered": 0.80, "c_dec": 0.040},
            16: {"fve": 0.94, "ce_recovered": 0.91, "c_dec": 0.030},
            32: {"fve": 0.96, "ce_recovered": 0.93, "c_dec": 0.035},
            64: {"fve": 0.97, "ce_recovered": 0.94, "c_dec": 0.034},
        })
        self.assertEqual(result["eligible_k"], [16, 32, 64])
        self.assertEqual(result["shortlist_k"], [16, 32])
        self.assertEqual(result["decision"], "TWO_SEED_PILOT")

    def test_k_bracket_expands_for_only_upper_boundary(self) -> None:
        result = select_k_bracket({
            8: {"fve": 0.80, "ce_recovered": 0.70, "c_dec": 0.030},
            16: {"fve": 0.85, "ce_recovered": 0.75, "c_dec": 0.029},
            32: {"fve": 0.90, "ce_recovered": 0.80, "c_dec": 0.031},
            64: {"fve": 0.96, "ce_recovered": 0.90, "c_dec": 0.028},
        })
        self.assertEqual(result["eligible_k"], [64])
        self.assertEqual(result["decision"], "EXPAND_TO_64_128")
        self.assertEqual(result["shortlist_k"], [])

    def test_k_bracket_expands_for_strictly_monotonic_geometry(self) -> None:
        result = select_k_bracket({
            8: {"fve": 0.95, "ce_recovered": 0.91, "c_dec": 0.020},
            16: {"fve": 0.96, "ce_recovered": 0.92, "c_dec": 0.025},
            32: {"fve": 0.97, "ce_recovered": 0.93, "c_dec": 0.030},
            64: {"fve": 0.98, "ce_recovered": 0.94, "c_dec": 0.035},
        })
        self.assertTrue(result["c_dec_strictly_monotonic"])
        self.assertEqual(result["decision"], "EXPAND_TO_64_128")

    def test_k_extension_stops_when_only_new_upper_boundary_is_eligible(self) -> None:
        result = select_k_extension({
            8: {"fve": 0.80, "ce_recovered": 0.60, "c_dec": 0.030},
            16: {"fve": 0.85, "ce_recovered": 0.65, "c_dec": 0.032},
            32: {"fve": 0.90, "ce_recovered": 0.70, "c_dec": 0.034},
            64: {"fve": 0.94, "ce_recovered": 0.80, "c_dec": 0.033},
            128: {"fve": 0.98, "ce_recovered": 0.90, "c_dec": 0.031},
        })
        self.assertEqual(result["eligible_k"], [128])
        self.assertEqual(result["decision"], "UNBOUNDED_HIGH")
        self.assertEqual(result["shortlist_k"], [])

    def test_k_extension_can_emit_internal_two_seed_shortlist(self) -> None:
        result = select_k_extension({
            8: {"fve": 0.80, "ce_recovered": 0.60, "c_dec": 0.030},
            16: {"fve": 0.90, "ce_recovered": 0.80, "c_dec": 0.032},
            32: {"fve": 0.96, "ce_recovered": 0.92, "c_dec": 0.035},
            64: {"fve": 0.98, "ce_recovered": 0.94, "c_dec": 0.033},
            128: {"fve": 0.97, "ce_recovered": 0.93, "c_dec": 0.034},
        })
        self.assertEqual(result["eligible_k"], [32, 64, 128])
        self.assertEqual(result["shortlist_k"], [64, 128])
        self.assertEqual(result["decision"], "TWO_SEED_PILOT")

    def test_k_extension_stops_when_best_endpoints_split(self) -> None:
        result = select_k_extension({
            8: {"fve": 0.99, "ce_recovered": 0.70, "c_dec": 0.030},
            16: {"fve": 0.90, "ce_recovered": 0.80, "c_dec": 0.032},
            32: {"fve": 0.91, "ce_recovered": 0.81, "c_dec": 0.034},
            64: {"fve": 0.92, "ce_recovered": 0.82, "c_dec": 0.033},
            128: {"fve": 0.93, "ce_recovered": 0.99, "c_dec": 0.031},
        })
        self.assertEqual(result["eligible_k"], [])
        self.assertEqual(result["decision"], "NO_JOINT_ELIGIBLE")


if __name__ == "__main__":
    unittest.main()
