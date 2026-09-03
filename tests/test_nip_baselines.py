from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from ccad.nip_baselines import (
    IMPLEMENTED_CONTINUOUS_REFERENCES,
    IMPLEMENTED_NATIVE_LANES,
    infer_rank_one_codes,
    run_continuous_reference,
    run_native_baseline,
)
from ccad.nip_synthetic_v3 import generate_endpoint_observed


class NIPBaselineTests(unittest.TestCase):
    def observed(self, family: str):
        return generate_endpoint_observed(family, structural_seed=913, sample_seed=1913, n=512)

    def run_lane(self, lane: str, family: str, tau: float = 1e-10):
        observed = self.observed(family)
        return run_native_baseline(
            lane, observed.source_contributions[:, 0, :], observed.target_contributions,
            observed.source_mean_contributions[:, 0], observed.target_mean_contributions,
            g_max=4, tau_ctr=tau, tau_mu=tau, epsilon=1e-12,
            tie_tolerance=1e-12, solver_seed=77,
        )

    def test_api_has_no_truth_or_evaluation_inputs(self):
        forbidden = {"truth", "label", "planted_support", "evaluation", "intervention"}
        self.assertFalse(forbidden & set(inspect.signature(run_native_baseline).parameters))

    def test_contribution_singleton_and_pw_mcc_keep_atom_scope(self):
        nearest = self.run_lane("CONTRIBUTION_NEAREST_ATOM", "N11_downstream_cliff", tau=0.05)
        pw = self.run_lane("PW_MCC_HUNGARIAN", "N11_downstream_cliff", tau=0.05)
        self.assertEqual(nearest.supports, ((0,),))
        self.assertEqual(pw.diagnostics["scope"], "DEGENERATE_SINGLETON_BASELINE")

    def test_decoder_cosine_refuses_membership_tie_in_scalar_hook(self):
        result = self.run_lane("GREEDY_DECODER_COSINE", "N01_structured_split")
        self.assertEqual(result.status, "BUDGET_REFUSAL")
        self.assertEqual(result.terminal_reason, "BOUNDARY_TIE")

    def test_omp_refuses_normalization_induced_split_tie(self):
        result = self.run_lane("BINARY_FORWARD_OMP", "N01_structured_split")
        self.assertEqual(result.status, "BUDGET_REFUSAL")
        self.assertEqual(result.terminal_reason, "SELECTION_TIE")

    def test_omp_recovers_untied_unweighted_native_support(self):
        rng = np.random.default_rng(55)
        x = rng.standard_normal((512, 1))
        y = rng.standard_normal((512, 1))
        first = np.concatenate([0.3 * x, np.zeros_like(x)], axis=1)
        second = np.concatenate([np.zeros_like(y), 0.7 * y], axis=1)
        source = first + second
        targets = np.stack([first, second, np.concatenate([y, x], axis=1)], axis=1)
        result = run_native_baseline(
            "BINARY_FORWARD_OMP", source, targets, np.zeros(2), np.zeros((2, 3)),
            g_max=4, tau_ctr=1e-10, tau_mu=1e-10, epsilon=1e-12,
            tie_tolerance=1e-12, solver_seed=77,
        )
        self.assertEqual(result.identification, "FOUND")
        self.assertEqual(result.supports, ((0, 1),))
        self.assertFalse(result.diagnostics["coefficients_used_for_native_endpoint"])

    def test_mean_mismatch_remains_unresolved_for_omp(self):
        result = self.run_lane("BINARY_FORWARD_OMP", "N12_mean_mismatch")
        self.assertEqual(result.identification, "UNRESOLVED")

    def test_random_primary_is_deterministic_and_budget_small(self):
        first = self.run_lane("RANDOM_MATCHED_GROUP", "N01_structured_split")
        second = self.run_lane("RANDOM_MATCHED_GROUP", "N01_structured_split")
        self.assertEqual(first, second)
        self.assertLessEqual(first.evaluated_support_count, 4)
        self.assertEqual(first.diagnostics["diagnostic_replicates_not_run"], 32)

    def test_signed_and_nonnegative_references_fit_split(self):
        observed = self.observed("N01_structured_split")
        source = observed.source_contributions[:, 0, :]
        signed = run_continuous_reference("SIGNED_CONTINUOUS_REGRESSION", source, observed.target_contributions)
        nonnegative = run_continuous_reference("NONNEGATIVE_CONTINUOUS_REGRESSION", source, observed.target_contributions)
        self.assertLess(signed.discovery_residual_sq, 1e-18)
        self.assertLess(nonnegative.discovery_residual_sq, 1e-8)
        self.assertTrue(nonnegative.converged)

    def test_frozen_registry_is_fully_implemented(self):
        config = json.loads((Path(__file__).parents[1] / "configs" / "m1_nip_parent_completion_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(IMPLEMENTED_NATIVE_LANES, set(config["registered_native_lanes"]) - {"MSCC"})
        self.assertEqual(IMPLEMENTED_CONTINUOUS_REFERENCES, set(config["registered_non_native_references"]))

    def test_sinkhorn_lanes_are_deterministic_and_report_single_query_scope(self):
        for lane in ("DUSTBIN_SINKHORN", "OT_MASS_NATIVE_SUPPORT"):
            first = self.run_lane(lane, "N11_downstream_cliff", tau=0.05)
            second = self.run_lane(lane, "N11_downstream_cliff", tau=0.05)
            self.assertEqual(first, second)
            self.assertEqual(first.diagnostics["scope"], "DEGENERATE_SINGLE_QUERY")
            self.assertTrue(first.diagnostics["converged"])
            self.assertLessEqual(first.evaluated_support_count, 4)

    def test_spectral_lane_is_truth_free_deterministic_and_factorized(self):
        first = self.run_lane("SPECTRAL_LOCAL_SVD_NATIVE_SUPPORT", "N01_structured_split")
        second = self.run_lane("SPECTRAL_LOCAL_SVD_NATIVE_SUPPORT", "N01_structured_split")
        self.assertEqual(first, second)
        self.assertIn(first.identification, {"FOUND", "UNRESOLVED"})
        self.assertIn("cluster_count", first.diagnostics)
        self.assertLessEqual(max(first.diagnostics["target_factorization_residuals"]), 1e-12)

    def test_rank_one_code_inference_fails_closed_on_rank_two_process(self):
        contributions = np.zeros((4, 1, 2), dtype=np.float64)
        contributions[:2, 0, 0] = 1.0
        contributions[2:, 0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "not rank one"):
            infer_rank_one_codes(contributions)


if __name__ == "__main__":
    unittest.main()
