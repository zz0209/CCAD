from __future__ import annotations

import unittest

import numpy as np

from ccad.mscc import minimum_support_contribution_correspondence, source_conditioned_topk_proposal
from ccad.nip_synthetic import FAMILIES, observed_kernels
from ccad.nip_synthetic_v3 import (
    N11_CENTERED_RESIDUAL,
    N11_CLIFF_NORMALIZED_MARGIN,
    N11_FEASIBILITY_THRESHOLD,
    N11_MAXIMUM_SMOOTH_RMSE,
    N11_MINIMUM_CLIFF_GAP,
    N11_MINIMUM_THRESHOLD_MARGIN,
    construction_certificate,
    evaluate_shared_hook_endpoint,
    generate_endpoint_observed,
)


class NIPSyntheticV3Tests(unittest.TestCase):
    def test_n11_is_close_but_not_pointwise_equal_and_remains_found(self):
        instance = generate_endpoint_observed("N11_downstream_cliff", structural_seed=101, sample_seed=202)
        source = instance.source_contributions[:, 0, :]
        target = instance.target_contributions[:, 0, :]
        self.assertGreater(float(np.max(np.abs(source - target))), 0.0)
        self.assertAlmostEqual(float(np.mean(target - source)), 0.0, places=14)
        k_ss, k_st, k_tt = observed_kernels(instance)
        proposal = source_conditioned_topk_proposal(
            k_ss, k_st, k_tt, source_atom_id=0, atom_cap=20, g_max=4,
            epsilon=1e-12, candidate_budget=7462, boundary_tie_tolerance=1e-12,
        )
        result = minimum_support_contribution_correspondence(
            k_ss, k_st, k_tt, instance.source_mean_contributions, instance.target_mean_contributions,
            source_atom_id=0, proposed_target_ids=proposal.proposed_target_ids, g_max=4,
            tau_ctr=N11_FEASIBILITY_THRESHOLD, tau_mu=N11_FEASIBILITY_THRESHOLD,
            epsilon=1e-12, candidate_budget=7462, complete_universe=False,
        )
        self.assertEqual(result.identification, "FOUND")
        self.assertEqual(tuple(item.target_ids for item in result.supports), ((0,),))
        self.assertAlmostEqual(result.supports[0].d_ctr, N11_CENTERED_RESIDUAL, places=12)

    def test_n11_endpoint_measures_cliff_failure_and_smooth_control(self):
        for pair in range(20):
            with self.subTest(pair=pair):
                instance = generate_endpoint_observed(
                    "N11_downstream_cliff", structural_seed=8100 + pair, sample_seed=9100 + pair
                )
                metrics = evaluate_shared_hook_endpoint(instance, (0,))
                self.assertEqual(metrics["cliff_disagreement_rate"], 1.0)
                self.assertGreaterEqual(metrics["cliff_effect_rmse"], N11_MINIMUM_CLIFF_GAP)
                self.assertGreaterEqual(
                    metrics["minimum_normalized_cliff_margin"], N11_CLIFF_NORMALIZED_MARGIN - 1e-12
                )
                self.assertLessEqual(metrics["smooth_effect_rmse"], N11_MAXIMUM_SMOOTH_RMSE)
                self.assertGreater(metrics["source_rms"], 0.0)
                self.assertGreater(metrics["raw_delta_rmse"], 0.0)
                self.assertGreater(metrics["minimum_raw_cliff_margin"], 0.0)
                self.assertAlmostEqual(
                    metrics["minimum_raw_cliff_margin"] / metrics["source_rms"],
                    metrics["minimum_normalized_cliff_margin"],
                    places=12,
                )

    def test_n11_certificate_has_non_circular_numeric_endpoint(self):
        instance = generate_endpoint_observed("N11_downstream_cliff", structural_seed=303, sample_seed=404)
        certificate = construction_certificate(instance)
        forbidden = {"causal_outcome", "truth", "identification", "minimum_supports"}
        self.assertFalse(set(certificate) & forbidden)
        self.assertTrue(certificate["endpoint_present"])
        self.assertAlmostEqual(certificate["n11_centered_residual"], N11_CENTERED_RESIDUAL, places=12)
        self.assertLessEqual(certificate["n11_centered_residual"], N11_FEASIBILITY_THRESHOLD)
        self.assertGreaterEqual(
            N11_FEASIBILITY_THRESHOLD - certificate["n11_centered_residual"],
            N11_MINIMUM_THRESHOLD_MARGIN,
        )
        self.assertLess(certificate["n11_sample_mean_delta_norm"], 1e-14)
        self.assertGreaterEqual(certificate["minimum_decoy_orthogonal_residual"], 0.075)
        self.assertTrue(certificate["cap_contract_pass"])

    def test_non_n11_families_are_observationally_unchanged_from_v2_contract(self):
        for family in FAMILIES:
            if family == "N11_downstream_cliff":
                continue
            with self.subTest(family=family):
                instance = generate_endpoint_observed(family, structural_seed=505, sample_seed=606)
                certificate = construction_certificate(instance)
                self.assertFalse(certificate["endpoint_present"])
                self.assertIsNone(certificate["n11_endpoint"])
                self.assertEqual(instance.target_contributions.shape[1], 20)

    def test_odd_n_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "even observation count"):
            generate_endpoint_observed("N11_downstream_cliff", structural_seed=1, sample_seed=2, n=33)

if __name__ == "__main__":
    unittest.main()
