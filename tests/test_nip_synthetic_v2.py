from __future__ import annotations

import unittest

import numpy as np

from ccad.mscc import minimum_support_contribution_correspondence, source_conditioned_topk_proposal
from ccad.nip_synthetic import FAMILIES, observed_kernels
from ccad.nip_synthetic_v2 import CAP_PRESSURE, DECOY_ORTHOGONAL_ENERGY, REGISTERED_CAPS, TARGET_ATOM_COUNT, construction_certificate, generate_cap_identifiable_observed
from ccad.nip_truth import nip_truth


class NIPSyntheticV2ProbeTests(unittest.TestCase):
    def _result(self, family: str, cap: int, structural_seed: int = 1101, sample_seed: int = 2202):
        observed = generate_cap_identifiable_observed(family, structural_seed=structural_seed, sample_seed=sample_seed)
        k_ss, k_st, k_tt = observed_kernels(observed)
        proposal = source_conditioned_topk_proposal(
            k_ss, k_st, k_tt, source_atom_id=0, atom_cap=cap, g_max=4,
            epsilon=1e-12, candidate_budget=7462, boundary_tie_tolerance=1e-12,
        )
        tau = 0.05 if family.startswith("N07") else 1e-10
        result = minimum_support_contribution_correspondence(
            k_ss, k_st, k_tt, observed.source_mean_contributions, observed.target_mean_contributions,
            source_atom_id=0, proposed_target_ids=proposal.proposed_target_ids, g_max=4,
            tau_ctr=tau, tau_mu=tau, epsilon=1e-12, candidate_budget=7462,
            complete_universe=cap == TARGET_ATOM_COUNT,
        )
        return observed, proposal, result

    def test_every_family_has_twenty_targets_and_no_boundary_refusal(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                observed, proposal, _ = self._result(family, 20)
                self.assertEqual(observed.target_contributions.shape[1], TARGET_ATOM_COUNT)
                self.assertEqual(proposal.status, "OK")
                self.assertEqual(proposal.planned_support_count, 6195)

    def test_complete_oracle_preserves_v1_truth(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                _, _, result = self._result(family, 20)
                truth = nip_truth(family)
                self.assertEqual(result.identification, truth.identification)
                self.assertEqual(result.multiplicity, truth.multiplicity)
                self.assertEqual(tuple(item.target_ids for item in result.supports), truth.minimum_supports)

    def test_registered_cap_pressure_is_identifiable_on_positive_families(self):
        for family, expected_cap in CAP_PRESSURE.items():
            with self.subTest(family=family):
                truth = nip_truth(family)
                successful = []
                for cap in REGISTERED_CAPS:
                    _, _, result = self._result(family, cap)
                    supports = tuple(item.target_ids for item in result.supports)
                    if result.identification == truth.identification and result.multiplicity == truth.multiplicity and supports == truth.minimum_supports:
                        successful.append(cap)
                self.assertTrue(successful)
                self.assertEqual(min(successful), expected_cap)

    def test_decoy_orthogonal_component_exceeds_approximate_threshold(self):
        self.assertGreater(DECOY_ORTHOGONAL_ENERGY, 0.05)
        self.assertGreaterEqual(DECOY_ORTHOGONAL_ENERGY - 0.05, 0.02)

    def test_construction_certificate_is_numeric_and_truth_label_free(self):
        forbidden = {"identification", "multiplicity", "minimum_supports", "causal_outcome", "truth"}
        for family in FAMILIES:
            with self.subTest(family=family):
                observed = generate_cap_identifiable_observed(family, structural_seed=1101, sample_seed=2202)
                certificate = construction_certificate(observed)
                self.assertFalse(set(certificate) & forbidden)
                self.assertGreaterEqual(certificate["minimum_decoy_orthogonal_residual"], 0.075)
                self.assertLess(certificate["maximum_decoy_orthogonality_error"], 1e-10)
                if family in CAP_PRESSURE:
                    self.assertTrue(certificate["cap_contract_pass"])
                else:
                    self.assertIsNone(certificate["cap_contract_pass"])

    def test_cap_pressure_and_oracle_truth_hold_across_five_probe_pairs(self):
        for pair_index in range(5):
            structural_seed = 8100 + pair_index
            sample_seed = 9100 + pair_index
            for family, expected_cap in CAP_PRESSURE.items():
                with self.subTest(pair=pair_index, family=family):
                    truth = nip_truth(family)
                    successful = []
                    for cap in REGISTERED_CAPS:
                        _, _, result = self._result(family, cap, structural_seed, sample_seed)
                        supports = tuple(item.target_ids for item in result.supports)
                        if result.identification == truth.identification and result.multiplicity == truth.multiplicity and supports == truth.minimum_supports:
                            successful.append(cap)
                    self.assertEqual(min(successful), expected_cap)
                    _, _, oracle = self._result(family, 20, structural_seed, sample_seed)
                    self.assertEqual(oracle.identification, truth.identification)
                    self.assertEqual(tuple(item.target_ids for item in oracle.supports), truth.minimum_supports)


if __name__ == "__main__":
    unittest.main()
