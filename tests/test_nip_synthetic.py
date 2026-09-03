from __future__ import annotations

import ast
import inspect
import unittest

import ccad.nip_synthetic as observed_module
from ccad.mscc import minimum_support_contribution_correspondence
from ccad.nip_synthetic import FAMILIES, assert_observed_schema_truth_free, generate_nip_observed, observed_kernels
from ccad.nip_truth import nip_truth


class NIPSyntheticRegistryTests(unittest.TestCase):
    def test_registry_has_exactly_the_twelve_frozen_families(self):
        self.assertEqual(len(FAMILIES), 12)
        self.assertEqual(len(set(FAMILIES)), 12)
        assert_observed_schema_truth_free()

    def test_observed_generator_module_does_not_import_truth_registry(self):
        tree = ast.parse(inspect.getsource(observed_module))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any("nip_truth" in name for name in imported))

    def test_every_family_is_deterministic_and_observed_shapes_are_valid(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                first = generate_nip_observed(family, structural_seed=101, sample_seed=202, n=256)
                second = generate_nip_observed(family, structural_seed=101, sample_seed=202, n=256)
                truth = nip_truth(family)
                self.assertEqual(first.source_contributions.shape, second.source_contributions.shape)
                self.assertEqual(first.target_contributions.shape, second.target_contributions.shape)
                self.assertEqual(first.source_contributions.tobytes(), second.source_contributions.tobytes())
                self.assertEqual(first.target_contributions.tobytes(), second.target_contributions.tobytes())
                self.assertEqual(truth.family_id, family)
                self.assertEqual(first.document_ids.shape[0], 256)
                changed = generate_nip_observed(family, structural_seed=102, sample_seed=202, n=256)
                structural_change = (
                    first.source_contributions.tobytes() != changed.source_contributions.tobytes()
                    or first.target_contributions.tobytes() != changed.target_contributions.tobytes()
                )
                self.assertTrue(structural_change)

    def test_complete_oracle_matches_exact_family_identification(self):
        exact_families = [family for family in FAMILIES if family != "N10_rare_occupancy"]
        for family in exact_families:
            with self.subTest(family=family):
                observed = generate_nip_observed(family, structural_seed=303, sample_seed=404, n=2048)
                truth = nip_truth(family)
                k_ss, k_st, k_tt = observed_kernels(observed)
                result = minimum_support_contribution_correspondence(
                    k_ss, k_st, k_tt,
                    observed.source_mean_contributions, observed.target_mean_contributions,
                    source_atom_id=0, proposed_target_ids=tuple(range(k_st.shape[1])),
                    g_max=4, tau_ctr=1e-10 if family != "N07_margin_separated_approximate_rotation" else 0.05,
                    tau_mu=1e-10 if family != "N07_margin_separated_approximate_rotation" else 0.05,
                    epsilon=1e-12, candidate_budget=7462, complete_universe=True,
                )
                self.assertEqual(result.identification, truth.identification)
                self.assertEqual(result.multiplicity, truth.multiplicity)
                self.assertEqual(tuple(item.target_ids for item in result.supports), truth.minimum_supports)

    def test_rotation_and_continuous_only_controls_are_explicit_truth_attributes(self):
        rotation = nip_truth(FAMILIES[5])
        continuous = nip_truth(FAMILIES[7])
        self.assertTrue(rotation.full_group_portable)
        self.assertFalse(rotation.continuous_reference_feasible)
        self.assertTrue(continuous.continuous_reference_feasible)
        self.assertFalse(continuous.full_group_portable)


if __name__ == "__main__":
    unittest.main()
