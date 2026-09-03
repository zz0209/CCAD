from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from ccad.nip_baselines import IMPLEMENTED_CONTINUOUS_REFERENCES, IMPLEMENTED_NATIVE_LANES
from ccad.nip_synthetic_v2 import generate_cap_identifiable_observed


ROOT = Path(__file__).parents[1]


class ParentCompletionP1Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/m1_nip_parent_completion_p1_v1.json").read_text(encoding="utf-8"))
        self.p2 = json.loads((ROOT / "configs/m1_nip_parent_completion_p2_v1.json").read_text(encoding="utf-8"))

    def test_p1_is_fresh_smoke_and_formal_seeds_remain_ungenerated(self):
        self.assertEqual(self.config["phase"], "P1")
        self.assertEqual(self.config["pairs_per_family"], 1)
        self.assertEqual(self.config["formal_seed_manifest_status"], "UNGENERATED")
        self.assertFalse(self.config["formal_seed_consumed"])
        self.assertFalse(self.config["truth_opened_in_prediction"])
        self.assertFalse(self.config["evaluation_opened_in_prediction"])
        self.assertFalse(self.config["intervention_opened_in_prediction"])

    def test_p1_registry_and_row_count_are_exact(self):
        self.assertEqual(set(self.config["native_lanes"]), IMPLEMENTED_NATIVE_LANES | {"MSCC"})
        self.assertEqual(set(self.config["continuous_references"]), IMPLEMENTED_CONTINUOUS_REFERENCES)
        self.assertEqual(self.config["expected_prediction_rows"], 12 * 11)

    def test_runner_and_validator_have_no_truth_import(self):
        for name in ("run_m1_nip_parent_completion_p1.py", "validate_m1_nip_parent_completion_p1.py"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            tree = ast.parse(text)
            imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
            forbidden_reads = {
                node.slice.value for node in ast.walk(tree)
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "seeds" and isinstance(node.slice, ast.Constant)
                and node.slice.value in {"evaluation", "intervention"}
            }
            self.assertNotIn("ccad.nip_truth", imported)
            self.assertFalse(forbidden_reads)

    def test_required_artifact_contract_is_complete(self):
        required = set(self.config["required_artifacts"])
        self.assertTrue({"proposals.jsonl", "predictions.jsonl", "prediction_closure.json", "prelabel_validation.json", "seed_ledger.json"} <= required)

    def test_odd_mean_stream_never_requires_n11_endpoint(self):
        observed = generate_cap_identifiable_observed(
            "N11_downstream_cliff", structural_seed=913, sample_seed=1913,
            n=self.config["sample_sizes"]["mean"],
        )
        self.assertEqual(observed.source_mean_contributions.shape, (1, 1))
        self.assertEqual(observed.target_mean_contributions.shape, (1, 20))

    def test_p2_is_formal_fresh_and_scientifically_frozen_to_p1(self):
        self.assertEqual(self.p2["phase"], "P2")
        self.assertEqual(self.p2["pairs_per_family"], 20)
        self.assertEqual(self.p2["expected_prediction_rows"], 12 * 20 * 11)
        self.assertEqual(self.p2["formal_seed_manifest_status"], "UNGENERATED")
        self.assertFalse(self.p2["formal_seed_consumed"])
        self.assertTrue(self.p2["consume_formal_seeds_on_execution"])
        for key in (
            "families", "sample_sizes", "required_seed_streams", "g_max", "target_atom_count",
            "candidate_budget", "epsilon", "exact_tau_ctr", "exact_tau_mu",
            "approximate_tau_ctr", "approximate_tau_mu", "approximate_families",
            "tie_tolerance", "native_lanes", "continuous_references", "runtime_protocol",
            "random_diagnostic_replicates",
        ):
            self.assertEqual(self.p2[key], self.config[key], key)
        self.assertEqual(len(self.p2["p1_gate_bindings"]), 3)

    def test_phase_namespaces_produce_disjoint_seeds(self):
        import run_m1_nip_parent_completion_p1 as runner
        p1 = runner.seed_for("protocol", "code", "P1", "N01_structured_split", 0, "structural")
        p2 = runner.seed_for("protocol", "code", "P2", "N01_structured_split", 0, "structural")
        self.assertNotEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
