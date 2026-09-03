from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from ccad.nip_baselines import IMPLEMENTED_CONTINUOUS_REFERENCES, IMPLEMENTED_NATIVE_LANES


ROOT = Path(__file__).parents[1]


class ParentCompletionP1Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/m1_nip_parent_completion_p1_v1.json").read_text(encoding="utf-8"))

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


if __name__ == "__main__":
    unittest.main()
