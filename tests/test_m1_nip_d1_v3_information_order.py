from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTOR = ROOT / "scripts/run_m1_nip_d1_predict_v3.py"
VALIDATOR = ROOT / "scripts/validate_m1_nip_d1_prediction_v3.py"


class D1V3InformationOrderTests(unittest.TestCase):
    def test_predictor_has_no_truth_or_orthogonal_outcome_import(self):
        tree = ast.parse(PREDICTOR.read_text(encoding="utf-8"))
        imports = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("nip_truth" in item for item in imports))

    def test_i1_and_formal_d1_use_distinct_declared_namespaces(self):
        i1 = json.loads((ROOT / "configs/m1_nip_i1_v3.json").read_text(encoding="utf-8"))
        d1 = json.loads((ROOT / "configs/m1_nip_d1_v3.json").read_text(encoding="utf-8"))
        self.assertEqual(i1["phase"], "I1")
        self.assertEqual(d1["phase"], "D1")
        self.assertFalse(i1["formal_d1_seed_consumed"])
        self.assertTrue(d1["formal_d1_seed_consumed"])
        self.assertEqual(i1["expected_prediction_rows"], 60)
        self.assertEqual(d1["expected_prediction_rows"], 1200)

    def test_configs_bind_protocol_diagnostics_and_n11_approximate_lane(self):
        for name in ("m1_nip_i1_v3.json", "m1_nip_d1_v3.json", "m1_nip_d2_v3.json"):
            config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256((ROOT / config["protocol_path"]).read_bytes()).hexdigest().upper(), config["protocol_sha256"])
            self.assertEqual(hashlib.sha256((ROOT / config["diagnostic_config_path"]).read_bytes()).hexdigest().upper(), config["diagnostic_config_sha256"])
            self.assertIn("N11_downstream_cliff", config["approximate_families"])
            self.assertFalse(config["truth_opened_in_prediction"])

    def test_d2_is_fresh_selected_cap_only(self):
        config = json.loads((ROOT / "configs/m1_nip_d2_v3.json").read_text(encoding="utf-8"))
        self.assertEqual(config["phase"], "D2")
        self.assertTrue(config["formal_d2_seed_consumed"])
        self.assertEqual(config["atom_caps"], [20])
        self.assertEqual(config["pairs_per_family"], 20)
        self.assertEqual(config["expected_prediction_rows"], 240)
        self.assertFalse(config["truth_opened_in_prediction"])

    def test_validator_is_prelabel_only(self):
        tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
        imports = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("nip_truth" in item for item in imports))


if __name__ == "__main__":
    unittest.main()
