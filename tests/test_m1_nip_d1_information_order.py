from __future__ import annotations

import ast, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCORER_PATH = ROOT / "scripts/score_m1_nip_d1_v2.py"
PREDICTOR_PATH = ROOT / "scripts/run_m1_nip_d1_predict_v2.py"


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_m1_nip_d1_v2", SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class D1InformationOrderTests(unittest.TestCase):
    def test_predictor_and_scorer_have_no_static_truth_import(self):
        for path in (PREDICTOR_PATH, SCORER_PATH):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            self.assertFalse(any("nip_truth" in ast.unparse(node) for node in imports))

    def test_tamper_fails_before_dynamic_truth_import(self):
        scorer = load_scorer()
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "sealed_run"
            run.mkdir()
            (run / "prediction_closure.json").write_text(json.dumps({"state": "SEALED", "run_id": "sealed_run", "files": {"predictions.raw.jsonl": "0" * 64}}), encoding="utf-8")
            (run / "predictions.raw.jsonl").write_text("tampered\n", encoding="utf-8")
            with patch.object(scorer.importlib, "import_module") as dynamic_import:
                with self.assertRaises(ValueError):
                    scorer.score_prediction_run(run, Path(temporary) / "scores")
                dynamic_import.assert_not_called()
                self.assertFalse((Path(temporary) / "scores").exists())

    def test_i1_does_not_consume_formal_d1_seed(self):
        config = json.loads((ROOT / "configs/m1_nip_i1_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(config["phase"], "I1")
        self.assertFalse(config["formal_d1_seed_consumed"])
        self.assertFalse(config["truth_opened_in_prediction"])

    def test_formal_d1_grid_matches_frozen_protocol(self):
        config = json.loads((ROOT / "configs/m1_nip_d1_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(config["phase"], "D1")
        self.assertTrue(config["formal_d1_seed_consumed"])
        self.assertEqual(config["pairs_per_family"], 20)
        self.assertEqual(config["expected_prediction_rows"], 12 * 20 * 5)
        self.assertFalse(config["truth_opened_in_prediction"])


if __name__ == "__main__":
    unittest.main()
