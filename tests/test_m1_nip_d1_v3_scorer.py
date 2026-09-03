from __future__ import annotations

import ast, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "scripts/score_m1_nip_d1_v3.py"


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_m1_nip_d1_v3", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class D1V3ScorerTests(unittest.TestCase):
    def test_truth_is_only_dynamically_imported(self):
        tree = ast.parse(SCORER.read_text(encoding="utf-8"))
        imports = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("nip_truth" in item for item in imports))

    def test_tamper_fails_before_score_directory_or_truth_import(self):
        scorer = load_scorer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "sealed"
            run.mkdir()
            (run / "prediction_closure.json").write_text(json.dumps({"state": "SEALED", "run_id": "sealed", "files": {"predictions.raw.jsonl": "0" * 64}}), encoding="utf-8")
            (run / "predictions.raw.jsonl").write_text("tampered\n", encoding="utf-8")
            with patch.object(scorer.importlib, "import_module") as dynamic_import:
                with self.assertRaises(ValueError):
                    scorer.score_prediction_run(run, root / "score")
                dynamic_import.assert_not_called()
                self.assertFalse((root / "score").exists())

    def test_n11_uses_named_cliff_effect_field(self):
        scorer = load_scorer()
        diagnostic = json.loads((ROOT / "configs/m1_nip_orthogonal_diagnostics_v3.json").read_text(encoding="utf-8"))
        row = {"family_id": "N11_downstream_cliff", "seeds": {"structural": 101, "sample": 102}, "prediction": {"identification": "FOUND", "supports": [{"target_ids": [0]}]}}
        measured = scorer.orthogonal_measurement(row, diagnostic, 512)
        self.assertEqual(measured["measured_attribute"], "CAUSAL_FAIL")
        self.assertGreaterEqual(measured["endpoint"]["cliff_effect_rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
