from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class ParentCompletionP1ScoreTests(unittest.TestCase):
    def test_truth_import_occurs_only_inside_main_after_closure_verification(self):
        path = ROOT / "scripts/score_m1_nip_parent_completion_p1.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        static_truth = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "ccad.nip_truth"]
        self.assertFalse(static_truth)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        verify_line = min(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "verify_closure")
        dynamic_line = min(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
        self.assertLess(verify_line, dynamic_line)

    def test_score_rows_has_no_selection_api(self):
        text = (ROOT / "scripts/score_m1_nip_parent_completion_p1.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        score_rows = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score_rows")
        imported_calls = {node.func.id for node in ast.walk(score_rows) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("run_native_baseline", imported_calls)
        self.assertNotIn("minimum_support_contribution_correspondence", imported_calls)

    def test_score_validator_recomputes_raw_identities(self):
        text = (ROOT / "scripts/validate_m1_nip_parent_completion_p1_score.py").read_text(encoding="utf-8")
        self.assertIn('checks["raw_native_identities"]', text)
        self.assertIn('checks["continuous_evaluation_recomputed"]', text)
        self.assertIn('expected_n08_controls = 2 * config["pairs_per_family"]', text)
        self.assertIn('row["false_unique"] ==', text)
        self.assertIn('family_cluster_standard_error', text)
        self.assertIn('evaluate_shared_hook_endpoint', text)

    def test_mscc_algorithm_diagnostics_are_typed_and_use_persisted_competitor(self):
        from score_m1_nip_parent_completion_p1 import _algorithm_diagnostics
        row = {
            "lane": "MSCC",
            "prediction": {
                "supports": [{"target_ids": [0, 1]}], "multiplicity": "UNIQUE",
                "best_candidate": {"d_ctr": 0.01, "d_mu": 0.02},
                "nearest_competitor": {"d_ctr": 0.08, "d_mu": 0.03},
                "unresolved_reason": None,
            },
        }
        diagnostics = _algorithm_diagnostics(row, True, True)
        self.assertEqual(diagnostics["nearest_competitor_margin"]["status"], "MEASURED")
        self.assertAlmostEqual(diagnostics["nearest_competitor_margin"]["value"], 0.06)
        self.assertTrue(diagnostics["solver_gap"]["reason"])
        self.assertTrue(diagnostics["proposal_stability"]["reason"])


if __name__ == "__main__":
    unittest.main()
