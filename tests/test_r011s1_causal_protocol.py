from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class R011S1CausalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs/r011s1_causal_calibration_screen_v1.json").read_text(encoding="utf-8"))

    def test_screen_is_bounded_rank_matched_and_audit_closed(self) -> None:
        self.assertEqual(self.config["selected_pairs"], self.config["candidate_ranks"][-2])
        self.assertEqual(self.config["rank"], 1)
        self.assertEqual(self.config["sequences_per_pair"], 2)
        self.assertEqual(self.config["forbidden_splits"], ["audit"])
        self.assertFalse(self.config["audit_opened"])

    def test_pair_and_sequence_selection_are_endpoint_blind(self) -> None:
        self.assertIn("query_selection_hash", self.config["pair_selection"])
        self.assertEqual(self.config["sequence_selection"], "highest_source_query_sum_squared_code_then_sequence_index")
        self.assertIn("all_rules_frozen_before_forward", self.config["threshold_source_split"])

    def test_raw_global_and_native_controls_are_mandatory(self) -> None:
        methods = set(self.config["methods"])
        self.assertTrue({"RAW_HOOK_QUERY_CONDITIONAL_PCA", "GLOBAL_SAE_PCA", "RELAXED_PAIRED_STITCHING", "MATCHED_RANK_RANDOM", "BEST_FUNCTIONAL_SINGLE_NATIVE"}.issubset(methods))
        self.assertIn("off_target", json.dumps(self.config["sae_specific_progression"]))


if __name__ == "__main__":
    unittest.main()
