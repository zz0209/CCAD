from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class R011S1ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs/r011s1_calibration_feasibility_v1.json").read_text(encoding="utf-8"))

    def test_split_order_and_fixed_rank_family_prevent_audit_selection(self) -> None:
        self.assertEqual(self.config["mean_constants_source_split"], "mean")
        self.assertEqual(self.config["allowed_splits"], ["mean", "discovery", "calibration"])
        self.assertEqual(self.config["forbidden_splits"], ["audit"])
        self.assertFalse(self.config["audit_opened"])
        self.assertEqual(self.config["candidate_ranks"], [1, 2, 4, 8, 16])

    def test_query_subset_is_source_only_and_stratified(self) -> None:
        self.assertEqual(self.config["queries_per_seed"], self.config["strata_per_seed"])
        self.assertEqual(
            self.config["query_selection"],
            "lowest_existing_r009b_selection_hash_per_seed_and_energy_stratum",
        )
        self.assertEqual(self.config["max_condition_tokens_per_split"], 256)

    def test_required_nontrivial_controls_and_progression_rule_are_frozen(self) -> None:
        methods = set(self.config["methods"])
        self.assertTrue({
            "RAW_HOOK_QUERY_CONDITIONAL_PCA", "GLOBAL_SAE_PCA", "MATCHED_RANK_RANDOM",
            "RELAXED_PAIRED_STITCHING", "BEST_FUNCTIONAL_SINGLE_NATIVE", "MSCC_NATIVE_REFUSAL",
        }.issubset(methods))
        self.assertTrue(self.config["screen_progression_rule"]["raw_hook_equal_or_better_requires_causal_nontriviality_test"])


if __name__ == "__main__":
    unittest.main()
