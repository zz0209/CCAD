from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class R011F1CausalGateProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = json.loads((ROOT / "configs/r011f1_euclidean_causal_gate_v1.json").read_text(encoding="utf-8"))

    def test_gate_is_bounded_endpoint_blind_and_audit_closed(self) -> None:
        self.assertEqual(self.cfg["selected_units"], 8)
        self.assertEqual(self.cfg["sequences_per_unit"], 2)
        self.assertEqual(self.cfg["primary_endpoint"], "next_state")
        self.assertEqual(self.cfg["forbidden_splits"], ["audit"])
        self.assertFalse(self.cfg["audit_opened"])
        self.assertIn("selection_hash", self.cfg["unit_selection"])
        self.assertIn("source_query", self.cfg["sequence_selection"])

    def test_primary_must_beat_both_raw_and_global(self) -> None:
        progression = self.cfg["progression"]
        self.assertEqual(progression["within_control_rule"], "effect_consistency_gain_or_query_specificity_gain")
        self.assertGreater(progression["minimum_gain_against_each_raw_and_global_control"], 0)
        self.assertIn("RAW_HOOK_QUERY_PCA", self.cfg["evaluated_methods"])
        self.assertIn("GLOBAL_FCC_RELATION", self.cfg["evaluated_methods"])

    def test_mscc_refusal_is_not_scored_as_zero_intervention(self) -> None:
        self.assertIn("MSCC_REFUSAL", self.cfg["methods"])
        self.assertNotIn("MSCC_REFUSAL", self.cfg["evaluated_methods"])


if __name__ == "__main__":
    unittest.main()
