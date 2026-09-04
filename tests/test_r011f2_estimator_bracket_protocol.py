from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class R011F2EstimatorBracketProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = json.loads((ROOT / "configs/r011f2_estimator_bracket_v1.json").read_text(encoding="utf-8"))

    def test_only_two_alternatives_and_no_audit(self) -> None:
        self.assertEqual(self.cfg["estimators"], ["ENERGY_BALANCED_PLS", "DIAGONAL_WHITENED_CORRELATION"])
        self.assertFalse(self.cfg["audit_opened"])
        self.assertEqual(self.cfg["forbidden_splits"], ["audit"])

    def test_meaningful_transfer_gate_inherits_causal_tolerances(self) -> None:
        self.assertEqual(self.cfg["minimum_calibration_bcc"], 0.8)
        self.assertEqual(self.cfg["maximum_calibration_normalized_residual"], 0.2)
        self.assertEqual(self.cfg["minimum_progression_coverage"], 0.10)
        self.assertTrue(self.cfg["require_all_represented_ordered_directions"])


if __name__ == "__main__":
    unittest.main()
