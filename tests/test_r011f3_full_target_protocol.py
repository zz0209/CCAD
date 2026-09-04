from __future__ import annotations
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class R011F3FullTargetProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = json.loads((ROOT / "configs/r011f3_full_target_v1.json").read_text(encoding="utf-8"))

    def test_single_complete_target_setting_and_audit_closed(self) -> None:
        self.assertEqual(self.cfg["target_candidate_count"], self.cfg["num_latents"])
        self.assertEqual(self.cfg["feature_pair_budget"], 32 * 3072)
        self.assertFalse(self.cfg["audit_opened"])
        self.assertEqual(self.cfg["forbidden_splits"], ["audit"])

    def test_transfer_gate_is_unchanged(self) -> None:
        self.assertEqual(self.cfg["minimum_calibration_bcc"], .8)
        self.assertEqual(self.cfg["maximum_calibration_normalized_residual"], .2)
        self.assertEqual(self.cfg["minimum_progression_coverage"], .1)

if __name__ == "__main__": unittest.main()
