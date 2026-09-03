from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from ccad.nip_synthetic_v3 import (
    N11_CENTERED_RESIDUAL,
    N11_CLIFF_NORMALIZED_MARGIN,
    N11_FEASIBILITY_THRESHOLD,
    N11_MAXIMUM_SMOOTH_RMSE,
    N11_MINIMUM_CLIFF_GAP,
)


ROOT = Path(__file__).resolve().parents[1]


class NIPProtocolV3Tests(unittest.TestCase):
    def test_protocol_binds_document_and_endpoint_constants(self):
        config = json.loads((ROOT / "configs/m1_nip_protocol_v3.json").read_text(encoding="utf-8"))
        document = ROOT / config["protocol_document"]
        self.assertEqual(hashlib.sha256(document.read_bytes()).hexdigest().upper(), config["protocol_sha256"])
        self.assertEqual(config["selected_option"], "A_PROSPECTIVE_OBSERVABLE_N11_ENDPOINT")
        self.assertEqual(config["n11"]["centered_residual"], N11_CENTERED_RESIDUAL)
        self.assertEqual(config["n11"]["tau_ctr"], N11_FEASIBILITY_THRESHOLD)
        self.assertEqual(config["n11"]["minimum_normalized_cliff_margin"], N11_CLIFF_NORMALIZED_MARGIN)
        self.assertEqual(config["n11"]["minimum_cliff_effect_rmse"], N11_MINIMUM_CLIFF_GAP)
        self.assertEqual(config["n11"]["maximum_smooth_effect_rmse"], N11_MAXIMUM_SMOOTH_RMSE)

    def test_protocol_freezes_fresh_unopened_phase_namespaces(self):
        config = json.loads((ROOT / "configs/m1_nip_protocol_v3.json").read_text(encoding="utf-8"))
        self.assertFalse(config["execution_enabled"])
        self.assertFalse(config["synthetic_evaluation_opened"])
        self.assertFalse(config["real_sae_audit_opened"])
        self.assertTrue(all(phase["seed_manifest_status"] == "UNGENERATED" for phase in config["phases"].values()))
        self.assertNotIn("RUNTIME", " ".join(config["selection_order"]))

    def test_d0_marks_n11_as_approximate_and_truth_closed(self):
        config = json.loads((ROOT / "configs/m1_nip_d0_v3.json").read_text(encoding="utf-8"))
        self.assertIn("N11_downstream_cliff", config["approximate_families"])
        self.assertFalse(config["truth_opened"])
        self.assertFalse(config["held_out_eval_opened"])
        self.assertGreaterEqual(config["approximate_tau_ctr"] - config["n11_centered_residual"], config["n11_minimum_threshold_margin"])


if __name__ == "__main__":
    unittest.main()
