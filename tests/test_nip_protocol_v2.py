from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from ccad.nip_synthetic_v2 import CAP_PRESSURE, DECOY_ORTHOGONAL_ENERGY, REGISTERED_CAPS, TARGET_ATOM_COUNT


ROOT = Path(__file__).resolve().parents[1]


class NIPProtocolV2Tests(unittest.TestCase):
    def test_frozen_config_binds_document_and_implementation_constants(self):
        config = json.loads((ROOT / "configs/m1_nip_protocol_v2.json").read_text(encoding="utf-8"))
        document = ROOT / config["protocol_document"]
        self.assertEqual(hashlib.sha256(document.read_bytes()).hexdigest().upper(), config["protocol_sha256"])
        self.assertEqual(config["target_atom_count"], TARGET_ATOM_COUNT)
        self.assertEqual(config["decoy_orthogonal_energy"], DECOY_ORTHOGONAL_ENERGY)
        self.assertEqual(tuple(config["proposal_atom_cap_grid_d1"]), REGISTERED_CAPS)
        self.assertEqual(config["positive_family_first_sufficient_cap"], CAP_PRESSURE)
        self.assertFalse(config["runtime_used_for_selection"])

    def test_no_v2_phase_seed_exists_at_protocol_freeze(self):
        config = json.loads((ROOT / "configs/m1_nip_protocol_v2.json").read_text(encoding="utf-8"))
        self.assertFalse(config["execution_enabled"])
        self.assertFalse(config["synthetic_evaluation_opened"])
        self.assertFalse(config["real_sae_audit_opened"])
        self.assertTrue(all(phase["seed_manifest_status"] == "UNGENERATED" for phase in config["phases"].values()))

    def test_selection_order_is_deterministic_and_runtime_free(self):
        config = json.loads((ROOT / "configs/m1_nip_protocol_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(config["selection_order"][-1], "MIN_ATOM_CAP")
        self.assertFalse(any("RUNTIME" in item for item in config["selection_order"]))
        self.assertEqual(config["certified_absence_lane"], "FULL_EXHAUSTIVE_ONLY")


if __name__ == "__main__":
    unittest.main()
