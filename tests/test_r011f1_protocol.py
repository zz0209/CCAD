from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class R011F1ProtocolTests(unittest.TestCase):
    def test_preaudit_protocol_is_fail_closed_and_binds_existing_inputs(self) -> None:
        cfg = json.loads((ROOT / "configs/r011_f1_preaudit_protocol_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(cfg["execution_enabled"])
        self.assertFalse(cfg["audit_opened"])
        self.assertEqual(cfg["forbidden_splits"], ["audit"])
        self.assertEqual(cfg["candidate_ranks"], [1, 2, 4, 8])
        self.assertEqual(cfg["source_candidate_count"] * cfg["target_candidate_cap"], cfg["feature_pair_budget"])
        self.assertEqual(cfg["primary_independent_causal_endpoint"], "next_state_residual")
        for path_key, hash_key in (
            ("query_panel_path", "query_panel_sha256"),
            ("source_census_path", "source_census_sha256"),
        ):
            path = ROOT / cfg[path_key]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), cfg[hash_key])
        self.assertTrue((ROOT / cfg["r009c_candidate_path"]).is_file())
        self.assertTrue((ROOT / cfg["protocol_document"]).is_file())

    def test_synthetic_gate_config_has_all_required_families(self) -> None:
        cfg = json.loads((ROOT / "configs/r011_f1_sparse_synthetic_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["families"], [
            "rotation", "split_merge", "overlap", "downstream_null", "hard_negative", "competing_relation",
        ])
        self.assertFalse(cfg["audit_opened"])

    def test_c040_probe_asset_is_discovery_only_and_protocol_bound(self) -> None:
        cfg = json.loads((ROOT / "configs/r011f1_c040_probe_metric_v1.json").read_text(encoding="utf-8"))
        parent_path = ROOT / cfg["protocol_config_path"]
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(parent_path.read_bytes()).hexdigest(), cfg["protocol_config_sha256"])
        self.assertFalse(parent["execution_enabled"])
        self.assertEqual(cfg["split"], "discovery")
        self.assertEqual(cfg["forbidden_splits"], ["mean", "calibration", "audit"])
        for field in (
            "probe_states", "probe_directions_per_state", "probe_relative_amplitude",
            "output_logit_sketch_dim", "probe_ridge_fraction", "metric_eigenvalue_relative_tolerance",
        ):
            self.assertEqual(cfg[field], parent[field])

    def test_c040_v2_changes_only_execution_identity(self) -> None:
        base = json.loads((ROOT / "configs/r011f1_c040_probe_metric_v1.json").read_text(encoding="utf-8"))
        suffix = json.loads((ROOT / "configs/r011f1_c040_probe_metric_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(set(suffix), {"inherits_config", "overrides"})
        self.assertEqual(set(suffix["overrides"]), {"run_id", "correction_from", "correction_reason"})
        self.assertEqual(suffix["inherits_config"], "configs/r011f1_c040_probe_metric_v1.json")
        self.assertEqual(base["run_id"], suffix["overrides"]["correction_from"])


if __name__ == "__main__":
    unittest.main()
