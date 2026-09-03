from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.validate_m1_nip_parent_completion_v1 import (
    CRITICAL_METRICS,
    EXPECTED_CONTROLS,
    EXPECTED_NATIVE_LANES,
    EXPECTED_REFERENCES,
    EXPECTED_STREAMS,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


class ParentCompletionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/m1_nip_parent_completion_v1.json").read_text(encoding="utf-8"))
        cls.config_v2_path = ROOT / "configs/m1_nip_parent_completion_v2.json"

    def test_fresh_closed_namespace_and_independent_units(self):
        self.assertFalse(self.config["execution_enabled"])
        self.assertEqual(self.config["formal_seed_manifest_status"], "UNGENERATED")
        self.assertEqual(self.config["repeat_unit"], "structural_seed_pair")
        self.assertEqual(self.config["block"], "family")
        self.assertEqual(set(self.config["required_seed_streams"]), EXPECTED_STREAMS)
        self.assertTrue(self.config["require_pairwise_distinct_seed_streams"])

    def test_registered_comparisons_and_simplicity_rule_are_complete(self):
        self.assertEqual(set(self.config["registered_native_lanes"]), EXPECTED_NATIVE_LANES)
        self.assertEqual(set(self.config["registered_non_native_references"]), EXPECTED_REFERENCES)
        self.assertEqual(self.config["simplicity_rule"]["challenger"], "BINARY_FORWARD_OMP")
        self.assertEqual(self.config["simplicity_rule"]["action"], "REMOVE_MSCC_FROM_HEADLINE_KEEP_AS_ABLATION")

    def test_raw_metric_and_family_control_surface_is_complete(self):
        self.assertTrue(CRITICAL_METRICS <= set(self.config["mandatory_metric_fields"]))
        self.assertEqual(set(self.config["mandatory_family_controls"]), EXPECTED_CONTROLS)
        self.assertTrue(self.config["prelabel_validation_must_pass_before_truth_import"])
        self.assertTrue(self.config["complete_universe_requires_gmax_cover_target_count"])
        self.assertEqual(self.config["scalable_negative_identification"], "UNRESOLVED")

    def test_static_validator_rejects_unfrozen_baseline_parameters(self):
        result = validate(ROOT, ROOT / "configs/m1_nip_parent_completion_v1.json")
        self.assertFalse(result["checks"]["baseline_operationalization"])
        self.assertEqual(result["status"], "FAIL")

    def test_v2_static_contract_is_fully_operationalized(self):
        result = validate(ROOT, self.config_v2_path)
        # The prospective no-run check correctly turns false after P1 starts;
        # operationalization fields remain invariant and are tested separately.
        invariant_checks = {name: value for name, value in result["checks"].items() if name != "no_existing_completion_run"}
        self.assertTrue(all(invariant_checks.values()))
        self.assertTrue(result["checks"]["baseline_operationalization"])
        self.assertTrue(result["checks"]["baseline_parameters_exact"])
        self.assertTrue(result["checks"]["common_native_rule"])
        self.assertTrue(result["checks"]["source_registry"])


if __name__ == "__main__":
    unittest.main()
