from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.run_r001_smoke import evaluate, metric_surface_errors


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIGS = {
    "r001_candidate_v1.json": {"F01_hadamard_gauge", "F02_local_block_rotations"},
    "r001_complements_v1.json": {"F10_same_span_different_computation", "F11_same_sum_bloated_span"},
    "r002_f02_formal_v1.json": {"F02_local_block_rotations"},
    "r002_f03_formal_v2.json": {"F03_unequal_split_merge"},
    "r002_f04_formal_v1.json": {"F04_partial_overlap"},
    "r002_f06_formal_v1.json": {"F06_cooccurrence_confounding"},
    "r003_f05_formal_v1.json": {"F05_cancellation"},
    "r003_f07_formal_v1.json": {"F07_rare_occupancy"},
    "r003_f08_f09_formal_v1.json": {"F08_competing_covers", "F09_whole_dictionary_only"},
    "r003_f10_f11_formal_v1.json": {"F10_same_span_different_computation", "F11_same_sum_bloated_span"},
    "r003_f12_formal_v1.json": {"F12_non_lipschitz_downstream_cliff"},
}


class FormalRunnerIntegrationTests(unittest.TestCase):
    def test_every_historical_formal_config_executes_through_public_evaluator(self) -> None:
        for filename, expected_families in FORMAL_CONFIGS.items():
            with self.subTest(config=filename):
                config = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))
                records, summary = evaluate(config)
                self.assertEqual(summary["status"], "PASS")
                self.assertEqual(set(summary["families_covered"]), expected_families)
                self.assertEqual(len(records), summary["records"])
                self.assertTrue(all(record is not None for record in records))

    def test_every_formal_family_emits_complete_metric_surface(self) -> None:
        family_records = {}
        for filename in FORMAL_CONFIGS:
            config = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))
            if filename == "r001_complements_v1.json":
                continue
            config["seed_pair_count"] = 1
            config["emit_complete_metric_surface"] = True
            records, summary = evaluate(config)
            self.assertEqual(summary["status"], "PASS", filename)
            self.assertEqual(summary["metric_surface_error_count"], 0, filename)
            for record in records:
                family_records[record["family_id"]] = record
                self.assertEqual(metric_surface_errors(record), [], record["family_id"])
                self.assertGreater(len(record["metric_surface"]["group_measurements"]), 0)
        self.assertEqual(set(family_records), {f"F{i:02d}_" + suffix for i, suffix in [
            (1, "hadamard_gauge"), (2, "local_block_rotations"), (3, "unequal_split_merge"),
            (4, "partial_overlap"), (5, "cancellation"), (6, "cooccurrence_confounding"),
            (7, "rare_occupancy"), (8, "competing_covers"), (9, "whole_dictionary_only"),
            (10, "same_span_different_computation"), (11, "same_sum_bloated_span"),
            (12, "non_lipschitz_downstream_cliff"),
        ]})


if __name__ == "__main__":
    unittest.main()
