import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_r006b3a_family_aggregation_simulation.py"
SPEC = importlib.util.spec_from_file_location("r006b3a_family_sim", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FamilyAggregationSimulationTests(unittest.TestCase):
    def test_wilson_interval_contains_empirical_rate(self) -> None:
        low, high = MODULE.wilson_interval(4000, 5000)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)

    def test_one_se_choice_prefers_smallest_eligible(self) -> None:
        means = np.asarray([0.0, 0.01, 0.02, 0.028, 0.03])
        self.assertEqual(MODULE.one_se_choice(means, 0.011), 2)

    def test_gate_fails_closed_on_undercoverage(self) -> None:
        metric = {"estimate": 0.95, "mc_ci_low": 0.94, "mc_ci_high": 0.96, "mc_half_width": 0.01}
        row = {"metrics": {name: dict(metric) for name in ("cluster_cover", "null_cluster_fp", "selector", "lofo")}}
        row["metrics"]["cluster_cover"]["mc_ci_low"] = 0.90
        thresholds = {"cluster_coverage_lower_mc_bound": 0.93, "null_false_positive_upper_mc_bound": 0.07,
                      "oracle_selector_agreement_lower_mc_bound": 0.8, "lofo_stability_lower_mc_bound": 0.8,
                      "maximum_mc_half_width": 0.015}
        self.assertFalse(MODULE.scenario_passes(row, thresholds))


if __name__ == "__main__":
    unittest.main()
