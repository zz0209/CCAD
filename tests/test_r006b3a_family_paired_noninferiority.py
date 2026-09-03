import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_r006b3a_family_paired_noninferiority_simulation.py"
SPEC = importlib.util.spec_from_file_location("r006b3a_ni", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FamilyPairedNoninferiorityTests(unittest.TestCase):
    def test_student_t_quantiles_match_references(self) -> None:
        self.assertAlmostEqual(MODULE.student_t_quantile(0.975, 5), 2.570582, places=5)
        self.assertAlmostEqual(MODULE.student_t_quantile(0.95, 10), 1.812461, places=5)

    def test_selector_uses_smallest_eligible_or_reference(self) -> None:
        ucb = np.asarray([[0.03, 0.009, 0.004, 0.002], [0.03, 0.02, 0.011, 0.0101]])
        np.testing.assert_array_equal(MODULE.select_from_ucb(ucb, 0.01), np.asarray([1, 4]))

    def test_gate_fails_closed_on_boundary_error(self) -> None:
        good = {"estimate": 0.9, "mc_ci_low": 0.85, "mc_ci_high": 0.95, "mc_half_width": 0.01}
        metrics = {name: dict(good) for name in ("false_noninferiority", "boundary_false_noninferiority", "correct_smallest_safe", "lofo_stability")}
        metrics["false_noninferiority"].update({"estimate": 0.02, "mc_ci_low": 0.01, "mc_ci_high": 0.03})
        metrics["boundary_false_noninferiority"].update({"estimate": 0.03, "mc_ci_low": 0.02, "mc_ci_high": 0.04})
        row = {"metrics": metrics}
        thresholds = {"false_noninferiority_upper_mc_bound": 0.07, "boundary_false_noninferiority_upper_mc_bound": 0.025,
                      "correct_smallest_safe_lower_mc_bound": 0.8, "lofo_stability_lower_mc_bound": 0.8,
                      "maximum_mc_half_width": 0.015}
        self.assertFalse(MODULE.cell_passes(row, thresholds))

    def test_small_simulation_executes_all_diagnostics(self) -> None:
        row = MODULE.simulate_cell(
            np.random.default_rng(1), n_sims=10, families=6, tasks=4, icc=0.5,
            losses=np.asarray([0.03, 0.02, 0.005, 0.001, 0.0]), margin=0.01,
            total_sd=0.04, cross_corr=0.5, familywise_alpha=0.05,
            lofo_required_fraction=0.8,
        )
        self.assertEqual(
            set(row["metrics"]),
            {"false_noninferiority", "boundary_false_noninferiority",
             "correct_smallest_safe", "lofo_stability", "argmax_correct", "one_se_correct"},
        )


if __name__ == "__main__":
    unittest.main()
