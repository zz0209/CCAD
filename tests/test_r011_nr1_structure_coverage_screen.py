import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime" / "r009"))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy.sparse import csr_matrix

from run_r011_nr1_structure_coverage_screen import calibration_metrics


class CalibrationEvaluabilityTests(unittest.TestCase):
    def test_zero_source_process_is_not_evaluable(self):
        source = csr_matrix(np.zeros((4, 1), dtype=np.float64))
        target = csr_matrix(np.zeros((4, 1), dtype=np.float64))
        result = calibration_metrics(
            [0], source, np.asarray([1.0, 0.0]), 0.0, target,
            np.asarray([[1.0, 0.0]]), np.asarray([0.0]), 4, 1e-12, 1e-12,
        )
        self.assertFalse(result["calibration_evaluable"])
        self.assertEqual(result["source_firing_count"], 0)
        self.assertEqual(result["d_ctr"], 0.0)

    def test_nonconstant_source_process_is_evaluable(self):
        source = csr_matrix(np.asarray([[0.0], [1.0], [0.0], [1.0]]))
        target = csr_matrix(np.asarray([[0.0], [1.0], [0.0], [1.0]]))
        result = calibration_metrics(
            [0], source, np.asarray([1.0, 0.0]), 0.5, target,
            np.asarray([[1.0, 0.0]]), np.asarray([0.5]), 4, 1e-12, 1e-12,
        )
        self.assertTrue(result["calibration_evaluable"])
        self.assertEqual(result["source_firing_count"], 2)
        self.assertAlmostEqual(result["d_ctr"], 0.0)


if __name__ == "__main__":
    unittest.main()
