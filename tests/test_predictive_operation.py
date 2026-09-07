import tempfile
from pathlib import Path
import unittest

import numpy as np

from ccad.predictive_operation import PredictiveOperation


class PredictiveOperationTest(unittest.TestCase):
    def test_signed_components_and_reciprocal_donor(self):
        operation = PredictiveOperation(np.array([3, 1]), np.array([[1., -2.], [-3., 4.]]), 5, {})
        delta = np.array([0., 2., 0., -1., 0.])
        np.testing.assert_array_equal(operation.predict(delta), [-7., 10.])
        np.testing.assert_array_equal(operation.component_deltas(delta[[3, 1]]), [[-1., 2.], [-6., 8.]])
        np.testing.assert_array_equal(operation.predict(-delta), [7., -10.])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'operation.npz'; operation.save(path)
            np.testing.assert_array_equal(PredictiveOperation.load(path).predict(delta), [-7., 10.])
            with self.assertRaises(FileExistsError): operation.save(path)

    def test_wrong_donor_changes_joint_unless_maps_agree_on_difference(self):
        pn = np.array([[1., 0.], [0., 0.5]])
        pt = np.array([[0.2, 0.], [0., 2.]])
        dn = np.array([1., 0.]); dt = np.array([0., 1.])
        correct = dn @ pn + dt @ pt
        wrong = dt @ pn + dn @ pt
        np.testing.assert_array_equal(correct, [1., 2.])
        np.testing.assert_array_equal(wrong, [0.2, 0.5])
        np.testing.assert_allclose(wrong - correct, (dt - dn) @ (pn - pt))
        np.testing.assert_array_equal(dt @ pn + dn @ pn, dn @ pn + dt @ pn)

    def test_invalid_member_order_and_code_shape_fail(self):
        with self.assertRaises(ValueError):
            PredictiveOperation(np.array([1, 1]), np.ones((2, 3)), 4, {})
        operation = PredictiveOperation(np.array([1]), np.ones((1, 3)), 4, {})
        with self.assertRaises(ValueError): operation.predict(np.ones(3))
        with self.assertRaises(ValueError): operation.predict_selected([np.nan])


if __name__ == '__main__': unittest.main()
