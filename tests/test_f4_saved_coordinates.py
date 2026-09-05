import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_f4_source_reference_causal import validate_saved_coordinate, saved_coordinate_key


class SavedCoordinatesTests(unittest.TestCase):
    def test_unscaled_masked_coordinates(self):
        value=np.array([[0.],[3.],[0.]])
        np.testing.assert_array_equal(validate_saved_coordinate(value,{'intervention_positions':[1]},3,1),value)
        with self.assertRaises(ValueError):validate_saved_coordinate(value,{'intervention_positions':[0]},3,1)

    def test_shape_and_finiteness(self):
        with self.assertRaises(ValueError):validate_saved_coordinate(np.zeros((3,2)),{'intervention_positions':[1]},3,1)
        with self.assertRaises(ValueError):validate_saved_coordinate(np.full((3,1),np.nan),{'intervention_positions':[1]},3,1)

    def test_method_and_condition_keys_differ(self):
        entry={'condition':'positive','sequence':1}
        self.assertNotEqual(saved_coordinate_key(1,2,3,entry,'a'),saved_coordinate_key(1,2,3,entry,'b'))


if __name__=='__main__':unittest.main()
