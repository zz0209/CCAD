import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from f4_component_interventions import choose_groups,group_coordinates


class ComponentTests(unittest.TestCase):
    def test_frozen_groups_and_budget(self):
        order=np.arange(100)[::-1];energy=np.arange(100.)
        a=choose_groups(order,energy,budget=16,pool_size=64,seed=42)
        self.assertEqual(a,choose_groups(order,energy,budget=16,pool_size=64,seed=42))
        self.assertEqual(len(set(a['random_atoms'])),16)
        self.assertTrue(set(a['random_atoms']).issubset(a['random_pool']))
        self.assertEqual(a['top_atoms'],order[:16].tolist())
        with self.assertRaises(ValueError):choose_groups(order,energy,budget=16,pool_size=8,seed=1)

    def test_signed_complement_reconstructs_not_abs(self):
        z=np.array([[1.,2.,3.,4.],[-2.,1.,-1.,2.]])
        beta=np.array([2.,-1.,3.,-4.]);c=group_coordinates(z,beta,[0,2],[1,3])
        np.testing.assert_allclose(c['top']+c['tail'],z@beta)
        np.testing.assert_allclose(c['tail'],z[:,[1,3]]@beta[[1,3]])
        np.testing.assert_allclose(c['random'],c['tail'])
        self.assertLess(c['tail'][0],0)


if __name__=='__main__':unittest.main()
