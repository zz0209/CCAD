import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_f4_atom_participation import participation
from summarize_f4_source_scope import selected


class SourceScopeTests(unittest.TestCase):
    def test_contribution_energy_and_scale_invariance(self):
        z=np.array([[1.,1.],[2.,2.]])
        a=participation(z,np.zeros(2),np.ones((2,1)))
        b=participation(z*.1,np.zeros(2),np.ones((2,1)))
        self.assertEqual(a['effective_energy_atoms'],2)
        self.assertEqual(a['largest_atom_energy_share'],.5)
        self.assertAlmostEqual(a['sum_atom_energy_over_aggregate'],.5)
        self.assertAlmostEqual(b['aggregate_energy']/a['aggregate_energy'],.01)
        self.assertAlmostEqual(a['largest_atom_energy_share'],b['largest_atom_energy_share'])

    def test_rule_boundaries_and_unsupported(self):
        rule={'maximum_largest_source_atom_energy_share':.5,'minimum_natural_source_hook_fraction':.1}
        row={'supported':True,'largest_atom_energy_share':.5,'natural_source_hook_fraction':.1}
        self.assertTrue(selected(row,rule))
        for change in [{'supported':False},{'largest_atom_energy_share':.50001},
                       {'natural_source_hook_fraction':.09999},{'largest_atom_energy_share':None}]:
            self.assertFalse(selected(row|change,rule))


if __name__=='__main__':unittest.main()
