import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_f4_atom_participation import participation
from summarize_f4_source_scope import selected
from run_f4_source_reference_causal import readout_atom_order, norm_match


class SourceScopeTests(unittest.TestCase):
    def test_readout_energy_ranking_and_rescaling(self):
        x=np.array([[1.,2.,3.],[3.,2.,1.],[2.,2.,2.]])
        beta=np.array([2.,100.,-1.]);weights=np.array([.2,.3,.5])
        order,energy=readout_atom_order(x,beta,weights)
        self.assertEqual(order.tolist(),[0,2,1])
        order2,energy2=readout_atom_order(x/np.array([2.,3.,4.]),beta*np.array([2.,3.,4.]),weights)
        np.testing.assert_array_equal(order,order2)
        np.testing.assert_allclose(energy,energy2)
        np.testing.assert_allclose(x[:,order]@beta[order],x@beta)

    def test_norm_matching_preserves_direction_and_zero(self):
        delta=np.array([[3.,4.],[-3.,4.]])
        matched=norm_match(delta,delta*2)
        np.testing.assert_allclose(matched,delta*2)
        np.testing.assert_array_equal(norm_match(np.zeros_like(delta),delta),np.zeros_like(delta))

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
