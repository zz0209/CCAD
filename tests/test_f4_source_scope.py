import sys
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_f4_atom_participation import participation
from summarize_f4_source_scope import selected
from run_f4_source_reference_causal import readout_atom_order, norm_match, prepare_readout_ablation, source_hash_queries


class SourceScopeTests(unittest.TestCase):
    def test_source_hash_offset_is_disjoint_and_preserves_default(self):
        panel={(s,a):dict(energy_stratum=s,selection_hash=str(a)) for s in range(8) for a in range(3)}
        available=sorted(panel)
        self.assertEqual(source_hash_queries(available,panel),[(s,0) for s in range(8)])
        self.assertEqual(source_hash_queries(available,panel,1),[(s,1) for s in range(8)])
        self.assertFalse(set(source_hash_queries(available,panel)) & set(source_hash_queries(available,panel,1)))
        with self.assertRaises(ValueError):source_hash_queries(available,panel,3)

    def test_saved_readout_reuses_support_and_rejects_changed_coefficients(self):
        beta=np.arange(20,dtype=float);signs=np.random.default_rng(7).choice([-1.,1.],size=20)
        record=dict(source_seed=2,source_atom=3,target_seed=1,top_atoms=list(range(19,-1,-1)),
                    sign_seed=7,beta_sha256=hashlib.sha256(beta.tobytes()).hexdigest(),
                    sign_sha256=hashlib.sha256(signs.tobytes()).hexdigest())
        payload=dict(fit_split='discovery',calibration_used_for_ranking=False,refitted=False,families=[record])
        cfg=dict(ranks=[1],donor_difference=True,num_latents=20,
                 readout_ablation=dict(native_budget=16,budgets=[16],saved_readout={'path':'fixture'}))
        factors={'query_target':beta.reshape(1,20,1)};dec={1:np.eye(20)}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'saved.json';path.write_text(json.dumps(payload))
            result=prepare_readout_ablation(cfg,factors,{(2,3,1):0},dec,Path(temp),{'saved_readout':path})
            np.testing.assert_array_equal(result[2,3,1]['order'],record['top_atoms'])
            np.testing.assert_array_equal(result[2,3,1]['beta'],beta)
            factors['query_target'][0,0,0]=1
            with self.assertRaisesRegex(ValueError,'coefficients changed'):
                prepare_readout_ablation(cfg,factors,{(2,3,1):0},dec,Path(temp),{'saved_readout':path})

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
