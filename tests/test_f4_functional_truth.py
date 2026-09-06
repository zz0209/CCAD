import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_f4_functional_truth import encode,measure

class FunctionalTruthTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[.75,.2,.3,0,.1],[.75,.2,.3,1,.1],[0,0,0,1,0]])

    def test_split_reconstructs_and_is_nonnegative(self):
        z=encode(self.x,'context_split_preserved');d=np.column_stack([np.eye(5)[:,0],np.eye(5)])
        np.testing.assert_allclose(z@d.T,self.x,atol=1e-15)
        self.assertTrue(np.all(z>=0))

    def test_deleted_codes_invariant_to_source_function(self):
        changed=self.x.copy();changed[:,0]=.9
        np.testing.assert_array_equal(encode(changed,'source_function_deleted'),encode(self.x,'source_function_deleted'))

    def test_alias_changes_only_context_one_source_channel(self):
        z=encode(self.x,'context_alias')
        np.testing.assert_allclose(z[:,:2].sum(1),[.75,.1,0])
        np.testing.assert_array_equal(z[:,2:],self.x[:,1:])

    def test_actual_consumers_share_source_energy_and_zero_oracle(self):
        cfg=json.loads((ROOT/'configs/f4_functional_truth_v2.json').read_text())
        cfg.update(seeds=[1101],mean_samples=32,discovery_samples=32,pairs_per_context=32)
        rows,arrays,details=measure(cfg)
        self.assertEqual(len(rows),36)
        for r in rows:
            if r['method']=='raw_source_hook_oracle':
                self.assertEqual(r['normalized_coordinate_error'],0)
                self.assertEqual(r['normalized_kl_error'],0)
        for scope in ['all','context0','context1']:
            self.assertEqual(len({r['source_kl_sum'] for r in rows if r['scope']==scope}),1)
            self.assertEqual(len({r['source_coordinate_energy'] for r in rows if r['scope']==scope}),1)
        self.assertTrue(all(d['mcc']==1 for d in details))

if __name__=='__main__':unittest.main()
