import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_f4_long_recipients import restore_fits, apply_recipient_fit


class FrozenRecipientTests(unittest.TestCase):
    def record(self):
        return dict(configuration='long128',source_seed=3,source_atom=7,target_seed=1,
                    array_key='b',top_atoms=[2,0],single_atom=dict(atom=1,coefficient=-3.))

    def test_signed_donor_apply_without_refit_or_ranking(self):
        beta=np.array([2.,-1.,4.]);r=self.record()
        with patch('prepare_f4_long_recipients.fixed_support_ridge',side_effect=AssertionError('refit')), patch('prepare_f4_long_recipients.readout_atom_order',side_effect=AssertionError('rerank')):
            fit=restore_fits([r],{'b':beta},3,2)['long128',3,7,1]
            x=np.array([[1.,2.,3.],[4.,0.,1.]]);d=np.array([[3.,1.,0.],[1.,2.,5.]])
            out=apply_recipient_fit(x-d,fit)
        np.testing.assert_array_equal(fit['keep'],[2,0])
        np.testing.assert_allclose(out['target'],(x-d)@beta)
        np.testing.assert_allclose(out['top16'],(x-d)[:,2]*4+(x-d)[:,0]*2)
        np.testing.assert_allclose(out['single_atom'],(x-d)[:,1]*-3)
        zero=apply_recipient_fit(x-x,fit)
        self.assertTrue(all(np.array_equal(v,np.zeros(2)) for v in zero.values()))

    def test_bad_or_duplicate_fit_rejected(self):
        r=self.record()
        for rows,arr in (([r,r],np.ones(3)),([r],np.ones(4)),([r],np.array([np.nan,1,2]))):
            with self.assertRaises(ValueError):restore_fits(rows,{'b':arr},3,2)


if __name__=='__main__':unittest.main()
