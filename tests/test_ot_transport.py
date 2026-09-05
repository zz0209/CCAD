import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ccad.ot_transport import signed_ot_readout, discovery_document_partition, weighted_difference_error


class OTReadoutTests(unittest.TestCase):
    def test_document_partition_excludes_packed_contexts(self):
        seq=[dict(sequence_index=i,split='discovery',document_ids=[str(i//2)]) for i in range(40)]
        seq.append(dict(sequence_index=40,split='discovery',document_ids=['0','1']))
        rows=list(range(0,41*128,128))
        part=discovery_document_partition(rows,seq,128,salt='test')
        self.assertEqual(part['excluded_multidoc_indices'],[40])
        self.assertFalse(set(part['train_documents']) & set(part['validation_documents']))
        self.assertTrue(part['train_indices'] and part['validation_indices'])
        for i in range(0,40,2):
            self.assertEqual(i in part['train_indices'],i+1 in part['train_indices'])

    def test_validation_difference_error_matches_all_pairs(self):
        s=np.array([1.,4.,8.]);p=np.array([-1.,2.,5.]);w=np.array([1.,3.,2.])
        ds=s[:,None]-s;dp=p[:,None]-p;ww=w[:,None]*w
        expected=np.sum(ww*(ds-dp)**2)/np.sum(ww*ds**2)
        self.assertAlmostEqual(weighted_difference_error(s,p+100,w)['normalized_error'],expected)

    def fixture(self):
        t=np.arange(128)*2*np.pi/128
        source=np.column_stack([np.sin(t),np.cos(t)])
        target=np.column_stack([-3*source[:,1],2*source[:,0]])
        return source,target,np.array([0.8,-0.3]),np.ones(len(t))

    def test_signed_permutation_and_gain(self):
        xs,xt,c,w=self.fixture()
        beta,plan,diag=signed_ot_readout(xs,xt,c,w,regularization=0.02)
        expected=np.array([0.1,0.4])/1.001
        np.testing.assert_allclose(beta,expected,atol=1e-10)
        self.assertEqual(diag['status'],'OK')
        self.assertGreater(diag['signed_lift_negative_entries'],0)
        self.assertTrue(np.all(plan>=0))

    def test_scale_gauge_and_difference_intercept(self):
        xs,xt,c,w=self.fixture()
        beta,plan,_=signed_ot_readout(xs,xt,c,w)
        ss=np.array([2.,5.]);st=np.array([4.,0.5])
        other,other_plan,_=signed_ot_readout(xs*ss+7,xt*st-2,c/ss,w)
        np.testing.assert_allclose(plan,other_plan,atol=1e-12)
        np.testing.assert_allclose(beta,other*st,atol=1e-12)
        np.testing.assert_allclose((xt[1]-xt[0])@beta,((xt[1]*st-2)-(xt[0]*st-2))@other,atol=1e-12)

    def test_zero_variance_is_explicit_and_invalid_weights_refused(self):
        xs,xt,c,w=self.fixture()
        beta,plan,diag=signed_ot_readout(np.full_like(xs,1.1),xt,c,np.arange(len(w))+0.3)
        self.assertEqual(diag['status'],'NO_CONDITIONAL_VARIATION')
        self.assertEqual(plan.shape[0],0)
        self.assertTrue(np.all(beta==0))
        with self.assertRaises(ValueError):
            signed_ot_readout(xs,xt,c,-w)


if __name__=='__main__':
    unittest.main()
