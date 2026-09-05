import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_f4_source_reference_causal import fixed_support_ridge, source_contrast_pairs


class FixedSupportTests(unittest.TestCase):
    def test_matches_weighted_normal_equation(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(32,16));y=rng.normal(size=32)
        w=rng.uniform(.1,1,size=32);w/=w.sum()
        beta,info=fixed_support_ridge(x,y,w,.001)
        xc=x-w@x;yc=y-w@y;gram=xc.T@(w[:,None]*xc)
        ridge=.001*np.trace(gram)/16
        expected=np.linalg.solve(gram+ridge*np.eye(16),xc.T@(w*yc))
        np.testing.assert_allclose(beta,expected,rtol=1e-10,atol=1e-12)
        self.assertAlmostEqual(info['ridge'],ridge,places=12)

    def test_intercepts_cancel_and_pairs_agree(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(12,5));y=rng.normal(size=12)
        w=rng.uniform(.1,1,size=12);w/=w.sum()
        beta,_=fixed_support_ridge(x,y,w,.001)
        shifted,_=fixed_support_ridge(x+100,y-50,w,.001)
        dx=(x[:,None]-x[None,:]).reshape(-1,5);dy=(y[:,None]-y[None,:]).ravel()
        pairs,_=fixed_support_ridge(dx,dy,(w[:,None]*w[None,:]).ravel(),.001)
        np.testing.assert_allclose(beta,shifted,atol=1e-10)
        np.testing.assert_allclose(beta,pairs,atol=1e-10)

    def test_constant_support_and_invalid_weights(self):
        beta,info=fixed_support_ridge(np.ones((20,16)),np.arange(20.),np.ones(20),.001)
        np.testing.assert_array_equal(beta,np.zeros(16))
        self.assertEqual(info['effective_rank'],0)
        with self.assertRaises(ValueError):
            fixed_support_ridge(np.ones((20,16)),np.arange(20.),np.zeros(20),.001)

    def test_difference_fit_keeps_between_condition_mean(self):
        x=np.full((128,16),2.);y=np.full(128,5.);w=np.ones(128)/128
        beta,info=fixed_support_ridge(x,y,w,.001,center=False)
        gram=x.T@(w[:,None]*x);ridge=.001*np.trace(gram)/16
        expected=np.linalg.solve(gram+ridge*np.eye(16),x.T@(w*y))
        np.testing.assert_allclose(beta,expected,atol=1e-9)
        self.assertEqual(info['intercept'],0.)
        self.assertLess(info['weighted_error'],1e-6)
        centered,_=fixed_support_ridge(x,y,w,.001)
        np.testing.assert_array_equal(centered,np.zeros(16))

    def test_source_pairs_respect_row_budget_and_context(self):
        scores=np.arange(20,dtype=float);coord=np.arange(20,dtype=float)
        rr,dd=source_contrast_pairs([0,1,2,3],scores,coord,4,max_rows=8)
        np.testing.assert_array_equal(rr,[0,1,2,3])
        np.testing.assert_array_equal(dd,[19,18,17,16])
        self.assertEqual(len(set(rr.tolist()+dd.tolist())),8)
        self.assertTrue(np.all(rr//4!=dd//4))
        with self.assertRaises(ValueError):source_contrast_pairs([0,1],np.zeros(4),np.zeros(4),2)

    def test_norm_constrained_fit_kkt_and_inactive_equivalence(self):
        rng=np.random.default_rng(22);x=rng.normal(size=(24,6));y=100*x[:,0]+2;w=np.ones(24)/24
        beta,info=fixed_support_ridge(x,y,w,.001,center=False,norm_bound=.7)
        self.assertTrue(info['norm_constraint_active'])
        self.assertAlmostEqual(np.linalg.norm(beta),.7,places=10)
        np.testing.assert_allclose((x.T@(w[:,None]*x)+info['ridge']*np.eye(6))@beta,x.T@(w*y),atol=1e-9)
        plain,_=fixed_support_ridge(x,y,w,.001,center=False)
        inactive,_=fixed_support_ridge(x,y,w,.001,center=False,norm_bound=1000)
        np.testing.assert_array_equal(plain,inactive)
        zero,_=fixed_support_ridge(x,y,w,.001,center=False,norm_bound=0)
        np.testing.assert_array_equal(zero,np.zeros(6))


if __name__=='__main__':unittest.main()
