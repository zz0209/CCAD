import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_f4_source_reference_causal import fixed_support_ridge


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


if __name__=='__main__':unittest.main()
