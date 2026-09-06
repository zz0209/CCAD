import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
from fit_f4_joint_sparse import standardized_inputs, fit_joint


class JointSparseTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(216)
        self.z=rng.normal(size=(128,20)); self.z[:,-1]=3.
        beta=np.zeros((20,2));beta[[1,4,8]]=[[2.,-1.],[-1.,2.],[.7,.4]]
        self.y=self.z@beta+[7.,-8.];self.w=rng.uniform(.1,1.,128)
        self.cfg=dict(pilot=False,minimum_alpha_fraction=.01,alpha_count=30,support_budget=4,
            path_stop_support=8,tol=1e-9,max_iter=5000,debias_ridge_fraction=.001,per_fit_budget_seconds=30)

    def test_weighted_normalization(self):
        x,y,w,mx,my,sx,sy,active=standardized_inputs(self.z,self.y,self.w)
        self.assertFalse(active[-1]);np.testing.assert_allclose(np.sum(x*x,axis=0)/len(x),1.,atol=1e-12)
        np.testing.assert_allclose(np.sum(y*y,axis=0)/len(y),1.,atol=1e-12)
        r=np.random.default_rng(9).normal(size=(active.sum(),2))
        beta=np.zeros((20,2));beta[active]=r*sy/sx[active,None]
        np.testing.assert_allclose(np.sum((y-x@r)**2)/len(y),np.sum(w[:,None]*((self.y-my-(self.z-mx)@beta)/sy)**2),rtol=1e-12)

    def test_shared_support_signed_donor_and_offset_invariance(self):
        beta,intercept,diag=fit_joint(self.z,self.y,self.w,self.cfg)
        self.assertLessEqual(np.count_nonzero(np.linalg.norm(beta,axis=1)),4)
        self.assertTrue(diag['selected']['converged'])
        np.testing.assert_allclose((self.z@beta+intercept)-(self.z[::-1]@beta+intercept),(self.z-self.z[::-1])@beta,atol=1e-12)
        changed,bias,_=fit_joint(self.z,self.y+[4.,-5.],self.w,self.cfg)
        np.testing.assert_allclose(changed,beta,atol=1e-10);np.testing.assert_allclose(bias,intercept+[4.,-5.],atol=1e-10)
        self.assertLess(diag['standardized_training_error'],.001)

    def test_zero_output_rejected(self):
        with self.assertRaises(ValueError):standardized_inputs(self.z,np.ones_like(self.y),self.w)


if __name__=='__main__':unittest.main()
