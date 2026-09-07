import unittest
import numpy as np
from ccad.pair_complete_correspondence import pair_parts, pair_ridge


class PairCompleteTests(unittest.TestCase):
    def test_nonnegative_pair_invariant_counterexample(self):
        x=np.array([[0.,1.],[1.,1.],[0.,2.],[1.,2.]])
        donor=np.array([1,0,3,2]); truth=np.array([[1.],[5.]])
        y=x@truth
        difference_only,_=pair_ridge(x,y,donor,1e-10,0.)
        complete,_=pair_ridge(x,y,donor,1e-10,1.)
        self.assertLess(np.max(np.abs((x-x[donor])@difference_only-(y-y[donor]))),1e-8)
        self.assertGreater(np.mean((x@difference_only-y)**2),50.)
        self.assertLess(np.max(np.abs(x@complete-y)),1e-7)
        # A single mean correction leaves the pair-dependent constant unresolved.
        fixed=x@difference_only+np.mean(y-x@difference_only,axis=0)
        self.assertGreater(np.mean((fixed-y)**2),6.)

    def test_projector_energy_and_complete_ridge_identity(self):
        rng=np.random.default_rng(713);x=rng.normal(size=(12,4));y=rng.normal(size=(12,3))
        donor=np.arange(12)^1;minus,plus=pair_parts(y,donor)
        self.assertAlmostEqual(float(np.sum(minus*plus)),0.,places=12)
        self.assertAlmostEqual(float(np.sum(y*y)),float(np.sum(minus**2)+np.sum(plus**2)),places=12)
        w,diag=pair_ridge(x,y,donor,.03,1.)
        direct=np.linalg.solve(x.T@x+len(x)*.03*diag['unit']**2*np.eye(4),x.T@y)
        np.testing.assert_allclose(w,direct,rtol=1e-12,atol=1e-12)

    def test_reject_nonreciprocal_donor(self):
        with self.assertRaises(ValueError):pair_parts(np.ones((3,2)),np.array([1,2,0]))


if __name__=='__main__':unittest.main()
