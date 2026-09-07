import itertools,unittest
import numpy as np
from ccad.component_correspondence import component_metric,metric_factor,component_update
from ccad.pair_complete_correspondence import pair_ridge


class ComponentCorrespondenceTests(unittest.TestCase):
    def test_exact_mask_expectation(self):
        rng=np.random.default_rng(9);d=rng.normal(size=(4,7));error=rng.normal(size=(9,4))
        exact=np.mean([np.sum(component_update(error,d,np.array(mask))**2) for mask in itertools.product([0,1],repeat=4)])
        m=component_metric(d,'half');l=metric_factor(m)
        self.assertAlmostEqual(exact,float(np.sum((error@l)**2)),places=10)
        self.assertAlmostEqual(float(np.sum((error@metric_factor(component_metric(d,'singleton')))**2)),float(np.mean([np.sum(component_update(error,d,np.eye(4)[i])**2) for i in range(4)])),places=10)

    def test_aggregate_does_not_identify_components(self):
        d=np.array([[1.,0.],[1.,0.]]);z=np.array([[1.,0.],[0.,1.]])
        np.testing.assert_array_equal(z@d,np.ones((2,1))@np.array([[1.,0.]]))
        self.assertFalse(np.array_equal(component_update(z[:1],d,[1,0]),component_update(z[1:],d,[1,0])))
        self.assertGreater(np.linalg.eigvalsh(component_metric(d,'half'))[0],0)

    def test_fixed_support_metric_ridge_equivariance(self):
        # A changed output metric with the matched transformed ridge penalty
        # cannot change a fixed-support, fixed-alpha coefficient solution.
        rng=np.random.default_rng(7);x=rng.normal(size=(12,5));z=rng.normal(size=(12,3));d=rng.normal(size=(3,7));pairs=np.arange(12)^1
        direct,_=pair_ridge(x,z,pairs,.01,.3)
        for family in ['whole','half','singleton']:
            l=metric_factor(component_metric(d,family));fit,_=pair_ridge(x,z@l,pairs,.01,.3)
            np.testing.assert_allclose(np.linalg.solve(l.T,fit.T).T,direct,atol=1e-12,rtol=1e-12)


if __name__=='__main__':unittest.main()
