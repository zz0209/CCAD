import unittest
import numpy as np
from ccad.source_state_projection import project_source_state,projection_diagnostics


class SourceStateProjectionTests(unittest.TestCase):
    def test_correlated_metric_optimum_and_state_bound(self):
        # Correlation makes coordinate clipping fail KKT: the free coordinate
        # must also change. Check optimality independently through KKT and the
        # projection inequality against many admissible source states.
        d=np.array([[1.,0.],[.8,.6]])
        v=np.array([[-1.,2.],[1.,-.5],[-1.,-2.],[2.,3.]])
        u,m=project_source_state(v,d)
        self.assertGreater(np.max(np.abs(u-np.maximum(v,0))),.1)
        np.testing.assert_array_equal(u[-1],v[-1])
        rng=np.random.default_rng(715)
        for _ in range(15):
            z=rng.uniform(0,3,v.shape);r=projection_diagnostics(v,u,z,m)
            self.assertLess(r['dual_violation_relative'],1e-12)
            self.assertLess(r['complementarity_relative'],1e-12)
            self.assertGreater(r['minimum_projection_inequality_slack'],-1e-12)

    def test_state_guarantee_does_not_imply_contrast_guarantee(self):
        d=np.ones((1,1));v=np.array([[-2.],[-1.]]);z=np.array([[0.],[1.]])
        u,m=project_source_state(v,d)
        self.assertLess(np.sum((u-z)**2),np.sum((v-z)**2))
        self.assertEqual(float(np.diff(v,axis=0)[0,0]),float(np.diff(z,axis=0)[0,0]))
        self.assertNotEqual(float(np.diff(u,axis=0)[0,0]),float(np.diff(z,axis=0)[0,0]))

    def test_deployment_projects_absolute_endpoints_before_signed_operation(self):
        from ccad.component_operation import ComponentOperation
        op=ComponentOperation([0],[0],[[-1.]],[[1.]],{'state_projection':'cone_half'})
        # Negative signed predictions of two admissible target states both
        # project to zero. Projecting their positive difference would be wrong.
        np.testing.assert_array_equal(op.apply([[2.]],consumer='contrast',donor_codes=[[1.]]),[[0.]])
        op=ComponentOperation([0],[0],[[1.]],[[1.]],{'state_projection':'cone_half'})
        np.testing.assert_array_equal(op.apply([[2.]],consumer='removal'),[[-2.]])
        np.testing.assert_array_equal(op.apply([[2.]],consumer='contrast',donor_codes=[[1.]]),[[-1.]])

if __name__=='__main__':unittest.main()
