import unittest
import numpy as np
from ccad.source_state_projection import project_source_state
from ccad.source_state_explanation import contextual_allocation_terms


class SourceStateExplanationTests(unittest.TestCase):
    def test_contextual_terms_sum_to_actual_projection_and_masked_edit(self):
        d=np.array([[1.,0.],[.8,.6]]);h=np.array([[-1.,2.],[2.,-1.]])
        operators=[]
        for x in [np.array([2.,.1]),np.array([.1,2.])]:
            v=x@h;u,m=project_source_state(v,d);terms,b=contextual_allocation_terms(x,h,u,m)
            np.testing.assert_allclose(terms.sum(0),u,atol=1e-12)
            np.testing.assert_allclose(b@v,u,atol=1e-12)
            mask=np.array([1.,0.]);np.testing.assert_allclose(((terms*mask)@d).sum(0),(u*mask)@d,atol=1e-12)
            operators.append(b)
        self.assertGreater(np.max(np.abs(operators[0]-operators[1])),.5)

    def test_fixed_source_unit_rescaling_preserves_physical_masked_values(self):
        d=np.array([[1.,0.,.1],[.8,.6,0.]])
        v=np.array([-1.,2.]);scale=np.array([.1,7.]);mask=np.array([1.,.25])
        u,m=project_source_state(v,d)
        rescaled,ms=project_source_state(v*scale,d/scale[:,None])
        np.testing.assert_allclose(rescaled,u*scale,atol=1e-11)
        np.testing.assert_allclose((rescaled*mask)@(d/scale[:,None]),(u*mask)@d,atol=1e-11)
        np.testing.assert_allclose(ms,m/scale[:,None]/scale[None,:],atol=1e-11)

if __name__=='__main__':unittest.main()
