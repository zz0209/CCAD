import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from f4_probability_endpoints import log_prob, endpoint_positions, probability_metrics


class ProbabilityTests(unittest.TestCase):
    def test_logsumexp_stability(self):
        a=log_prob(np.array([[10000.,9999.,-10000.]]))
        self.assertAlmostEqual(float(np.exp(a).sum()),1.)
        np.testing.assert_allclose(a,log_prob(np.array([[0.,-1.,-20000.]])))

    def test_reference_and_zero_controls(self):
        b=np.zeros((3,3));s=np.array([[2.,0.,-2.]]*3);tokens=[1,2,1]
        same=probability_metrics(b,s,s,tokens,[0,1]);zero=probability_metrics(b,s,b,tokens,[0,1])
        self.assertEqual((same['normalized_kl_error'],same['normalized_nll_delta_squared_error']),(0.,0.))
        self.assertEqual((zero['normalized_kl_error'],zero['normalized_nll_delta_squared_error']),(1.,1.))
        tiny=probability_metrics(b,b,b,tokens,[0]);self.assertIsNone(tiny['normalized_kl_error'])
        self.assertIsNone(tiny['normalized_nll_delta_squared_error'])
        self.assertGreater(same['source_nll_delta_mean'],0)

    def test_masks_exclude_boundary_and_unavailable_gold(self):
        r=endpoint_positions([4,3,2,0,3,2,1,0,3,1],[1,5,9])
        self.assertEqual(r,dict(intervention_positions=[1,5],same_document_downstream=[1,5]))
        self.assertEqual(endpoint_positions([3,2,1,4,3],[1])['same_document_downstream'],[1,2,3])

    def test_direct_two_token_kl(self):
        b=np.log([[.5,.5],[.5,.5]]);s=np.log([[.8,.2],[.8,.2]]);c=np.log([[.6,.4],[.6,.4]])
        r=probability_metrics(b,s,c,[0,1],[0])
        expected=(.8*np.log(.8/.6)+.2*np.log(.2/.4))/(.8*np.log(.8/.5)+.2*np.log(.2/.5))
        self.assertAlmostEqual(r['normalized_kl_error'],expected)
        self.assertAlmostEqual(r['source_nll_delta_mean'],np.log(.5/.2))


if __name__=='__main__':unittest.main()
