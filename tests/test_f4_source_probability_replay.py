import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_f4_source_probability import probability_order, source_anchor, observed_next_id


class ProbabilityMassTests(unittest.TestCase):
    def test_observed_token_boundaries(self):
        self.assertIsNone(observed_next_id([1,2],1))
        self.assertIsNone(observed_next_id([1,0,2],0))
        self.assertIsNone(observed_next_id([1,0,2],1))
        self.assertEqual(observed_next_id([1,2],0),2)

    def test_source_anchor_excludes_candidate_kl(self):
        p=dict(source_nll_deltas=[.1],source_to_baseline_kl=[.2],positions=[0],observed_next_token_ids=[2],source_to_candidate_kl=[.3])
        q=dict(p,source_to_candidate_kl=[.9])
        self.assertEqual(source_anchor(p),source_anchor(q))

    def test_mass_not_extreme_logits(self):
        b=np.array([[0.,0.,-30.]]); s=np.array([[1.,0.,-20.]])
        pb,ps,order=probability_order(b,s,1)
        self.assertEqual(order[0]['increase'].tolist(),[0])
        self.assertEqual(order[0]['decrease'].tolist(),[1])
        self.assertAlmostEqual(float(ps.sum()),1.)
        self.assertGreater(s[0,2]-b[0,2],s[0,0]-b[0,0])

    def test_shift_and_ties(self):
        b=np.zeros((1,4)); pb,ps,order=probability_order(b,b+20,2)
        np.testing.assert_allclose(pb,ps)
        self.assertEqual(order[0]['increase'].tolist(),[0,1])
        self.assertEqual(order[0]['base'].tolist(),[0,1])


if __name__=='__main__': unittest.main()
