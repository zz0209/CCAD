import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_f4_agreement_relations import native_fit


class NativeFitTests(unittest.TestCase):
    def test_hadamard_gram_matches_explicit_vector_design(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(12,7));d=rng.normal(size=(7,3));b=np.array([1.,0.,0.]);c=rng.normal(size=12)*.02;ids=np.array([1,3,5])
        g,info=native_fit(x,d,c,b,ids,.001)
        design=np.stack([(x[:,j,None]*d[j]).reshape(-1) for j in ids],axis=1)
        y=(c[:,None]*b).reshape(-1)
        expected=np.linalg.solve(design.T@design/len(x)+info['ridge']*np.eye(3),design.T@y/len(x))
        np.testing.assert_allclose(g,expected,rtol=1e-10,atol=1e-12)
        self.assertEqual(info['clipped'],0)


if __name__=='__main__':unittest.main()
