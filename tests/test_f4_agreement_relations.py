import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_f4_agreement_relations import native_fit,neighborhood_pairs,native_vector_fit,fit_basis_constrained_transport


class NativeFitTests(unittest.TestCase):
    def test_preserved_basis_keeps_later_active_coordinate(self):
        x=np.arange(1,5,dtype=float)[:,None];y=np.column_stack([np.zeros(4),x[:,0]])
        fit=fit_basis_constrained_transport(x,y,np.eye(2),np.ones(4)/4,preserve_basis=True)
        self.assertEqual(fit.effective_rank,1);self.assertEqual(fit.target_factors.shape,(1,2))
        self.assertGreater(float(fit.target_factors[0,1]),.99)

    def test_vector_teacher_matches_explicit_design(self):
        rng=np.random.default_rng(81);x=rng.normal(size=(18,5));d=rng.normal(size=(5,3));teacher=rng.normal(size=(18,3))*.01;ids=np.array([0,2,4])
        g,info=native_vector_fit(x,d,teacher,ids,.001)
        a=np.stack([(x[:,j,None]*d[j]).ravel() for j in ids],axis=1)
        expected=np.linalg.solve(a.T@a/18+info['ridge']*np.eye(3),a.T@teacher.ravel()/18)
        np.testing.assert_allclose(g,expected,rtol=1e-5,atol=1e-8)

    def test_neighborhood_unique_paired_deterministic(self):
        candidates=np.array([[1.,0.],[0.,1.],[2.,0.],[0.,2.]])
        selected,prototypes,scores=neighborhood_pairs(candidates,np.eye(2),[1,0],4)
        np.testing.assert_array_equal(selected,[0,1,2,3])
        np.testing.assert_array_equal(prototypes,[0,1,0,1])
        np.testing.assert_allclose(scores,1.)

    def test_hadamard_gram_matches_explicit_vector_design(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(12,7));d=rng.normal(size=(7,3));b=np.array([1.,0.,0.]);c=rng.normal(size=12)*.02;ids=np.array([1,3,5])
        g,info=native_fit(x,d,c,b,ids,.001)
        design=np.stack([(x[:,j,None]*d[j]).reshape(-1) for j in ids],axis=1)
        y=(c[:,None]*b).reshape(-1)
        expected=np.linalg.solve(design.T@design/len(x)+info['ridge']*np.eye(3),design.T@y/len(x))
        np.testing.assert_allclose(g,expected,rtol=1e-10,atol=1e-12)
        self.assertEqual(info['clipped'],0)


if __name__=='__main__':unittest.main()
