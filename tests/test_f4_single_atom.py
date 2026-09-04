import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_f4_source_reference_causal import best_single_atom


class SingleAtomTests(unittest.TestCase):
    def test_matches_separate_weighted_scalar_ridge(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(32,7));y=rng.normal(size=32)
        w=rng.uniform(.1,1,size=32);w/=w.sum();fraction=.001
        fit=best_single_atom(x,y,w,fraction)
        results=[]
        for atom in range(x.shape[1]):
            xx=x[:,atom:atom+1]*np.sqrt(w)[:,None];yy=y*np.sqrt(w)
            ridge=fraction*float((xx.T@xx)[0,0])
            coef=np.linalg.solve(xx.T@xx+ridge*np.eye(1),xx.T@yy)[0]
            results.append((float(np.sum(w*(y-x[:,atom]*coef)**2)),atom,coef))
        expected=min(results)
        self.assertEqual(fit['atom'],expected[1])
        self.assertAlmostEqual(fit['coefficient'],expected[2],places=12)
        self.assertAlmostEqual(fit['weighted_error'],expected[0],places=12)

    def test_predictor_rescaling_and_zero_ties(self):
        x=np.array([[1.,2.,0.],[2.,-1.,0.],[3.,4.,0.]])
        y=x[:,0]*3;w=np.ones(3)/3
        fit=best_single_atom(x,y,w,.001)
        scaled=x*np.array([17.,.04,8.])
        other=best_single_atom(scaled,y,w,.001)
        self.assertEqual(fit['atom'],other['atom'])
        np.testing.assert_allclose(x[:,fit['atom']]*fit['coefficient'],scaled[:,other['atom']]*other['coefficient'])
        self.assertEqual(best_single_atom(np.zeros((3,4)),y,w,.001)['atom'],0)

    def test_conditional_variation_equals_all_weighted_pairs(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(12,5))+rng.normal(size=5)*10
        y=rng.normal(size=12)+40;w=rng.uniform(.1,1,size=12);w/=w.sum()
        fit=best_single_atom(x,y,w,.001,conditional_variation=True)
        differences=(x[:,None,:]-x[None,:,:]).reshape(-1,5)
        dy=(y[:,None]-y[None,:]).ravel();pairweights=(w[:,None]*w[None,:]).ravel()
        pairfit=best_single_atom(differences,dy,pairweights,.001)
        self.assertEqual(fit['atom'],pairfit['atom'])
        self.assertAlmostEqual(fit['coefficient'],pairfit['coefficient'],places=12)
        self.assertAlmostEqual(2*fit['weighted_error'],pairfit['weighted_error'],places=12)


if __name__=='__main__': unittest.main()
