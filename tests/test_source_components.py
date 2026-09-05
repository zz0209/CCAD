import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_f4_source_components import np,partition_source,additive_ridge,family_scale


class SourceComponentTests(unittest.TestCase):
    def test_partition_sum_and_source_id_ties(self):
        z=np.array([[0.,0.,0.,0.],[1.,1.,1.,1.],[2.,2.,2.,2.]])
        mean=np.ones(4)*.4;d=np.ones((4,2));b=np.array([1.,0.]);weights=np.ones(3)
        groups,y,_=partition_source(z,mean,d,b,[3,1,2,0],weights,2)
        self.assertEqual([g.tolist() for g in groups],[[0,1],[2,3]])
        np.testing.assert_allclose(y.sum(axis=1),(z-mean)@d@b)
        groups,yi,_=partition_source(z,mean,d,b,[3,1,2,0],weights,2,mode='interleaved')
        self.assertEqual([g.tolist() for g in groups],[[0,2],[1,3]])
        np.testing.assert_allclose(yi.sum(axis=1),y.sum(axis=1))

    def test_ridge_is_linear_in_component_outputs(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(12,7));y=rng.normal(size=(12,2));weights=rng.uniform(.1,1.,12)
        w,diag=additive_ridge(x,y,weights,.001)
        q=weights/weights.sum();penalty=.001*np.sum(q[:,None]*x*x)/min(x.shape)
        expected=np.linalg.solve(x.T@(q[:,None]*x)+penalty*np.eye(x.shape[1]),x.T@(q[:,None]*y))
        np.testing.assert_allclose(w,expected,rtol=1e-8,atol=1e-10)
        self.assertLess(diag['full_sum_relative_error'],1e-8)

    def test_common_scale_preserves_cancelling_parts(self):
        c=np.array([[8.,-7.],[0.,0.]]);b=np.array([1.,0.]);h=np.array([[10.,0.],[0.,0.]])
        scale=family_scale(c,b,h,.1)
        self.assertEqual(scale,.125)
        np.testing.assert_allclose((c*scale).sum(axis=1),c.sum(axis=1)*scale)


if __name__=='__main__':unittest.main()
