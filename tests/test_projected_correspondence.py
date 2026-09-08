import itertools
import unittest
import numpy as np
from ccad.projected_correspondence import ordered_operator,penalized_low_rank


class ProjectedCorrespondenceTests(unittest.TestCase):
    def test_noncommuting_order_and_continuous_vertex_identity(self):
        p=np.array([[[1.,0.],[0.,0.]],[[.5,.5],[.5,.5]]])
        x=np.array([1.,0.]);a=ordered_operator(p,[1,1])@x;b=ordered_operator(p[::-1],[1,1])@x
        np.testing.assert_allclose(a,[1,0]);np.testing.assert_allclose(b,[1,.5])
        c=np.array([.37,.68]);interpolated=np.zeros(2)
        for vertex in itertools.product([0,1],repeat=2):
            weight=np.prod([c[j] if value else 1-c[j] for j,value in enumerate(vertex)])
            interpolated+=weight*(ordered_operator(p,vertex)@x)
        np.testing.assert_allclose(ordered_operator(p,c)@x,interpolated,atol=1e-14)

    def test_weighted_rrr_matches_independent_whitened_svd(self):
        rng=np.random.default_rng(71);x=rng.normal(size=(17,6));y=rng.normal(size=(17,5))
        q,_=np.linalg.qr(rng.normal(size=(5,3)));factor=q*np.array([.3,1.1,2.2]);inverse=np.linalg.pinv(factor)
        penalty=3.7;gram=x.T@x+penalty*np.eye(6);cross=x.T@y;full=np.linalg.solve(gram,cross)
        left,right,_=penalized_low_rank(full,cross,2,factor,inverse);got=left@right
        upper=np.linalg.cholesky(gram).T
        whitened=np.linalg.solve(upper.T,cross@factor)
        u,s,vt=np.linalg.svd(whitened,full_matrices=False)
        expected=np.linalg.solve(upper,(u[:,:2]*s[:2])@vt[:2])@inverse
        np.testing.assert_allclose(got,expected,rtol=1e-10,atol=1e-11)
        objective=lambda w:np.sum(((x@w-y)@factor)**2)+penalty*np.sum((w@factor)**2)
        self.assertAlmostEqual(objective(got),objective(expected),places=10)
        self.assertLessEqual(objective(got),objective(np.zeros_like(got)))


if __name__=='__main__':unittest.main()
