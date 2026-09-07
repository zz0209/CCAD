import unittest
import numpy as np
from ccad.complete_support import orthogonal_least_squares_support, pair_design


class CompleteSupportTests(unittest.TestCase):
    def test_each_greedy_step_matches_direct_refit(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(18,8));x[:,7]=x[:,0]+.02*x[:,7]
        y=x[:,[0,3,5]]@rng.normal(size=(3,4))+.01*rng.normal(size=(18,4))
        selected,d=orthogonal_least_squares_support(x,y,4);support=[]
        for member,step in zip(selected,d['path']):
            candidates=[]
            for j in range(x.shape[1]):
                if j in support:candidates.append(np.inf);continue
                z=x[:,support+[j]];candidates.append(np.sum((y-z@np.linalg.lstsq(z,y,rcond=None)[0])**2))
            self.assertEqual(member,int(np.argmin(candidates)));support.append(int(member))
            self.assertAlmostEqual(step['predicted_sse_gain'],step['actual_sse_gain'],places=9)

    def test_joint_objective_and_basis_rotation(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(20,7));y=rng.normal(size=(20,3));donors=np.arange(20)^1
        a,b=pair_design(x,y,donors,.2);w=rng.normal(size=(7,3));e=x@w-y
        expected=np.sum(((e-e[donors])/2)**2)+.2*np.sum(((e+e[donors])/2)**2)
        self.assertAlmostEqual(np.sum((a@w-b)**2),expected,places=10)
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        s,_=orthogonal_least_squares_support(a,b,4)
        t,_=orthogonal_least_squares_support(a*np.arange(1,8),b@q,4)
        np.testing.assert_array_equal(s,t)

    def test_nonnegative_contrast_invisible_member(self):
        x=np.array([[0.,1.],[1.,1.],[0.,2.],[1.,2.]])
        y=x@np.array([[1.],[5.]]);donors=np.array([1,0,3,2])
        a,b=pair_design(x,y,donors,0);s,_=orthogonal_least_squares_support(a,b,2)
        np.testing.assert_array_equal(s,[0])
        a,b=pair_design(x,y,donors,1);s,_=orthogonal_least_squares_support(a,b,2)
        self.assertEqual(set(s),{0,1})


if __name__=='__main__':unittest.main()
