"""Finite fixtures for section13 identities; not proofs of empirical assumptions."""
import unittest
import numpy as np


class OperationalIdentitiesTests(unittest.TestCase):
    def test_donor_mean_cancellation_and_iid_variance(self):
        r=np.array([[1.,-2.],[3.,4.]])
        z=np.array([[1.,3.],[4.,-2.],[0.,1.]])
        mu=np.array([9.,-7.]);c=(z-mu)@r.T
        np.testing.assert_allclose(c[:,None]-c[None,:],(z[:,None]-z[None,:])@r.T)
        pair_error=np.mean(np.sum((c[:,None]-c[None,:])**2,axis=-1))
        variance=2*(np.mean(np.sum(c*c,axis=-1))-np.sum(c.mean(0)**2))
        self.assertAlmostEqual(pair_error,variance)

    def test_component_gram_and_full_cancellation(self):
        v=np.array([2.,-3.]);e=np.stack([v,-v],axis=1);gram=e.T@e
        self.assertEqual(float(np.ones(2)@gram@np.ones(2)),0.)
        self.assertEqual(gram[0,0],13.)
        theta=np.array([.2,-.7])
        self.assertAlmostEqual(float(theta@gram@theta),float(np.sum((e@theta)**2)))

    def test_average_jacobian_is_not_average_gram(self):
        js=np.array([[[1.,2.]], [[-1.,-2.]]])
        mean=js.mean(0);gram=np.mean(np.transpose(js,(0,2,1))@js,axis=0)
        self.assertEqual(float(np.sum(mean*mean)),0.)
        self.assertEqual(float(np.trace(gram)),5.)

    def test_quadratic_interaction(self):
        h=np.array([.2,.7]);u=np.array([.1,-.2]);v=np.array([.3,.4])
        f=lambda x:.5*float(x@x)
        interaction=f(h-u-v)-f(h-u)-f(h-v)+f(h)
        self.assertAlmostEqual(interaction,float(u@v))
        self.assertLessEqual(abs(interaction),np.linalg.norm(u)*np.linalg.norm(v))

    def test_logit_bounds_finite_fixture(self):
        rng=np.random.default_rng(31)
        def lp(x):
            x=x-x.max();return x-np.log(np.exp(x).sum())
        for _ in range(50):
            s=rng.normal(size=7);t=rng.normal(size=7);a,b=lp(s),lp(t)
            kl=float(np.exp(a)@(a-b));error=float(np.sum((t-s)**2))
            self.assertLessEqual(kl,.25*error+1e-12)
            self.assertLessEqual(float(np.max(np.abs(a-b))),np.sqrt(2*error)+1e-12)


if __name__=='__main__':unittest.main()
