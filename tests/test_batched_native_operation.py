import unittest
import numpy as np
import torch
from scipy.optimize import lsq_linear
from ccad.native_operation import batched_project_native


class BatchedNativeTests(unittest.TestCase):
    def test_identity_decoder_has_exact_clipped_solution(self):
        v=torch.tensor([[2.,-3.,1.],[-2.,1.,-1.]],dtype=torch.float64)
        z=torch.tensor([[0.,1.,0.],[.5,0.,2.]],dtype=torch.float64)
        d=torch.eye(3,dtype=torch.float64).repeat(2,1,1)
        u,realized,diag=batched_project_native(v,z,d,tolerance=1e-10)
        torch.testing.assert_close(u,torch.maximum(v,-z),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(realized,u)
        self.assertEqual(diag['converged_rows'],2)

    def test_context_specific_coupled_supports_match_scipy(self):
        rng=np.random.default_rng(71);d=rng.normal(size=(3,4,6));v=rng.normal(size=(3,6));z=rng.uniform(0,.4,size=(3,4))
        u,realized,diag=batched_project_native(torch.tensor(v),torch.tensor(z),torch.tensor(d),max_steps=4000,tolerance=1e-9)
        for j in range(3):
            reference=lsq_linear(d[j].T,v[j],bounds=(-z[j],np.inf),tol=1e-13,lsmr_tol=1e-13)
            np.testing.assert_allclose(u[j].numpy(),reference.x,atol=2e-7,rtol=2e-7)
        self.assertEqual(diag['converged_rows'],3)
        self.assertGreaterEqual(diag['minimum_final_state'],-1e-12)


if __name__=='__main__':unittest.main()
