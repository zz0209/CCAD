import unittest
import numpy as np
import torch
from ccad.semantic_family import SharedDAS,get_bases,apply_numpy,apply_torch


class FamilyTests(unittest.TestCase):
    def test_shared_blocks_commute_and_add(self):
        torch.manual_seed(18);prior=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64);op=SharedDAS(11,3,3)
        finally:torch.set_default_dtype(prior)
        x=torch.randn(9,11,dtype=torch.float64)
        b=get_bases(op);c=torch.tensor([.25,.75,.5],dtype=torch.float64)
        y=op(x,None,c)
        self.assertTrue(torch.allclose(y,apply_torch(x,b,c),atol=2e-13,rtol=2e-13))
        self.assertTrue(torch.allclose(y,apply_torch(x,b,c,True),atol=2e-13,rtol=2e-13))
        np.testing.assert_allclose(y.detach().numpy(),apply_numpy(x.numpy(),b.detach().numpy(),c.numpy()),atol=2e-13)
        grad=torch.autograd.grad(y.square().sum(),op.rotation.parametrizations.weight.original)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)

    def test_order_counterexample_and_request_validation(self):
        b=np.array([[[1.,0.]],[[2**-.5,2**-.5]]]);x=np.array([[1.,0.]])
        np.testing.assert_allclose(apply_numpy(x,b,[1,1]),[[1,0]],atol=1e-14)
        np.testing.assert_allclose(apply_numpy(x,b,[1,1],True),[[1,.5]],atol=1e-14)
        with self.assertRaises(ValueError):apply_numpy(x,b,[1,2])


if __name__=='__main__':unittest.main()
