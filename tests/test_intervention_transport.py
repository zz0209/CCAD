"""Analytic checks for capacity and preservation of the pre-capacity support."""
import unittest
from types import SimpleNamespace
import torch
from ccad.intervention_transport import refine_columns


class FixedTarget:
    def __init__(self,decoder,encoder,code):
        self.decoder=SimpleNamespace(weight=decoder)
        self.encoder=SimpleNamespace(weight=encoder)
        self.code=code
    def encode(self,h):return self.code.expand(*h.shape[:-1],len(self.code))


class TestInterventionTransport(unittest.TestCase):
    def test_joint_capacity_analytic_solution(self):
        eye=torch.eye(2,dtype=torch.float64);h=torch.ones(1,2,dtype=torch.float64)
        target=FixedTarget(eye,eye,torch.tensor([.5,2.],dtype=h.dtype))
        source=dict(center=torch.zeros(2,dtype=h.dtype),encoder=eye,encoder_bias=torch.zeros(2,dtype=h.dtype),decoder=torch.tensor([[1.,0.],[1.,1.]],dtype=h.dtype))
        actual=refine_columns(h,target,source,torch.full((1,2,2),-.1,dtype=h.dtype),2)
        expected=torch.tensor([[[-.25,-.25],[0.,-1.]]],dtype=h.dtype)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-10))

    def test_clipped_support_cannot_recruit_zero_ties(self):
        h=torch.tensor([[1.,0.]],dtype=torch.float64)
        target=FixedTarget(torch.tensor([[-1.,0.,1.],[0.,1.,0.]],dtype=h.dtype),
                           torch.tensor([[0.,0.],[0.,0.],[10.,0.]],dtype=h.dtype),torch.zeros(3,dtype=h.dtype))
        source=dict(center=torch.zeros(2,dtype=h.dtype),encoder=torch.tensor([[1.,0.]],dtype=h.dtype),
                    encoder_bias=torch.zeros(1,dtype=h.dtype),decoder=torch.tensor([[1.,0.]],dtype=h.dtype))
        # Initial selection is member2. Capacity clips its only negative entry.
        # Member0 would implement the update perfectly but was never selected.
        actual=refine_columns(h,target,source,torch.zeros(1,3,1,dtype=h.dtype),1)
        self.assertTrue(torch.equal(actual,torch.zeros_like(actual)))


if __name__=='__main__':unittest.main()
