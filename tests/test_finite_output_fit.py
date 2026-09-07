import unittest
import torch
from ccad.finite_output_fit import SubspaceLayerNormHead,DirectLayerNormHead,fit_finite_map


class TestSubspaceLayerNormHead(unittest.TestCase):
    def test_lbfgs_accepts_transposed_saved_map(self):
        torch.manual_seed(830)
        x=torch.randn(8,3,dtype=torch.float64);truth=torch.randn(3,2,dtype=torch.float64)
        initial=torch.randn(2,3,dtype=torch.float64).T
        self.assertFalse(initial.is_contiguous())
        h=torch.randn(8,11,dtype=torch.float64);d=torch.randn(2,11,dtype=torch.float64)
        norm=torch.nn.LayerNorm(11,dtype=torch.float64);linear=torch.nn.Linear(11,7,dtype=torch.float64)
        for parameter in list(norm.parameters())+list(linear.parameters()):parameter.requires_grad_(False)
        head=DirectLayerNormHead(h,d,norm,linear)
        weights,diag=fit_finite_map(head,x,x@truth,initial,torch.tensor([1,0,3,2,5,4,7,6]),[('whole',torch.ones(2,dtype=torch.float64))],torch.arange(6),torch.arange(6,8),dict(fit_batch_size=2,anchor_fraction=.01,lbfgs_inner_steps=2,history_size=3,fit_checkpoints=1))
        self.assertEqual(weights.shape,(3,2))
        self.assertLess(diag['history'][1]['train_kl'],diag['history'][0]['train_kl'])

    def test_direct_head_preserves_production_update_rounding(self):
        torch.manual_seed(903)
        h=torch.randn(6,19,dtype=torch.float32)
        d=torch.randn(3,19,dtype=torch.float64)
        norm=torch.nn.LayerNorm(19);head=torch.nn.Linear(19,29,bias=False)
        direct=DirectLayerNormHead(h,d,norm,head)
        ids=torch.tensor([1,3,5]);c=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
        update=torch.stack([sum(c[i,k]*d[k] for k in range(3)) for i in range(3)]).float()
        expected=torch.log_softmax(head(norm(h[ids]+update)).double(),-1)
        actual=direct.logprobs(c,ids)
        self.assertLess(float((actual-expected).abs().max().detach()),1e-6)
        a=torch.autograd.grad(actual[:,4].sum(),c,retain_graph=True)[0]
        b=torch.autograd.grad(expected[:,4].sum(),c)[0]
        self.assertLess(float((a-b).abs().max()),1e-6)

    def test_original_logits_and_gradients_with_affine_normalization(self):
        torch.manual_seed(731)
        h=torch.randn(9,23,dtype=torch.float64)+2.
        d=torch.randn(4,23,dtype=torch.float64)
        norm=torch.nn.LayerNorm(23,eps=.003,dtype=torch.float64)
        head=torch.nn.Linear(23,31,bias=True,dtype=torch.float64)
        with torch.no_grad():norm.weight.copy_(torch.randn(23));norm.bias.copy_(torch.randn(23))
        fast=SubspaceLayerNormHead(h,d,norm,head)
        ids=torch.tensor([1,4,8]);c=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        direct=head(norm(h[ids]+c@d));projected=fast.logits(c,ids)
        self.assertLess(float((direct-projected).abs().max()),1e-12)
        teacher=torch.softmax(torch.randn(3,31,dtype=torch.float64),-1)
        direct_loss=-(teacher*torch.log_softmax(direct,-1)).sum()
        projected_loss=-(teacher*fast.logprobs(c,ids)).sum()
        a=torch.autograd.grad(direct_loss,c,retain_graph=True)[0]
        b=torch.autograd.grad(projected_loss,c)[0]
        self.assertLess(float((a-b).abs().max()),1e-12)

    def test_constant_decoder_shift_has_no_output_or_gradient(self):
        torch.manual_seed(114)
        h=torch.randn(5,13,dtype=torch.float64);d=torch.ones(1,13,dtype=torch.float64)
        norm=torch.nn.LayerNorm(13,dtype=torch.float64);head=torch.nn.Linear(13,17,bias=False,dtype=torch.float64)
        fast=SubspaceLayerNormHead(h,d,norm,head)
        c=torch.randn(5,1,dtype=torch.float64,requires_grad=True);ids=torch.arange(5)
        logits=fast.logits(c,ids)
        self.assertLess(float((logits-head(norm(h))).abs().max()),1e-12)
        gradient=torch.autograd.grad(logits.sum(),c)[0]
        self.assertEqual(float(gradient.abs().max()),0.)


if __name__=='__main__':unittest.main()
