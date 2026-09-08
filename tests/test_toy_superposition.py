import unittest
import numpy as np
import torch
from ccad.toy_superposition import ridge_fit,relu_secant,fit_rectified_state,SAEs,sample


class TestLearnedToyOperations(unittest.TestCase):
    def test_exact_encoder_composition_and_secant_components(self):
        rng=np.random.default_rng(117)
        z=rng.uniform(0,1,(35,6));d=rng.normal(size=(6,4));enc=rng.normal(size=(4,3))
        bt=rng.normal(size=4);bs=rng.normal(size=4);be=rng.normal(size=3)
        linear=d@enc;bias=(bt-bs)@enc+be
        np.testing.assert_allclose(z@linear+bias,(z@d+bt-bs)@enc+be,atol=1e-13)
        z0=z[:17];z1=z[17:34];a0=z0@linear+bias;a1=z1@linear+bias
        q=relu_secant(a0,a1)
        components=(z1-z0)[:,:,None]*linear[None,:,:]*q[:,None,:]
        np.testing.assert_allclose(components.sum(1),np.maximum(a1,0)-np.maximum(a0,0),atol=1e-13)
        self.assertTrue(np.all((q>=0)&(q<=1)))
        np.testing.assert_equal(relu_secant(np.array([-1,0,1]),np.array([-1,0,1])),np.zeros(3))

    def test_rectified_fit_relearns_threshold_on_same_one_member(self):
        x=np.linspace(0,1,401)[:,None];y=np.maximum(2*x-.7,0);mx=x.mean(0);my=y.mean(0)
        coef=ridge_fit(x,y,mx,my,1e-5)
        old=np.mean((np.maximum((x-mx)@coef+my,0)-y)**2)
        new,bias,info=fit_rectified_state(x,y,np.ones((1,1)),coef,mx,my)
        fresh=np.linspace(.0003,.9987,599)[:,None]
        mse=np.mean((np.maximum((fresh-mx)@new+bias,0)-np.maximum(2*fresh-.7,0))**2)
        self.assertLess(mse,old/1000);self.assertLess(info['final_loss'],info['initial_loss'])

    def test_seed_instances_have_equal_gradients_for_equal_initialization(self):
        sae=SAEs(4,8,[91,91],device='cpu');h=torch.linspace(-1,1,60).reshape(15,4)
        pred,_,z=sae(h);loss=((pred-h[:,None,:]).square().sum(2)+.05*z.sum(2)).mean(0).sum();loss.backward()
        for parameter in sae.parameters():torch.testing.assert_close(parameter.grad[0],parameter.grad[1],rtol=0,atol=0)

    def test_coactive_amplitudes_are_distinct_and_duplicate_is_exact(self):
        g=torch.Generator().manual_seed(81);x=sample(10000,12,.1,'coactive',g,'cpu')
        torch.testing.assert_close(x[:,0]>0,x[:,1]>0,rtol=0,atol=0)
        self.assertGreater(float(torch.sum((x[:,0]-x[:,1])**2)),10)
        g=torch.Generator().manual_seed(81);x=sample(10000,12,.1,'correlated',g,'cpu')
        torch.testing.assert_close(x[:,0],x[:,1],rtol=0,atol=0)

    def test_empty_group_is_zero_operation_but_nonzero_truth_error(self):
        x=np.zeros((8,0));y=np.zeros((8,0));coef=ridge_fit(x,y,np.zeros(0),np.zeros(0))
        delta=-(x@coef)@np.zeros((0,4));truth=np.ones((8,4))
        self.assertEqual(float(np.sum(delta**2)),0);self.assertEqual(float(np.sum((delta-truth)**2)),32)

    def test_near_dead_column_does_not_get_unpenalized_huge_weight(self):
        t=np.linspace(0,1,401);y=np.maximum(2*t-.7,0)[:,None]
        x=np.column_stack([t,1e-6*y[:,0]]);mx=x.mean(0);my=y.mean(0)
        coef=ridge_fit(x,y,mx,my,1e-5)
        new,bias,info=fit_rectified_state(x,y,np.ones((1,1)),coef,mx,my)
        self.assertLess(abs(new[1,0]),1);self.assertGreater(info['common_input_scale'],.1)

if __name__=='__main__':unittest.main()
