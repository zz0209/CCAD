import torch
from ccad.ravel_controls import MultiDBM,MultiDAS

def test_dbm_binary_empty_and_shared_union():
    op=MultiDBM(4,3).eval();x=torch.arange(8,dtype=torch.float32).reshape(2,4)
    assert torch.equal(op(x,None,torch.ones(3)),torch.zeros_like(x))
    with torch.no_grad():op.mask[1,0]=1;op.mask[1,1]=1;op.mask[3,1]=1
    out=op(x,None,torch.tensor([1.,1.,0.]))
    assert torch.equal(out,x*torch.tensor([0.,1.,0.,1.]))

def test_das_single_matches_original_projection_and_is_idempotent():
    torch.manual_seed(107);op=MultiDAS(9,3,2);x=torch.randn(5,9);w=op.rotations[1].weight
    y=op(x,None,torch.tensor([0.,1.,0.]));original=(x@w.T)@w
    assert torch.allclose(y,original,atol=1e-6)
    assert torch.allclose(y,(y@w.T)@w,atol=1e-6)
    assert torch.allclose(w@w.T,torch.eye(2),atol=1e-6)

def test_compact_position_scatter_preserves_values_and_gradients():
    import numpy as np
    from types import SimpleNamespace
    from scripts.run_ravel_semantic_source import expand_delta
    w=SimpleNamespace(torch=torch,device=torch.device('cpu'),semantic_positions=np.array([1,3,2]),max_length=5,dim=4)
    ids=np.array([2,0]);x=torch.randn(2,4,requires_grad=True)
    y=expand_delta(w,ids,x)
    assert torch.equal(y[0,2],x[0]) and torch.equal(y[1,1],x[1])
    assert torch.count_nonzero(y).item()==torch.count_nonzero(x).item()
    y.square().sum().backward();assert torch.allclose(x.grad,2*x)
