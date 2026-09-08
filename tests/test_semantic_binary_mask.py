import torch
from ccad.semantic_binary_mask import SemanticBinaryMask


def test_code_mask_decodes_true_native_difference_and_empty_is_zero():
    model=SemanticBinaryMask(3,2)
    decoder=torch.tensor([[1.,2.],[-3.,1.],[2.,0.]])
    dz=torch.tensor([[2.,-1.,.5]])
    with torch.no_grad():model.mask.copy_(torch.tensor([[1.,1.],[-1.,1.],[1.,-1.]]))
    model.eval()
    torch.testing.assert_close(model(dz,decoder,torch.tensor([1.,0.])),(dz*torch.tensor([1.,0.,1.]))@decoder)
    torch.testing.assert_close(model(dz,decoder,torch.ones(2)),dz@decoder)
    with torch.no_grad():model.mask.fill_(-1)
    torch.testing.assert_close(model(dz,decoder,torch.ones(2)),torch.zeros(1,2))
