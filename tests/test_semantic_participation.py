import torch
from ccad.semantic_participation import union_participation,project_sparse_gates,state_delta
from ccad.union_family import binary_controls,vertex_weights


def test_union_matches_sequential_fixed_donor_mixing():
    base=torch.tensor([0.,4.,1.]);donor=torch.tensor([3.,0.,5.])
    g=torch.tensor([[1.,.6],[.2,.8],[0.,.5]])
    c=torch.tensor([.25,1.]);alpha=union_participation(g,c)
    joint=(1-alpha)*base+alpha*donor
    sequential=base
    for j in range(2):sequential=(1-g[:,j]*c[j])*sequential+g[:,j]*c[j]*donor
    torch.testing.assert_close(joint,sequential)
    assert joint.min()>=0 and alpha.max()<=1
    # Shared membership saturates; adding two whole-control deltas is different.
    assert union_participation(torch.tensor([[1.,1.]]),torch.ones(2)).item()==1


def test_sparse_projection_uses_distance_gain_and_equal_weight_budget():
    v=torch.tensor([[4.,.8],[.9,.1],[-2.,.7]])
    shared=v.clone();exclusive=v.clone()
    project_sparse_gates(shared,2);project_sparse_gates(exclusive,2,True)
    # Projection keeps the clipped large entry by distance reduction, not ties.
    torch.testing.assert_close(shared,torch.tensor([[1.,0.],[.9,0.],[0.,0.]]))
    torch.testing.assert_close(exclusive,shared)
    assert int((shared>0).sum())==int((exclusive>0).sum())==2


def test_union_cube_interpolation_and_worst_case_hook_error():
    torch.manual_seed(160908)
    controls=torch.rand(61,3,dtype=torch.float64)
    base=torch.rand(7,5,dtype=torch.float64);donor=torch.rand_like(base)
    ds=torch.randn(5,4,dtype=torch.float64);dt=torch.randn(5,4,dtype=torch.float64)
    gs=torch.rand(5,3,dtype=torch.float64);gt=torch.rand_like(gs)
    vertices=binary_controls(3,dtype=torch.float64)
    errors=torch.stack([state_delta(base,donor,ds,gs,c)-state_delta(base,donor,dt,gt,c) for c in vertices])
    weights=vertex_weights(controls)
    torch.testing.assert_close(weights.sum(1),torch.ones(61,dtype=torch.float64))
    direct=torch.stack([state_delta(base,donor,ds,gs,c)-state_delta(base,donor,dt,gt,c) for c in controls])
    interpolated=torch.einsum('cv,vnh->cnh',weights,errors)
    torch.testing.assert_close(direct,interpolated,atol=1e-12,rtol=1e-12)
    assert torch.linalg.vector_norm(direct.flatten(1),dim=1).max()<=torch.linalg.vector_norm(errors.flatten(1),dim=1).max()+1e-12


def test_matching_single_controls_misses_shared_intersection():
    # Both dictionaries exactly reconstruct the same scalar base=0, donor=1.
    # The target has a third signed decoder atom that compensates full state.
    bs=torch.zeros(1);ds=torch.ones(1);sd=torch.ones(1,1);sg=torch.ones(1,2)
    bt=torch.zeros(3);dt=torch.ones(3);td=torch.tensor([[1.],[1.],[-1.]])
    tg=torch.tensor([[1.,0.],[0.,1.],[0.,0.]])
    torch.testing.assert_close(ds@sd,dt@td)
    for c in torch.eye(2):
        torch.testing.assert_close(state_delta(bs,ds,sd,sg,c),state_delta(bt,dt,td,tg,c))
    assert state_delta(bs,ds,sd,sg,torch.ones(2)).item()==1
    assert state_delta(bt,dt,td,tg,torch.ones(2)).item()==2
