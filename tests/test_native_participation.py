import numpy as np
from ccad.native_participation import participant_delta,project_participation_rows,feasibility


def test_native_participation_composes_and_preserves_nonnegative_states():
    rng=np.random.default_rng(915)
    base=rng.random((3,4,5));donor=rng.random((3,4,5));decoder=rng.normal(size=(5,7))
    g=rng.random((5,2));g/=np.maximum(g.sum(1,keepdims=True),1)
    a,za=participant_delta(base,donor,decoder,g,[1,0])
    b,zb=participant_delta(base,donor,decoder,g,[0,1])
    whole,z=participant_delta(base,donor,decoder,g,[1,1])
    np.testing.assert_allclose(a+b,whole,atol=1e-14)
    assert min(z.min(),za.min(),zb.min())>=0
    np.testing.assert_allclose(whole,(z-base)@decoder,atol=1e-14)
    # Decoder permutation changes neither the native edit nor membership count.
    order=np.array([3,0,4,1,2])
    moved,_=participant_delta(base[...,order],donor[...,order],decoder[order],g[order],[1,1])
    np.testing.assert_allclose(moved,whole,atol=1e-14)


def test_sparse_row_projection_has_expected_boundary_solution():
    import torch
    g=torch.tensor([[1.2,.8],[-2.,3.],[.1,.2],[-.1,-.2]],dtype=torch.float64)
    project_participation_rows(g,2)
    np.testing.assert_allclose(g.numpy(),[[.7,.3],[0,1],[0,0],[0,0]],atol=1e-14)
    info=feasibility(g.numpy())
    assert info['nonnegative'] and info['row_sum_at_most_one'] and info['actual_members']==2


def test_exclusive_projection_selects_squared_distance_gain():
    import torch
    from ccad.native_participation import project_exclusive_rows
    # Clipped norm alone would tie rows 0 and 1; distance gain selects row 1.
    values=torch.tensor([[1.1,.9],[5.,-3.],[.4,.6],[-1.,-2.]],dtype=torch.float64)
    project_exclusive_rows(values,1)
    np.testing.assert_array_equal(values.numpy(),[[0,0],[1,0],[0,0],[0,0]])
