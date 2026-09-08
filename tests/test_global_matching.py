import numpy as np
from scipy.optimize import linear_sum_assignment
from prepare_f4_global_matching import full_assignment


def test_blocked_assignment_matches_complete_dense_objective():
    rng=np.random.default_rng(160908)
    source=rng.normal(size=(9,6));target=rng.normal(size=(9,6))
    source[3]=0;target[7]=0
    ns=np.linalg.norm(source,axis=1);nt=np.linalg.norm(target,axis=1)
    denominator=ns[:,None]*nt[None,:]
    correlation=np.divide(source@target.T,denominator,out=np.zeros_like(denominator),where=denominator>0)
    rows,cols=linear_sum_assignment(abs(correlation),maximize=True)
    mapping,scale,meta=full_assignment(source,target,block_columns=2)
    # Zero-atom ties can choose different bijections, so test objective and
    # per-assigned-pair physical scale against the independently built matrix.
    np.testing.assert_allclose(abs(correlation[rows,mapping]).sum(),abs(correlation[rows,cols]).sum(),atol=1e-12)
    expected=np.divide(np.sign(correlation[rows,mapping])*nt[mapping],ns,out=np.zeros_like(ns),where=ns>0)
    np.testing.assert_allclose(scale,expected,atol=1e-12)
    assert len(set(mapping))==9 and meta['zero_norm_source']==meta['zero_norm_target']==1


def test_signed_rescaled_permutation_recovers_paper_pw_mcc():
    rng=np.random.default_rng(91);source=rng.normal(size=(6,11))
    order=np.array([2,4,0,5,1,3]);factor=np.array([2.,-3.,4.,.5,-2.,1.])
    target=source[order]*factor[:,None]
    mapping,scale,meta=full_assignment(source,target,block_columns=2)
    np.testing.assert_array_equal(order[mapping],np.arange(6))
    np.testing.assert_allclose(scale,factor[mapping],rtol=1e-12,atol=1e-12)
    assert abs(meta['mean_absolute_cosine']-1)<1e-12 and meta['negative_matches']==2
