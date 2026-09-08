"""A distributional counterexample and unit-invariance witness for retrieval."""
import numpy as np
from ccad.semantic_context_matching import top_distributions,retrieve,entropic_dual_plan
from ccad.nip_baselines import _balanced_log_sinkhorn


def test_equal_centroids_do_not_force_equal_context_match():
    # The first candidate is concentrated at zero; the second shares the two
    # endpoint contexts. Both centroids are zero, but only the latter is the
    # same distribution as the query. Tie-breaking alone would choose first.
    reference=np.array([[-1.],[0.],[1.]])
    source_ix=np.array([[0],[0],[0]])
    source_act=np.array([[2.],[0.],[2.]])
    target_ix=np.array([[1],[0],[1]])
    target_act=np.array([[3.],[6.],[3.]])
    def run(a,b):
        sd,sc,sn=top_distributions(source_ix,a,1,2,reference)
        td,tc,tn=top_distributions(target_ix,b,2,2,reference)
        return retrieve([0],sd,td,sc,tc,sn,tn,reference,candidate_count=2,minimum_count=1,
                        regularization=.01,tolerance=1e-10,max_iterations=1000)[0]
    original=run(source_act,target_act)
    scaled=run(source_act*7,target_act*13)
    assert original['centroid_target_member']==0
    assert original['target_member']==1
    assert original['distance']<1e-8
    assert original['candidates'][1]['distance']==1
    assert original==scaled


def test_dual_matches_the_same_regularized_transport_problem():
    cost=np.array([[0.,.7,1.8],[.4,.2,1.2],[1.7,.8,.1]])
    a=np.array([.5,.3,.2]);b=np.array([.3,.4,.3])
    reference,_,ok,_=_balanced_log_sinkhorn(cost,a,b,regularization=.17,tolerance=1e-10,max_iterations=10000)
    plan,_,converged,error=entropic_dual_plan(cost,a,b,.17,1e-8)
    assert ok and converged and error<=1e-8
    assert np.max(np.abs(reference-plan))<1e-8


if __name__=='__main__':
    test_equal_centroids_do_not_force_equal_context_match()
    test_dual_matches_the_same_regularized_transport_problem()
    print('SEMANTIC_CONTEXT_DISTRIBUTION_WITNESS_PASS')
