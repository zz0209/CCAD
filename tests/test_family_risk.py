import numpy as np
from ccad.family_risk import relative_family_risk

def test_cancellation_and_coordinate_reparameterization():
    s=np.eye(2);g=np.array([[1.,-1.],[-1.,1.]])
    assert np.ones(2)@g@np.ones(2)==0
    r=relative_family_risk(s,g);assert np.isclose(r['risk'],2.)
    for t in [np.diag([100.,.01]),np.array([[2.,1.],[-1.,3.]])]:
        assert np.isclose(relative_family_risk(t.T@s@t,t.T@g@t)['risk'],r['risk'],rtol=1e-9)

def test_null_source_error_is_not_erased_by_pseudoinverse():
    assert relative_family_risk(np.diag([1.,0]),np.diag([0.,.5]))['status']=='SOURCE_NULL_MISMATCH'
    assert relative_family_risk(np.diag([1.,0]),np.diag([.2,0]))['risk']==.2
    assert relative_family_risk(np.zeros((2,2)),np.zeros((2,2)))['status']=='NO_SOURCE_ENERGY'

if __name__=='__main__':
    test_cancellation_and_coordinate_reparameterization()
    test_null_source_error_is_not_erased_by_pseudoinverse()
    print('family risk: cancellation, congruence and null-space cases pass')
