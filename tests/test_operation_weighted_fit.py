import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from ccad.operation_weighted_fit import fit_operation_weighted

def test_explicit_stacked_least_squares():
    rng=np.random.default_rng(14);x=rng.normal(size=(30,5));y=rng.normal(size=(30,2));a=rng.normal(size=(30,2,2));q=a.transpose(0,2,1)@a;b0=rng.normal(size=(5,2));b,d=fit_operation_weighted(x,y,q,b0,.1)
    scale=np.sqrt(np.mean(x*x,axis=0));rows=[];targets=[]
    for xi,yi,qi in zip(x/scale,y,q):
        root=np.linalg.cholesky(qi).T;rows.append(root@np.kron(xi[None,:],np.eye(2))/np.sqrt(len(x)));targets.append(root@yi/np.sqrt(len(x)))
    design=np.concatenate(rows);target=np.concatenate(targets);lam=.1*np.sum(design**2)/10
    expected=np.linalg.lstsq(np.vstack([design,np.sqrt(lam)*np.eye(10)]),np.concatenate([target,np.sqrt(lam)*(scale[:,None]*b0).ravel()]),rcond=None)[0].reshape(5,2)/scale[:,None]
    assert np.allclose(b,expected,rtol=1e-10,atol=1e-10);assert d['fitted_objective']<=d['prior_objective'];assert d['normal_equation_relative_error']<1e-12
    unchanged,dd=fit_operation_weighted(x,y,np.zeros_like(q),b0);assert np.array_equal(unchanged,b0) and dd['status']=='NO_WEIGHTED_DESIGN_ENERGY'
if __name__=='__main__':test_explicit_stacked_least_squares();print('OPERATION_WEIGHTED_FIT_CHECK_PASS')
