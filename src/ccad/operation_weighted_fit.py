"""Fixed-support multivariate quadratic regression, anchored to a saved map."""
import numpy as np

def fit_operation_weighted(x, y, grams, prior, ridge_fraction=0.1):
    x,y,q,b0=(np.asarray(a,dtype=float) for a in (x,y,grams,prior))
    n,p=x.shape;k=y.shape[1]
    if y.shape[0]!=n or q.shape!=(n,k,k) or b0.shape!=(p,k) or ridge_fraction<=0:
        raise ValueError('Inconsistent dimensions or nonpositive ridge')
    if not all(np.isfinite(a).all() for a in (x,y,q,b0)) or not np.allclose(q,q.transpose(0,2,1),atol=1e-10,rtol=1e-10):
        raise ValueError('Nonfinite or asymmetric inputs')
    if np.linalg.eigvalsh(q).min() < -1e-10*max(1.,abs(q).max()):raise ValueError('Indefinite Gram')
    scale=np.sqrt(np.mean(x*x,axis=0));scale=np.where(scale>1e-12,scale,1.);u=x/scale;c0=scale[:,None]*b0
    h=np.zeros((p*k,p*k));rhs=np.zeros(p*k)
    # Row-major vec(C): each design row maps vec(C) to u_i C.
    for ui,yi,qi in zip(u,y,q):
        design=np.kron(ui[None,:],np.eye(k));h+=design.T@qi@design/n;rhs+=design.T@qi@yi/n
    tr=float(np.trace(h));lam=ridge_fraction*tr/(p*k)
    if tr<=0:return b0.copy(),dict(status='NO_WEIGHTED_DESIGN_ENERGY',unchanged=True,design_rank=int(np.linalg.matrix_rank(u)))
    c=np.linalg.solve(h+lam*np.eye(p*k),rhs+lam*c0.ravel()).reshape(p,k);b=c/scale[:,None]
    def loss(cc):
        err=u@cc-y
        return float(np.einsum('ni,nij,nj->',err,q,err)/n+lam*np.sum((cc-c0)**2))
    normal=(h+lam*np.eye(p*k))@c.ravel()-rhs-lam*c0.ravel()
    return b,dict(status='FIT',parameters=p*k,rows=n,design_rank=int(np.linalg.matrix_rank(u)),gram_rank=int(np.linalg.matrix_rank(h)),ridge=lam,ridge_fraction=ridge_fraction,prior_objective=loss(c0),fitted_objective=loss(c),normal_equation_relative_error=float(np.linalg.norm(normal)/max(np.linalg.norm(rhs+lam*c0.ravel()),1e-30)),coefficient_delta_norm=float(np.linalg.norm(b-b0)),prior_norm=float(np.linalg.norm(b0)),fitted_norm=float(np.linalg.norm(b)),scales=scale.tolist())
