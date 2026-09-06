"""Source-energy-relative worst fixed-combination error (standard Rayleigh quotient)."""
import numpy as np

def relative_family_risk(source_gram,error_gram,rtol=1e-12):
    s=np.asarray(source_gram,dtype=float);g=np.asarray(error_gram,dtype=float)
    if s.shape!=g.shape or s.ndim!=2 or s.shape[0]!=s.shape[1]:raise ValueError('Gram dimensions differ')
    if not np.isfinite(s).all() or not np.isfinite(g).all():raise ValueError('Nonfinite Gram')
    if not np.allclose(s,s.T,rtol=rtol,atol=rtol) or not np.allclose(g,g.T,rtol=rtol,atol=rtol):raise ValueError('Nonsymmetric Gram')
    s=(s+s.T)/2;g=(g+g.T)/2;sv,u=np.linalg.eigh(s);gv=np.linalg.eigvalsh(g)
    st=rtol*max(float(np.max(abs(sv))),1e-30);gt=rtol*max(float(np.max(abs(gv))),1e-30)
    if sv.min() < -st or gv.min() < -gt:raise ValueError('Gram is not PSD')
    keep=sv>st;null=u[:,~keep]
    if null.shape[1] and np.linalg.norm(g@null,ord=2)>10*gt:
        return dict(status='SOURCE_NULL_MISMATCH',risk=None,source_eigenvalues=sv.tolist(),error_eigenvalues=gv.tolist())
    if not keep.any():return dict(status='NO_SOURCE_ENERGY',risk=None,source_eigenvalues=sv.tolist(),error_eigenvalues=gv.tolist())
    white=u[:,keep]/np.sqrt(sv[keep]);h=white.T@g@white;ev,v=np.linalg.eigh((h+h.T)/2);theta=white@v[:,-1];theta/=np.linalg.norm(theta)
    return dict(status='FINITE',risk=max(0.,float(ev[-1])),worst_unit_coefficients=theta.tolist(),source_eigenvalues=sv.tolist(),error_eigenvalues=gv.tolist(),source_energy_at_worst=float(theta@s@theta),error_energy_at_worst=float(theta@g@theta),rtol=rtol)
