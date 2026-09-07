"""Source-component operation metrics and reusable signed allocations.

The mask second moment and weighted least-squares identities are classical.
The source feature axes define the operation family; changing those axes in
general changes that family. No semantic or target-native identity is implied.
"""
from __future__ import annotations
import numpy as np


def component_metric(decoder, family='half'):
    """Return E[diag(a) G diag(a)], with source decoder rows and G=D D.T.

whole: a=1; half: independent Bernoulli(1/2); singleton: uniformly choose
one source member. These are exact expectations, not sampled mask estimates.
"""
    d=np.asarray(decoder,dtype=np.float64)
    if d.ndim!=2 or not len(d) or not np.isfinite(d).all():raise ValueError('Finite nonempty decoder matrix required')
    g=d@d.T
    if family=='whole':return g
    if family=='half':return .25*(g+np.diag(np.diag(g)))
    if family=='singleton':return np.diag(np.diag(g))/len(g)
    raise ValueError('Unknown source mask family')


def metric_factor(metric):
    """Full-rank factor L with L L.T=M, for row-response Z L.

Reject a singular component metric. Aggregate-only predictions may have a
minimum-norm lift, but cannot in general identify arbitrary source masks.
"""
    m=np.asarray(metric,dtype=np.float64)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or not np.isfinite(m).all():raise ValueError('Finite square metric required')
    if not np.allclose(m,m.T,atol=1e-12,rtol=1e-12):raise ValueError('Symmetric metric required')
    return np.linalg.cholesky(m)


def component_update(source_codes, decoder, scales):
    z=np.asarray(source_codes,dtype=np.float64);d=np.asarray(decoder,dtype=np.float64);a=np.asarray(scales,dtype=np.float64)
    if d.ndim!=2 or z.shape[-1]!=len(d) or a.shape!=(len(d),):raise ValueError('Source code/decoder/mask dimensions differ')
    if not np.isfinite(z).all() or not np.isfinite(d).all() or not np.isfinite(a).all():raise ValueError('Finite source allocation required')
    return (z*a)@d


def predefined_masks(count, seed):
    """Source-only controls fixed by rank order and RNG; no outcome selection."""
    if count<2:raise ValueError('At least two source members required')
    first=np.zeros(count);first[:count//2]=1
    alt=(np.arange(count)%2==0).astype(float)
    single=np.zeros(count);single[0]=1
    masks=[('whole',np.ones(count)),('rank_first_half',first),('rank_second_half',1-first),('alternating',alt),('rank_first_member',single)]
    rng=np.random.default_rng(seed)
    for index in range(2):
        mask=np.zeros(count);mask[rng.choice(count,count//2,replace=False)]=1;masks.append((f'random_half_{index}',mask))
    return masks
