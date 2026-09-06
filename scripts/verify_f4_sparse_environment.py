"""Seeded CPU solver witness for the isolated F4 sparse-fit overlay."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import MultiTaskLasso
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT/'.aris/compute/local-f4-sparse-env-spec.json').read_text())
versions = {name: metadata.version(name) for name in ('numpy', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl', 'narwhals')}
assert versions == dict(numpy='2.5.2', **{'scikit-learn':'1.9.0', 'scipy':'1.18.1', 'joblib':'1.5.2', 'threadpoolctl':'3.6.0', 'narwhals':'2.25.0'})
assert sys.version.split()[0] == '3.13.7'
rng = np.random.default_rng(40216)
x = rng.normal(size=(96, 12)); beta = np.zeros((12, 2)); beta[[1, 4]] = [[2., -1.], [-1., 3.]]
y = x @ beta + np.array([3., -2.]); weights = rng.uniform(.2, 1, 96); weights /= weights.sum()
with threadpool_limits(limits=4):
    fitted = MultiTaskLasso(alpha=.02, tol=1e-10, max_iter=10000).fit(x, y, sample_weight=weights)
    xc=x-weights@x; yc=y-weights@y; scale=np.sqrt(len(x)*weights)[:,None]
    transformed=MultiTaskLasso(alpha=.02, fit_intercept=False, tol=1e-10, max_iter=10000).fit(xc*scale, yc*scale)
support=np.flatnonzero(np.linalg.norm(fitted.coef_,axis=0)>0).tolist()
error=float(np.max(np.abs(fitted.coef_-transformed.coef_)))
assert support == [1,4] and error < 1e-10 and fitted.dual_gap_ < 1e-8
print(json.dumps(dict(sentinel='F4_SPARSE_ENV_READY', python=sys.version.split()[0], versions=versions,
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    support=support, weighted_transform_error=error, dual_gap=float(fitted.dual_gap_), coef=fitted.coef_.tolist())))
