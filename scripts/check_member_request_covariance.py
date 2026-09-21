from pathlib import Path
from datetime import datetime, timezone
from itertools import permutations
import json

import numpy as np


def main():
    rng=np.random.default_rng(92105)
    records=[]
    for n in range(2,8):
        h=np.eye(n)-np.ones((n,n))/n
        e=rng.normal(size=(5,n))
        for alpha in [.0,.13,.5,.83,1.]:
            s=n*alpha;k=int(np.floor(s));f=s-k
            vertex=np.zeros(n);vertex[:k]=1
            if k<n:vertex[k]=f
            q=np.asarray(list(set(permutations(vertex.tolist()))))
            centered=q-alpha
            observed=centered.T@centered/len(q)
            lam=(k+f*f-n*alpha*alpha)/(n-1)
            covariance_error=float(np.max(np.abs(observed-lam*h)))
            risk=float(np.mean(np.sum((q@e.T)**2,axis=1)))
            predicted=float(np.sum((e@(alpha*np.ones(n)))**2)+lam*np.sum((e@h)**2))
            assert covariance_error<1e-12 and abs(risk-predicted)<1e-10
            records.append(dict(n=n,alpha=alpha,permutations=len(q),covariance_error=covariance_error,
                                risk_error=abs(risk-predicted)))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                scope='Exact enumeration checks of the fixed-state covariance and risk identities. This is a mathematical implementation check, independent of model outcomes.',
                cases=records)
    path=Path('artifacts/final_science_20260921_round05/COVARIANCE_CHECK.json')
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),maximum_covariance_error=max(r['covariance_error'] for r in records),
                         maximum_risk_error=max(r['risk_error'] for r in records))))


if __name__=='__main__':
    main()
