"""Compare conditional predictions on identical cases with matching standalone edits."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def identity(p):
    return dict(path=p.as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    methods = plan['compared_methods']
    masks, errors, changes, details, inputs = [], [], [], [], [identity(args.plan)]
    source_late, source_old = [], []
    for seed in plan['target_seeds']:
        old = ROOT/'runs'/f'MORNING_R05_restore_confirm_seed{seed}_v1_20260916'
        late = ROOT/'runs'/f'MORNING_R06_standalone_seed{seed}_v2_20260916'
        for folder in [old, late]:
            assert json.loads((folder/'status.json').read_text())['status'] == 'PASS'
        ix = json.loads((old/'BANK_INDEX.json').read_text())
        il = json.loads((late/'BANK_INDEX.json').read_text())
        x = np.load(old/'responses.npz')['logits'].astype(float)
        l = np.load(late/'responses.npz')['logits'].astype(float)
        assert ix['documents'] == il['documents']
        assert ix['methods'] == il['methods']
        si, ni = ix['methods'].index('source'), ix['methods'].index('none')
        assert np.array_equal(x[ni,0], l[ni,0])
        source_late.append(l[si]); source_old.append(x[si])
        old_lookup = {q['name']: j for j,q in enumerate(ix['queries'])}
        late_lookup = {(q['cut'],q['part']): j for j,q in enumerate(il['queries'])}
        tm, te, tc = [], [], []
        for qi,q in enumerate(ix['queries']):
            if q['role'] != 'restore':
                continue
            ai = old_lookup[q['ablation_reference']]
            bi = late_lookup[(q['cut'],q['restore_part'])]
            accept = np.ones(x.shape[-1], dtype=bool)
            for method in methods:
                j = ix['methods'].index(method)
                accept &= (x[j,ai] > 0) == (x[si,ai] > 0)
                accept &= (l[j,bi] > 0) == (l[si,bi] > 0)
            err = np.stack([(x[ix['methods'].index(m),qi] > 0) != (x[si,qi] > 0)
                            for m in methods])
            change = (x[si,qi] > 0) != (x[si,ai] > 0)
            tm.append(accept); te.append(err); tc.append(change)
            details.append(dict(seed=seed, request=q['name'], cases=int(accept.sum()),
                conditional_changed=int((accept&change).sum()),
                wrong={m:int((err[j]&accept).sum()) for j,m in enumerate(methods)}))
        masks.append(tm); errors.append(te); changes.append(tc)
        inputs.extend(identity(folder/file) for folder in [old,late]
                      for file in ['BANK_INDEX.json','responses.npz'])
    assert all(np.array_equal(source_late[0],v) for v in source_late)
    assert all(np.array_equal(source_old[0],v) for v in source_old)
    mask = np.asarray(masks) # T,Q,N
    error = np.asarray(errors).transpose(0,2,1,3) # T,M,Q,N
    changed = np.asarray(changes)
    t,q,n = mask.shape
    m = len(methods)
    num = (error*mask[:,None]).sum(2) # T,M,N
    den = mask.sum(1) # T,N
    primary = num.sum((0,2))/den.sum()
    rng = np.random.default_rng(2026091606)
    b = 4000
    weights = np.zeros((b,n))
    for y in [0,1]:
        for g in [0,1]:
            ids = np.flatnonzero((np.asarray(ix['labels']) == y)&(np.asarray(ix['genders']) == g))
            assert len(ids) == 256
            weights[:,ids] = rng.multinomial(len(ids), np.full(len(ids),1/len(ids)),size=b)
    seed_draw = rng.integers(0,t,(b,t))
    sc = np.stack([(seed_draw==j).sum(1) for j in range(t)],axis=1)
    def weighted(value):
        # T,K,N -> B,K, with common paired weights.
        per = (value.reshape(-1,n) @ weights.T).reshape(t,-1,b).transpose(2,0,1)
        return np.einsum('bt,btk->bk',sc,per)
    db = weighted(den[:,None])[:,0]
    boot = weighted(num)/db[:,None]
    balanced, balanced_boot = [], []
    strata = []
    for c in [False,True]:
        sub = mask&(changed==c)
        dn = sub.sum(1)
        nr = ((~error)*sub[:,None]).sum(2)
        strata.append(dict(source_changed_after_restoration=bool(c),cases=int(dn.sum())))
        assert dn.sum()>0
        balanced.append(nr.sum((0,2))/dn.sum())
        dd = weighted(dn[:,None])[:,0]
        assert np.all(dd>0)
        balanced_boot.append(weighted(nr)/dd[:,None])
    balanced = np.mean(balanced,axis=0)
    bb = np.mean(balanced_boot,axis=0)
    def stat(v,d):
        return dict(value=float(v),ci95=np.quantile(d,[.025,.975]).tolist())
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope=plan['scope'],source_replays_exact=True,
        total_cases=int(t*q*n),accepted_cases=int(den.sum()),coverage=float(den.sum()/(t*q*n)),
        conditional_strata=strata,
        source_early_answer_reference_error=float((mask&changed).sum()/mask.sum()),
        source_early_answer_reference_balanced_agreement=.5,
        summary={name:dict(conditional_error=stat(primary[j],boot[:,j]),
                           conditional_balanced_agreement=stat(balanced[j],bb[:,j]))
                 for j,name in enumerate(methods)},
        comparisons={},
        details=details,inputs=inputs,analyzer=identity(Path(__file__)))
    for other in ['geometry_gain','geometry','raw']:
        i,j=methods.index('native'),methods.index(other)
        out['comparisons']['native_minus_'+other]=dict(
            error_points=stat(100*(primary[i]-primary[j]),100*(boot[:,i]-boot[:,j])),
            balanced_agreement_points=stat(100*(balanced[i]-balanced[j]),100*(bb[:,i]-bb[:,j])))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k not in ['details','inputs']},indent=2))

if __name__ == '__main__':
    main()
