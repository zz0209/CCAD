"""Declared, reproducible selection policies using only calibration prefixes.

Costs are explicit forward-equivalent accounting, not measured wall time. The
research harness materializes all outcomes; each policy consumes only its log.
"""
import numpy as np

def budget_select(names,actual,predicted,scores,budgets,writer_forward_ratio=0.,source_rows=None):
    sigmoid=lambda x:1/(1+np.exp(-np.clip(x,-80,80)))
    C,N=actual.shape;assert predicted.shape==(C,N)
    result=[]
    def pick(indices,n):
        means=sigmoid(actual[indices,:n]).mean(1)
        return int(indices[np.argmax(means)])
    for budget in budgets:
        n=min(N,budget//C)
        if n:
            j=pick(np.arange(C),n);result.append(dict(policy='direct_all',budget=budget,selected=names[j],source_backward_rows=0,target_rows=n*C,consumed_forward_equivalents=n*C,calibration_indices=list(range(n))))
        if budget>=C:
            # Spend the integer remainder as well. The recipient order is
            # fixed without inspecting any candidate outcome.
            allocated=np.full(C,min(N,budget//C),dtype=int)
            order=np.random.default_rng(420921).permutation(C);remaining=budget-int(allocated.sum())
            for j in order:
                if remaining and allocated[j]<N:allocated[j]+=1;remaining-=1
            utility=np.array([sigmoid(actual[j,:allocated[j]]).mean() for j in range(C)]);j=int(np.argmax(utility))
            spent=int(allocated.sum());result.append(dict(policy='direct_balanced',budget=budget,selected=names[j],source_backward_rows=0,target_rows=spent,candidate_vector_rows=spent,consumed_native_trial_equivalents=spent,writer_forward_ratio=writer_forward_ratio,consumed_forward_equivalents=spent*(1+writer_forward_ratio),candidate_calibration_indices={names[k]:list(range(int(v))) for k,v in enumerate(allocated)},allocation='Balanced counts differ by at most one; fixed outcome-independent permutation allocates remainder.'))
            # Direct successive halving divides the remaining allowance over
            # its remaining rungs. Survivors receive longer prefixes; the last
            # pair can spend the full residual budget before the final choice.
            allocated=np.zeros(C,dtype=int);active=np.arange(C);used=0;rungs=[];rng=np.random.default_rng(420922)
            while used<budget:
                rounds=max(1,int(np.ceil(np.log2(len(active)))))
                allowance=min(budget-used,max(len(active),(budget-used)//rounds))
                recipients=rng.permutation(active);spent=0
                while spent<allowance and any(allocated[k]<N for k in active):
                    for k in recipients:
                        if spent<allowance and allocated[k]<N:allocated[k]+=1;used+=1;spent+=1
                rungs.append(dict(active=[names[k] for k in active],used=used,prefixes={names[k]:int(allocated[k]) for k in active}))
                utility=np.array([sigmoid(actual[k,:allocated[k]]).mean() for k in active]);ranked=active[np.argsort(-utility,kind='stable')]
                if used>=budget or len(active)<=2 or all(allocated[k]>=N for k in active):break
                active=ranked[:max(2,(len(active)+1)//2)]
            utility=np.array([sigmoid(actual[k,:allocated[k]]).mean() for k in active]);j=int(active[np.argmax(utility)])
            result.append(dict(policy='direct_halving',budget=budget,selected=names[j],source_backward_rows=0,target_rows=used,candidate_vector_rows=used,consumed_native_trial_equivalents=used,writer_forward_ratio=writer_forward_ratio,consumed_forward_equivalents=used*(1+writer_forward_ratio),candidate_calibration_indices={names[k]:list(range(int(v))) for k,v in enumerate(allocated)},rungs=rungs,allocation='Direct successive halving; divide remaining budget over remaining rungs, retain stronger half, allocate the final allowance before choosing between the last pair. Fixed random order allocates integer remainders.'))
        for policy in ['source_screen','cosine_screen','pw_screen','natural_screen']:
            source_n=(source_rows if source_rows is not None else min(4,max(1,budget//12))) if policy=='source_screen' else 0
            # One budget unit is a writer+LM trial. A source screen computes
            # every candidate vector on its prefix, and reuses those vectors
            # during refinement. This prevents hidden free native writing.
            ratio=float(writer_forward_ratio);limit=budget*(1+ratio)
            feasible=[n for n in range(1,N+1) if source_n*C*ratio+3*source_n+3*n+3*max(0,n-source_n)*ratio<=limit]
            n=max(feasible,default=0)
            if n<1:continue
            rank=sigmoid(predicted[:,:source_n]).mean(1) if source_n else np.array([scores[{'cosine_screen':'cosine','pw_screen':'pw_mcc','natural_screen':'natural_mse'}[policy]][name] for name in names])
            top=np.argsort(-rank,kind='stable')[:3];j=pick(top,n)
            spent=source_n*C*ratio+3*source_n+n*len(top)+len(top)*max(0,n-source_n)*ratio
            result.append(dict(policy=policy,budget=budget,selected=names[j],shortlist=[names[i] for i in top],source_backward_rows=source_n,target_rows=n*len(top),candidate_vector_rows=source_n*C+len(top)*max(0,n-source_n),consumed_forward_equivalents=spent,consumed_native_trial_equivalents=spent/(1+ratio),writer_forward_ratio=ratio,source_calibration_indices=list(range(source_n)),calibration_indices=list(range(n))))
        for r in result:
            if r['budget']==budget and r['policy']=='direct_all':
                r.update(candidate_vector_rows=r['target_rows'],consumed_native_trial_equivalents=r['target_rows'],writer_forward_ratio=writer_forward_ratio,consumed_forward_equivalents=r['target_rows']*(1+writer_forward_ratio))
    return result
