"""Activation-weighted shared-context retrieval (SemanticOT Algorithm 1).

Implements the published distribution/centroid/Sinkhorn recipe using the
project's existing balanced log-Sinkhorn kernel. This is not circuit compression
or a reproduction of the paper's corpus/model scale and human evaluation.
"""
import numpy as np
from .nip_baselines import _balanced_log_sinkhorn


def top_distributions(indices, activations, width, top_k, reference):
    """Deterministic positive top contexts, linear normalized activation weights."""
    ids = np.asarray(indices).reshape(-1)
    values = np.asarray(activations).reshape(-1).astype(np.float64)
    tokens = np.repeat(np.arange(len(indices)), indices.shape[1])
    positive = values > 0
    ids, values, tokens = ids[positive], values[positive], tokens[positive]
    # Primary key feature ID, then descending activation, then token identity.
    order = np.lexsort((tokens, -values, ids))
    ids, values, tokens = ids[order], values[order], tokens[order]
    counts = np.bincount(ids, minlength=width)
    offsets = np.r_[0,np.cumsum(counts)]
    distributions = []
    centroids = np.zeros((width, reference.shape[1]), np.float64)
    for feature in range(width):
        start = offsets[feature]
        stop = start+min(top_k,counts[feature])
        selected = tokens[start:stop]
        mass = values[start:stop]
        if len(mass):
            mass = mass/mass.sum()
            centroids[feature] = mass@reference[selected]
        distributions.append((selected,mass))
    return distributions,centroids,counts


def euclidean_cost(left, right):
    squared = np.sum(left*left,axis=1)[:,None]+np.sum(right*right,axis=1)[None,:]-2*left@right.T
    return np.sqrt(np.maximum(squared,0))


def entropic_dual_plan(cost, left, right, regularization, tolerance):
    """Solve the same entropic OT objective through its one-potential dual.

    This standard smooth convex dual is used only when alternating Sinkhorn
    scaling stalls. Row marginals are exact by construction. The returned plan
    still has to satisfy the original column-marginal tolerance.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    a,b=np.asarray(left),np.asarray(right)
    scaled=np.asarray(cost)/regularization
    initial=a@scaled
    initial=initial-initial[-1]
    def objective(short, return_plan=False):
        potential=np.r_[short,0.]
        scores=potential[None,:]-scaled
        norm=logsumexp(scores,axis=1)
        plan=a[:,None]*np.exp(scores-norm[:,None])
        if return_plan:return plan
        return float(a@norm-b@potential),(plan.sum(axis=0)-b)[:-1]
    if len(b)==1:
        return a[:,None],0,True,0.
    result=minimize(objective,initial[:-1],jac=True,method='L-BFGS-B',
                    options=dict(maxiter=5000,maxls=40,gtol=min(1e-10,tolerance*.01),ftol=1e-15))
    plan=objective(result.x,True)
    error=max(float(np.max(np.abs(plan.sum(axis=1)-a))),float(np.max(np.abs(plan.sum(axis=0)-b))))
    # L-BFGS can stop on objective roundoff just above the marginal tolerance.
    # Diagonal rebalancing stays in the same entropic scaling family. It refines
    # that same plan/objective; it neither changes epsilon nor relaxes tolerance.
    polishing=0
    while error>tolerance and polishing<20000:
        plan*=np.divide(a,plan.sum(axis=1),out=np.ones_like(a),where=plan.sum(axis=1)>0)[:,None]
        plan*=np.divide(b,plan.sum(axis=0),out=np.ones_like(b),where=plan.sum(axis=0)>0)[None,:]
        polishing+=1
        if polishing%20==0:error=max(float(np.max(np.abs(plan.sum(axis=1)-a))),float(np.max(np.abs(plan.sum(axis=0)-b))))
    return plan,int(result.nit)+polishing,error<=tolerance,error


def retrieve(source_ids, source_distributions, target_distributions, source_centroids,
             target_centroids, source_counts, target_counts, reference, *, candidate_count,
             minimum_count, regularization, tolerance, max_iterations, progress=None, dual_fallback=False):
    candidates = np.flatnonzero(target_counts>=minimum_count)
    if len(candidates)<candidate_count:
        raise ValueError('Insufficient natural-reference target candidates')
    rows = []
    for source in source_ids:
        if source_counts[source]<minimum_count:
            row = dict(source_member=int(source),status='INSUFFICIENT_REFERENCE_ACTIVITY',
                       reference_activations=int(source_counts[source]),target_member=None,candidates=[])
            rows.append(row)
            if progress:progress(row)
            continue
        distances = np.linalg.norm(target_centroids[candidates]-source_centroids[source],axis=1)
        order = np.lexsort((candidates,distances))[:candidate_count]
        shortlisted = candidates[order]
        ix,mass = source_distributions[source]
        costs = []
        for target,centroid_distance in zip(shortlisted,distances[order]):
            iy,target_mass = target_distributions[target]
            cost = euclidean_cost(reference[ix].astype(float),reference[iy].astype(float))
            # Identical token positions must have exactly zero metric cost.
            cost[ix[:,None]==iy[None,:]] = 0
            plan,iterations,converged,error = _balanced_log_sinkhorn(cost,mass,target_mass,
                regularization=regularization,tolerance=tolerance,max_iterations=max_iterations)
            solver='balanced_log_sinkhorn';scaling_error=error
            if not converged and dual_fallback:
                plan,dual_steps,converged,error=entropic_dual_plan(cost,mass,target_mass,regularization,tolerance)
                iterations+=dual_steps;solver='sinkhorn_then_same_objective_dual_lbfgs'
            if not converged:
                raise RuntimeError(f'Sinkhorn marginal residual {error} for {source}->{target}')
            costs.append(dict(target_member=int(target),distance=float(np.sum(plan*cost)),
                              centroid_distance=float(centroid_distance),iterations=iterations,marginal_error=error,
                              solver=solver,initial_sinkhorn_error=scaling_error,
                              target_reference_activations=int(target_counts[target]),
                              source_top_contexts=len(ix),target_top_contexts=len(iy)))
        costs.sort(key=lambda r:(r['distance'],r['target_member']))
        row = dict(source_member=int(source),status='MATCHED',reference_activations=int(source_counts[source]),
                   target_member=costs[0]['target_member'],centroid_target_member=int(shortlisted[0]),
                   distance=costs[0]['distance'],candidates=costs)
        rows.append(row)
        if progress:progress(row)
    return rows


class TensorContextIndex:
    """Same top-context/centroid/transport recipe, batched on the active device.

    Tensorization changes no cost or marginal objective. Any unconverged pair
    uses the existing same-objective dual; missing activity stays unmatched.
    """
    def __init__(self,z,reference,top_k=64,minimum_count=30,reference_distance=None):
        import torch
        self.reference=reference;self.minimum_count=minimum_count
        self.distance=reference_distance if reference_distance is not None else torch.cdist(reference.double(),reference.double())
        self.distance.diagonal().zero_()
        self.count=(z>0).sum(0);self.value,self.index=z.topk(top_k,dim=0,sorted=True)
        self.value=self.value.T.double();self.index=self.index.T
        self.mass=self.value/self.value.sum(1,keepdim=True).clamp_min(1e-30)
        self.centroid=torch.zeros((z.shape[1],reference.shape[1]),device=z.device,dtype=reference.dtype)
        for off in range(0,z.shape[1],128):
            self.centroid[off:off+128]=(reference[self.index[off:off+128]]*self.mass[off:off+128,:,None]).sum(1).to(reference.dtype)
        self.cache={}

    def match(self,source,source_ids,*,candidate_count=50,regularization,tolerance=1e-7,max_iterations=512,progress=None):
        import torch
        eligible=torch.where(self.count>=self.minimum_count)[0];assert len(eligible)>=candidate_count
        pending=[];all_targets=[];all_distances=[]
        for s in source_ids:
            s=int(s);key=(id(source),s)
            if key in self.cache:continue
            if int(source.count[s])<source.minimum_count:
                row=dict(source_member=s,status='INSUFFICIENT_REFERENCE_ACTIVITY',target_member=None,reference_activations=int(source.count[s]),candidates=[])
                self.cache[key]=row
                if progress:progress(row)
                continue
            distances=(self.centroid[eligible]-source.centroid[s]).norm(dim=1)
            order=torch.argsort(distances,stable=True)[:candidate_count];targets=eligible[order]
            pending.append(s);all_targets.append(targets);all_distances.append(distances[order])
        if not pending:return [self.cache[id(source),int(s)] for s in source_ids]
        source_flat=torch.tensor(pending,device=self.reference.device).repeat_interleave(candidate_count)
        target_flat=torch.cat(all_targets);distance_flat=torch.cat(all_distances);cost_records=[]
        # Ground distances are shared across every feature pair; batch many
        # transport problems to avoid repeated tiny decoder-space products.
        for off in range(0,len(source_flat),2048):
            sf=source_flat[off:off+2048];tf=target_flat[off:off+2048];ix=source.index[sf];iy=self.index[tf]
            left=source.mass[sf];right=self.mass[tf];C=self.distance[ix[:,:,None],iy[:,None,:]]
            loga=left.log();logb=right.log();K=-C/regularization;u=torch.zeros_like(left);v=torch.zeros_like(right);error=None
            for it in range(max_iterations):
                u=loga-torch.logsumexp(K+v[:,None,:],2)
                v=logb-torch.logsumexp(K+u[:,:,None],1)
                if it%32==31 or it==max_iterations-1:
                    P=(K+u[:,:,None]+v[:,None,:]).exp();error=torch.maximum((P.sum(2)-left).abs().amax(1),(P.sum(1)-right).abs().amax(1))
                    if float(error.max())<=tolerance:break
            assert bool(torch.isfinite(error).all()),'Nonfinite transport residual'
            for j,t in enumerate(tf.cpu().tolist()):
                err=float(error[j]);solver='tensor_log_sinkhorn';steps=it+1
                if err>tolerance:
                    keepa=left[j]>0;keepb=right[j]>0;cost=C[j][keepa][:,keepb].cpu().numpy()
                    plan,extra,ok,err=entropic_dual_plan(cost,left[j][keepa].cpu().numpy(),right[j][keepb].cpu().numpy(),regularization,tolerance)
                    if not ok:raise RuntimeError(f'Context transport marginal residual {err}: {int(sf[j])}->{t}')
                    val=float((plan*cost).sum());solver='tensor_then_same_objective_dual';steps+=extra
                else:val=float((P[j]*C[j]).sum())
                cost_records.append(dict(target_member=t,distance=val,centroid_distance=float(distance_flat[off+j]),marginal_error=err,solver=solver,iterations=steps))
            if progress:progress(dict(status='TRANSPORT_BATCH_COMPLETE',pairs=min(off+2048,len(source_flat)),total_pairs=len(source_flat)))
        for i,s in enumerate(pending):
            costs=cost_records[i*candidate_count:(i+1)*candidate_count]
            costs.sort(key=lambda r:(r['distance'],r['target_member']))
            row=dict(source_member=s,status='MATCHED',target_member=costs[0]['target_member'],reference_activations=int(source.count[s]),distance=costs[0]['distance'],candidates=costs)
            self.cache[id(source),s]=row
            if progress:progress(row)
        return [self.cache[id(source),int(s)] for s in source_ids]
