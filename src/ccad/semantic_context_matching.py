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
    return plan,int(result.nit),error<=tolerance,error


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
