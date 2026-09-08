"""Source-controlled fuzzy native participation across multiple LM positions.

Actual target donor-code changes, not source-decoder writes, execute the native
matrix. Source grouping is supervised and target matching uses source responses.
Sparse projections and Adam are standard optimization, not solver novelty claims.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
import numpy as np

from run_causalgym_multisite import MultisiteWork, ROOT, write
from ccad.native_participation import project_participation_rows, project_exclusive_rows, feasibility
from ccad.factor_correspondence import ridge, choose_ridge, group_support


def code_difference(w, ids, codes, mode):
    valid,aligned = w.alignment(ids,mode)
    if not valid.all():
        raise ValueError('The chosen common intervention must cover every task row')
    result = np.zeros((len(ids),w.max_length,codes.shape[-1]),np.float32)
    positions = np.zeros((len(ids),w.max_length),bool)
    for j,i in enumerate(ids):
        p,q = np.asarray(aligned[j]).T
        result[j,p] = codes[w.donors[i],q]-codes[i,p]
        positions[j,p] = True
    return result,positions


def decode(w, codes, decoder):
    torch = w.torch
    with torch.no_grad():
        return (torch.as_tensor(codes,device=w.device)@torch.as_tensor(decoder,device=w.device)).cpu().numpy()


def component_membership(n):
    # A fixed interleaving of source-only score ranks, not semantic annotations.
    membership = np.zeros((n,2),np.float32)
    membership[np.arange(n),np.arange(n)%2] = 1
    return membership


def loss_from_response(w, ids, delta, reference=None, supervised=False):
    torch = w.torch
    live,_ = w.forward(ids,delta,differentiable=True)
    if supervised:
        ix = torch.arange(len(ids),device=w.device)
        labels = torch.as_tensor([w.label_ids[w.panel[i]['donor_label']] for i in ids],device=w.device)
        return -live[ix,labels].mean()
    ref = torch.as_tensor(reference,device=w.device,dtype=live.dtype)
    return (torch.exp(ref)*(ref-live)).sum(1).mean()


def fit_response_operator(w, context, kind, initial, inputs=None, supervised=False):
    """Same finite-response updates/calibration schedule for native and readers."""
    torch = w.torch
    cfg = w.cfg
    ids,fit,cal = context['ids'],context['fit'],context['cal']
    task = context['task']
    param = torch.nn.Parameter(torch.as_tensor(initial,dtype=torch.float32,device=w.device).clone())
    native = kind.startswith('native')
    projection = project_exclusive_rows if kind=='native_exclusive' else project_participation_rows
    x = torch.as_tensor(context['dzt'] if native else inputs,dtype=torch.float32,device=w.device)
    decoder = torch.as_tensor(context['td'] if native else context['sd'],dtype=torch.float32,device=w.device)
    membership = torch.as_tensor(context['membership'],device=w.device)
    operations = [('whole',np.ones(1 if supervised else 2,np.float32))] if supervised else context['training_controls']
    trace = []

    def make_delta(local,control):
        m = torch.as_tensor(control,dtype=torch.float32,device=w.device)
        if native:
            return (x[local]*(param@m))@decoder
        return ((x[local]@param)*(membership@m))@decoder

    def objective(local,name,control):
        reference = None if supervised else context['references'][name][local]
        return loss_from_response(w,ids[local],make_delta(local,control),reference,supervised)

    def calibration(step):
        losses = []
        with torch.no_grad():
            for name,m in operations:
                loss = objective(cal,name,m)
                losses.append(float(loss))
        row = dict(step=step,calibration_loss=float(np.mean(losses)),operation_losses=losses,
                   fit_objective='true donor-label CE' if supervised else 'equal-weight whole/part full-output source KL')
        trace.append(row)
        return row['calibration_loss']

    if native:
        # Initial native candidates use source-response gradients on fitting
        # components only; target labels are only used by the named CE control.
        gains = torch.zeros_like(param)
        for name,m in operations:
            for local in w.batches(fit):
                loss = objective(local,name,m)
                gains -= torch.autograd.grad(loss,param)[0]*(len(local)/len(fit)/len(operations))
        with torch.no_grad():
            positive = gains.clamp_min(0)
            ranked = torch.argsort(positive.sum(1),descending=True,stable=True)[:cfg['target_budget']]
            param.zero_()
            param[ranked] = positive[ranked]/positive[ranked].sum(1,keepdim=True).clamp_min(1e-30)
            projection(param,cfg['target_budget'])
        np.savez_compressed(w.run/f'{task}_{kind}_initial.npz',scores=gains.cpu().numpy(),matrix=param.detach().cpu().numpy())
    best_loss = calibration(0)
    best,best_step = param.detach().cpu().numpy().copy(),0
    optimizer = torch.optim.Adam([param],lr=cfg['response_lr'])
    rng = np.random.default_rng(cfg['optimizer_seed'])
    order = rng.permutation(fit)
    batches = list(w.batches(order))
    for step in range(1,cfg['response_steps']+1):
        if (step-1)%len(batches)==0 and step>1:
            batches = list(w.batches(rng.permutation(fit)))
        local = batches[(step-1)%len(batches)]
        name,m = operations[(step-1)%len(operations)]
        optimizer.zero_grad(set_to_none=True)
        loss = objective(local,name,m)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite finite-response loss')
        loss.backward()
        optimizer.step()
        if native:
            projection(param,cfg['target_budget'])
        if step in cfg['calibration_steps']:
            value = calibration(step)
            if value<best_loss:
                best_loss,best,best_step = value,param.detach().cpu().numpy().copy(),step
            write(w.run/f'{task}_{kind}_fit_progress.json',dict(trace=trace,best_step=best_step,best_loss=best_loss))
            w.progress('RESPONSE_OPTIMIZATION',task=task,kind=kind,step=step,best_step=best_step,calibration_loss=value)
    meta = dict(kind=kind,task=task,selected_step=best_step,selected_calibration_loss=best_loss,
                updates=cfg['response_steps'],lr=cfg['response_lr'],parameter_shape=list(best.shape),trace=trace,
                target_labels_accessed=supervised,source_full_distribution_accessed=not supervised,
                information='Source components/outputs are supervised teacher inputs, not a label-free or endpoint-free matcher')
    if native:
        meta['feasibility'] = feasibility(best)
        w.checks[task+'_'+kind+'_feasible'] = all(meta['feasibility'][k] for k in ['nonnegative','row_sum_at_most_one']) and meta['feasibility']['actual_members']<=cfg['target_budget']
    write(w.run/f'{task}_{kind}_fit.json',meta)
    np.savez_compressed(w.run/f'{task}_{kind}_matrix.npz',matrix=best)
    w.progress('RESPONSE_OPERATOR_FIT',task=task,kind=kind,selected_step=best_step,calibration_loss=best_loss)
    return best,meta


def initial_readers(w, context):
    """Shared positionwise predictors; all later get identical KL update counts."""
    cfg = w.cfg
    positions = context['positions']
    rr,pp = np.where(positions)
    counts = positions.sum(1)
    weight = 1/np.sqrt(counts[rr])
    fit = np.isin(rr,context['fit'])
    cal = np.isin(rr,context['cal'])
    y = context['dzs'][rr,pp][:,context['support']].astype(np.float64)
    yweighted = y*weight[:,None]
    all_target = context['dzt'][rr,pp].astype(np.float64)
    xweighted = all_target*weight[:,None]
    alphas = cfg['ridge_alphas']
    alpha,errors = choose_ridge(xweighted,yweighted,fit,cal,alphas,'native_units')
    full = ridge(xweighted[fit],yweighted[fit],alpha,'native_units')
    selected,selection = group_support(xweighted[fit],yweighted[fit],cfg['target_budget'],iterations=cfg['lasso_steps'],scaling='native_units')
    dense = np.argsort(-np.sqrt(np.mean(xweighted[fit]**2,axis=0))*np.linalg.norm(full,axis=1),kind='stable')[:cfg['target_budget']]
    readers = {'full':(full,context['dzt'],dict(alpha=alpha,calibration_mse=errors,actual_read_members=context['dzt'].shape[-1]))}
    for name,members in [('compact',selected),('dense',dense)]:
        xx = xweighted[:,members]
        a,err = choose_ridge(xx,yweighted,fit,cal,alphas,'native_units')
        matrix = ridge(xx[fit],yweighted[fit],a,'native_units')
        readers[name] = (matrix,context['dzt'][...,members],dict(alpha=a,calibration_mse=err,actual_read_members=len(members),members=members.tolist(),selection=selection if name=='compact' else 'RMS times full coefficient row norm'))
    xraw = context['raw'][rr,pp].astype(np.float64)*weight[:,None]
    a,err = choose_ridge(xraw,yweighted,fit,cal,alphas,'native_units')
    matrix = ridge(xraw[fit],yweighted[fit],a,'native_units')
    readers['raw'] = (matrix,context['raw'],dict(alpha=a,calibration_mse=err,read_dimensions=xraw.shape[-1]))
    write(w.run/f"{context['task']}_reader_initialization.json",dict(methods={k:v[2] for k,v in readers.items()},
        source_supervision='Source code-component differences; code MSE initialization, followed by same finite full-output KL training as native participation',
        position_weight='Each prompt has total squared position weight1; models shared across all edited positions'))
    return readers


def native_baselines(w, context):
    from scipy.optimize import linear_sum_assignment
    cfg = w.cfg
    ns,nt = len(context['support']),context['dzt'].shape[-1]
    part = np.argmax(context['membership'],axis=1)
    matrices = {}
    if hasattr(w,'global_assignment'):
        assignment = w.global_assignment
    else:
        assignment_path=ROOT/cfg['global_assignment']
        parent_cfg=json.loads(w.checked(assignment_path.parent/'config.resolved.json').read_text())
        if parent_cfg['seed_pairs']!=[[cfg['source_seed'],cfg['target_seed']]] or parent_cfg['sae_root']!=cfg['sae_root'] or parent_cfg['layer']!=cfg['layer']:
            raise ValueError('Global assignment is not for the exact current SAE pair and layer')
        assignment = np.load(w.checked(assignment_path,'Existing complete-dictionary normalized absolute decoder-cosine Hungarian control'))['target_indices']
    if len(np.unique(assignment))!=nt or assignment.min()<0 or assignment.max()>=nt:
        raise ValueError('Global dictionary assignment is not a bijection')
    mapped = assignment[context['support']]
    g = np.zeros((nt,2),np.float32)
    g[mapped,part] = 1
    matrices['global_hungarian_native'] = g
    rr,pp = np.where(context['positions'])
    use = np.isin(rr,context['fit'])
    xs = context['dzs'][rr[use],pp[use]][:,context['support']].astype(np.float64)
    xt = context['dzt'][rr[use],pp[use]].astype(np.float64)
    xs-=xs.mean(0);xt-=xt.mean(0)
    corr=(xs.T@xt)/(np.maximum(np.linalg.norm(xs,axis=0)[:,None]*np.linalg.norm(xt,axis=0)[None,:],1e-30))
    si,tj=linear_sum_assignment(-np.abs(corr))
    g=np.zeros((nt,2),np.float32);g[tj,part[si]]=1
    matrices['task_activation_assignment_native'] = g
    eligible=np.flatnonzero(np.any(context['dzt'][context['fit']]!=0,axis=(0,1)))
    rng=np.random.default_rng(cfg['optimizer_seed']+97)
    random_members=rng.choice(eligible,size=min(cfg['target_budget'],len(eligible)),replace=False)
    g=np.zeros((nt,2),np.float32);g[random_members,np.arange(len(random_members))%2]=1
    matrices['random_native'] = g
    for name,g in matrices.items():
        np.savez_compressed(w.run/f"{context['task']}_{name}.npz",matrix=g)
    write(w.run/f"{context['task']}_native_baselines.json",dict(
        global_assignment='Full8192 decoder geometry assignment, not PW-MCC',
        activation_assignment='Task-conditional Pearson across aligned training positions, not natural-corpus PW-MCC or SemanticOT',
        actual={k:feasibility(v) for k,v in matrices.items()},source_parts='Even and odd source-only score ranks, fixed before target fitting'))
    return matrices


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    sources=['scripts/run_causalgym_native_participation.py','scripts/run_causalgym_multisite.py',
             'scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py',
             'src/ccad/native_participation.py','src/ccad/activation_contract.py','src/ccad/artifacts.py',
             'src/ccad/factor_correspondence.py','src/ccad/native_operation.py','src/ccad/nip_baselines.py',
             'scripts/prepare_f4_global_matching.py']
    w=MultisiteWork(cfg,args.config,sources)
    error=None
    contexts={}
    try:
        w.setup()
        torch=w.torch
        ss,ts=cfg['source_seed'],cfg['target_seed']
        source,target=w.load_sae(ss),w.load_sae(ts)
        if cfg.get('compute_global_assignment',False):
            from prepare_f4_global_matching import full_assignment
            mapping,scale,diagnostic=full_assignment(source['decoder'],target['decoder'])
            w.global_assignment=mapping
            np.savez_compressed(w.run/f'global_assignment_s{ss}_t{ts}.npz',target_indices=mapping,geometric_scale=scale)
            write(w.run/'global_assignment_fit.json',diagnostic)
            w.progress('GLOBAL_DICTIONARY_ASSIGNED',source_seed=ss,target_seed=ts)
        for task in cfg['tasks']:
            ids=np.asarray([i for i,r in enumerate(w.panel) if r['task']==task])
            fit=np.asarray([j for j,i in enumerate(ids) if w.panel[i]['split']=='fit'])
            cal=np.asarray([j for j,i in enumerate(ids) if w.panel[i]['split']=='calibration'])
            evaluation=np.asarray([j for j,i in enumerate(ids) if w.panel[i]['split']=='held_component_development'])
            mode=cfg['mode']
            dzs,positions=code_difference(w,ids,source['codes'],mode)
            dzt,tpositions=code_difference(w,ids,target['codes'],mode)
            assert np.array_equal(positions,tpositions)
            raw,valid,_=w.delta(ids,mode,w.hidden)
            assert valid.all()
            full_source=decode(w,dzs,source['decoder'])
            w.measure(ids,mode,'raw_donor',raw,seed=0,operation='whole')
            w.measure(ids,mode,'full_source_sae',full_source,seed=ss,operation='whole')
            w.measure(ids,mode,'full_target_sae',decode(w,dzt,target['decoder']),seed=ts,operation='whole')
            mean_gradient=np.zeros_like(full_source[fit],np.float64)
            for alpha in cfg['ig_points']:
                for local in w.batches(np.arange(len(fit))):
                    _,gradient=w.forward(ids[fit[local]],alpha*full_source[fit[local]],gradient=True)
                    mean_gradient[local]+=gradient/len(cfg['ig_points'])
            with torch.no_grad():
                sensitivity=(torch.as_tensor(mean_gradient,dtype=torch.float32,device=w.device)@
                             torch.as_tensor(source['decoder'].T,device=w.device)).cpu().numpy()
            attribution=np.sum(sensitivity*dzs[fit],axis=1,dtype=np.float64)
            scores=attribution.mean(0)
            absolute=np.abs(attribution).mean(0)
            support=np.argsort(-scores,kind='stable')[:cfg['source_budget']]
            support=support[scores[support]>0]
            if not len(support):
                raise ValueError('No positive source-only causal attribution; keep failed task rather than pick by target')
            np.savez_compressed(w.run/f'{task}_source_selection.npz',source_support=support,signed_scores=scores,
                mean_absolute_scores=absolute,attribution=attribution,fit_row_ids=ids[fit],mean_gradient=mean_gradient)
            # Cheap source comparisons answer whether signed cross-position
            # grouping changes the source deficiency, without target selection.
            for selection,score in [('signed',scores),('absolute',absolute)]:
                ordered=np.argsort(-score,kind='stable')
                for budget in cfg['source_curve_budgets']:
                    keep=ordered[:budget]
                    delta=decode(w,dzs[evaluation][...,keep],source['decoder'][keep])
                    w.measure(ids[evaluation],mode,f'source_{selection}_{budget}',delta,seed=ss,operation='whole',actual_source_members=len(keep))
            membership=component_membership(len(support))
            sd=source['decoder'][support]
            context=dict(task=task,ids=ids,fit=fit,cal=cal,evaluation=evaluation,positions=positions,
                         dzs=dzs,dzt=dzt,base_target=target['codes'][ids],sd=sd,td=target['decoder'],raw=raw,support=support,membership=membership,
                         training_controls=[('whole',np.array([1,1],np.float32)),('part0',np.array([1,0],np.float32)),('part1',np.array([0,1],np.float32))],
                         test_controls=[('half_dose',np.array([.5,.5],np.float32)),('mixed_dose',np.array([.25,1],np.float32))],references={})
            for name,m in context['training_controls']+context['test_controls']:
                native=decode(w,dzs[...,support]*(membership@m),sd)
                context['references'][name]=w.measure(ids,mode,'source_selected',native,seed=ss,operation=name,
                    actual_source_members=len(support),operation_seen_by_target_fit=name in ['whole','part0','part1'])
            write(w.run/f'{task}_source_definition.json',dict(source_seed=ss,members=support.tolist(),
                membership=membership.tolist(),source_only=True,selection='Mean signed integrated-gradient contribution summed over aligned positions on fit components',
                source_labels_used=True,parts='Alternating IG score ranks; reproducible experimental components, not human semantic labels',
                fit_row_ids=ids[fit].tolist(),calibration_row_ids=ids[cal].tolist(),held_development_row_ids=ids[evaluation].tolist()))
            context['native_controls']=native_baselines(w,context)
            initial=np.zeros((dzt.shape[-1],2),np.float32)
            g,gmeta=fit_response_operator(w,context,'native_participation',initial)
            context['matrix'],context['matrix_meta']=g,gmeta
            if cfg.get('exclusive_control',False):
                # This baseline independently selects and optimizes its rows
                # with the same source supervision, update count and schedule.
                exclusive,emeta=fit_response_operator(w,context,'native_exclusive',np.zeros_like(initial))
                context['native_controls']['native_exclusive']=exclusive
            supervised,smeta=fit_response_operator(w,context,'native_target_supervision',np.zeros((dzt.shape[-1],1),np.float32),supervised=True)
            context['supervised']=supervised
            readers=initial_readers(w,context)
            context['readers']={}
            for name,(initial,x,meta) in readers.items():
                matrix,fitmeta=fit_response_operator(w,context,name+'_readout',initial,inputs=x)
                context['readers'][name]=(matrix,x,meta,fitmeta,initial)
            contexts[task]=context
            w.progress('TASK_FITTED',task=task,source_members=len(support))
        # All task matrices exist before wrong-task controls are evaluated.
        for ti,task in enumerate(cfg['tasks']):
            c=contexts[task]
            ev,ids=c['evaluation'],c['ids'][c['evaluation']]
            wrong=contexts[cfg['tasks'][(ti+1)%len(cfg['tasks'])]]['matrix']
            matrices=dict(native_participation=c['matrix'],wrong_task_participation=wrong,**c['native_controls'])
            for operation,m in c['training_controls']+c['test_controls']:
                reference=c['references'][operation][ev]
                common=dict(target_seed=ts,operation=operation,reference_identity='source_selected_'+operation,
                            operation_seen_by_target_fit=operation in ['whole','part0','part1'])
                for name,g in matrices.items():
                    code_change=c['dzt'][ev]*(g@m)
                    native=decode(w,code_change,c['td'])
                    state_min=float((c['base_target'][ev]+code_change).min())
                    w.checks[task+'_'+name+'_'+operation+'_actual_state']=state_min>=-1e-5
                    info=feasibility(g)
                    w.measure(ids,cfg['mode'],name,native,seed=ss,reference=reference,
                              actual_target_members=info['actual_members'],shared_target_members=info['shared_members'],final_code_min=state_min,**common)
                for name,(matrix,x,meta,fitmeta,initial) in c['readers'].items():
                    for variant,weights in [('response_fit',matrix),('ridge',initial)]:
                        predicted=x[ev]@weights
                        delta=decode(w,(predicted*(c['membership']@m)).astype(np.float32),c['sd'])
                        w.measure(ids,cfg['mode'],name+'_'+variant,delta,seed=ss,reference=reference,
                                  actual_read_members=meta.get('actual_read_members',meta.get('read_dimensions')),native=False,**common)
                if operation=='whole':
                    native=decode(w,c['dzt'][ev]*c['supervised'].sum(1),c['td'])
                    w.measure(ids,cfg['mode'],'native_direct_target_supervision',native,seed=ss,reference=reference,
                              actual_target_members=feasibility(c['supervised'])['actual_members'],target_labels_used=True,**common)
            np.savez_compressed(w.run/f'{task}_participation_export.npz',source_members=c['support'],
                                source_components=c['membership'],target_participation=c['matrix'])
            w.progress('TASK_EVALUATED',task=task)
        w.checks['all_tasks_fitted']=len(contexts)==len(cfg['tasks'])
        del contexts,source,target
        gc.collect()
    except Exception:
        error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':
    raise SystemExit(main())
