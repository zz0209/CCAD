"""Source-only Cause/Iso controls with actual sparse native and DBM/DAS fits."""
from __future__ import annotations
import argparse
import json
import traceback
import numpy as np
from pathlib import Path
from run_causalgym_multisite import MultisiteWork,ROOT,write
from run_causalgym_native_participation import code_difference
from run_ravel_source_coverage import semantic_measure

def expand_delta(w,ids,delta):
    """Scatter a compact single-site edit into its actual recipient token."""
    if delta.ndim==3:return delta
    if delta.ndim!=2:raise ValueError('Expected single-site or full-position update')
    torch=w.torch
    positions=torch.as_tensor(w.semantic_positions[ids],device=w.device)
    out=torch.zeros((len(ids),w.max_length,w.dim),device=w.device,dtype=delta.dtype)
    out[torch.arange(len(ids),device=w.device),positions]=delta
    return out

def label_loss(w,ids,lp,control):
    torch=w.torch;terms=[];weights=[]
    for j,i in enumerate(ids):
        row=w.panel[i];a=w.cfg['tasks'].index(row['task']);cause=bool(control[a]>.5)
        accepted=row['donor_expected_ids'] if cause else row['expected_ids']
        terms.append(-torch.logsumexp(lp[j,accepted],dim=0));weights.append(1.5 if cause else .75)
    return (torch.stack(terms)*torch.as_tensor(weights,device=w.device)).mean()

def fit_operator(w,operator,x,decoder,kind):
    torch=w.torch;cfg=w.cfg;ids=np.arange(len(w.panel))
    fit=ids[[r['split']=='fit' for r in w.panel]];cal=ids[[r['split']=='calibration' for r in w.panel]]
    controls=torch.eye(3,device=w.device);trace=[]
    def objective(local,control):
        live,_=w.forward(local,expand_delta(w,local,operator(x[local],decoder,control)),differentiable=True)
        return label_loss(w,local,live,control)
    def calibration(step):
        operator.eval();values=[];accuracies=[]
        with torch.no_grad():
            for control in controls:
                losses=[];cause_scores=[];iso_scores=[]
                for jj in w.batches(cal):
                    lp,_=w.forward(jj,expand_delta(w,jj,operator(x[jj],decoder,control)),differentiable=True)
                    losses.append(float(label_loss(w,jj,lp,control))*len(jj))
                    prediction=lp.argmax(1).cpu().tolist()
                    for i,top in zip(jj,prediction):
                        row=w.panel[i];a=cfg['tasks'].index(row['task']);cause=bool(control[a]>.5)
                        accepted=row['donor_expected_ids'] if cause else row['expected_ids']
                        (cause_scores if cause else iso_scores).append(top in accepted)
                values.append(sum(losses)/len(cal))
                accuracies.append(dict(cause=float(np.mean(cause_scores)),iso=float(np.mean(iso_scores))))
        score=float(np.mean([(a['cause']+a['iso'])/2 for a in accuracies]))
        row=dict(step=step,loss=float(np.mean(values)),per_control_loss=values,disentangle_score=score,per_control_accuracy=accuracies)
        trace.append(row)
        torch.save({k:v.detach().cpu() for k,v in operator.state_dict().items()},w.run/(kind+f'_checkpoint_{step}.pt'))
        operator.train()
        # The metric and the tie break are declared before observing new runs.
        return (-score,row['loss']) if cfg.get('selection_metric')=='disentangle' else (row['loss'],0.)
    if kind.startswith('native'):
        init_fit=fit
        if cfg.get('initialization_rows') and len(fit)>cfg['initialization_rows']:
            init_fit=np.sort(np.random.default_rng(cfg['optimizer_seed']).choice(fit,cfg['initialization_rows'],replace=False))
        gains=torch.zeros_like(operator.gates)
        for control in controls:
            for jj in w.batches(init_fit):
                gains-=torch.autograd.grad(objective(jj,control),operator.gates)[0]*(len(jj)/len(init_fit)/3)
        with torch.no_grad():
            operator.gates.copy_(gains.clamp_min(0)/gains.clamp_min(0).amax(dim=0,keepdim=True).clamp_min(1e-20))
            operator.project()
        np.savez_compressed(w.run/(kind+'_initial.npz'),gains=gains.cpu().numpy(),gates=operator.gates.detach().cpu().numpy(),row_ids=init_fit)
        w.progress('SEMANTIC_SOURCE_INITIALIZED',method=kind,initialization_rows=len(init_fit),fit_rows=len(fit))
    best_loss=calibration(0);best_step=0
    best={k:v.detach().cpu().clone() for k,v in operator.state_dict().items()}
    lr=cfg['learning_rates'][kind.split('_')[0]]
    opt=torch.optim.AdamW(operator.parameters(),lr=lr,weight_decay=0)
    rng=np.random.default_rng(cfg['optimizer_seed']);order=list(w.batches(rng.permutation(fit)))
    for step in range(1,cfg['response_steps']+1):
        if step>1 and (step-1)%len(order)==0:order=list(w.batches(rng.permutation(fit)))
        jj=order[(step-1)%len(order)];control=controls[(step-1)%3]
        if kind in ['mdbm','sae_mdbm']:
            start_temp,end_temp=cfg.get('mask_temperatures',{}).get(kind,[.01,1e-7])
            operator.temperature=start_temp+(end_temp-start_temp)*(step-1)/max(1,cfg['response_steps']-1)
        opt.zero_grad(set_to_none=True);loss=objective(jj,control)
        if not torch.isfinite(loss):raise ValueError('Nonfinite source fit')
        loss.backward();opt.step()
        if kind.startswith('native'):operator.project()
        if step in cfg['calibration_steps']:
            current=calibration(step)
            if current<best_loss:
                best_loss,best_step=current,step;best={k:v.detach().cpu().clone() for k,v in operator.state_dict().items()}
            write(w.run/(kind+'_fit_progress.json'),dict(trace=trace,best_step=best_step,best_loss=best_loss))
            w.progress('SEMANTIC_SOURCE_FIT',method=kind,step=step,selection_key=current,
                calibration_ce=trace[-1]['loss'],calibration_disentangle=trace[-1]['disentangle_score'],best_step=best_step)
    operator.load_state_dict(best);operator.eval()
    torch.save(best,w.run/(kind+'_state.pt'))
    chosen=next(r for r in trace if r['step']==best_step)
    meta=dict(kind=kind,trace=trace,selected_step=best_step,selection_key=best_loss,selected_loss=chosen['loss'],
        selected_disentangle=chosen['disentangle_score'],selection_metric=cfg.get('selection_metric','label_ce'),updates=cfg['response_steps'],lr=lr,
        parameter_count=sum(p.numel() for p in operator.parameters()),supervision='Source original training Cause/Iso label aliases; Cause one half and mean Iso one half, single controls only.',
        adaptation='Local RAVEL operator/objective adaptation to Pythia1B first-token endpoint, disjoint paired city panel and shared positions. Not a full original benchmark replication.')
    if kind.startswith('native'):
        g=operator.gates.detach().cpu().numpy();np.savez_compressed(w.run/(kind+'_gates.npz'),gates=g)
        meta.update(active_weights=int((g>0).sum()),active_members=int((g>0).any(1).sum()),
                    weights_per_control=(g>0).sum(0).tolist(),overlap_members=int(((g>0).sum(1)>1).sum()))
        w.checks[kind+'_bounded_budget']=bool(g.min()>=0 and g.max()<=1 and (g>0).sum()<=cfg['weight_budget'])
    elif kind in ['mdbm','sae_mdbm']:
        g=(operator.mask.detach().cpu().numpy()>0);meta.update(binary_weights=int(g.sum()),weights_per_control=g.sum(0).tolist(),
            active_members=int(g.any(1).sum()),overlap_members=int((g.sum(1)>1).sum()),selected_count_constraint='None; actual binary support reported')
        np.savez_compressed(w.run/(kind+'_gates.npz'),gates=g.astype(np.float32))
        meta['linear_temperature_schedule']=cfg.get('mask_temperatures',{}).get(kind,[.01,1e-7])
    else:
        errors=[float((r.weight@r.weight.T-torch.eye(cfg['das_rank'],device=w.device)).detach().abs().max()) for r in operator.rotations]
        meta.update(rank_per_control=cfg['das_rank'],orthogonality_errors=errors)
        w.checks[kind+'_orthogonal']=max(errors)<1e-4
    write(w.run/(kind+'_fit.json'),meta)
    return operator

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args();cfg=json.loads(args.config.read_text())
    sources=['scripts/run_ravel_semantic_source.py','scripts/run_ravel_source_coverage.py','scripts/run_causalgym_multisite.py',
        'scripts/run_causalgym_native_participation.py','scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/semantic_participation.py','src/ccad/ravel_controls.py','src/ccad/semantic_binary_mask.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,sources);error=None
    try:
        w.setup();torch=w.torch
        from ccad.ravel_controls import SemanticNative,MultiDBM,MultiDAS
        from ccad.semantic_binary_mask import SemanticBinaryMask
        ids=np.arange(len(w.panel));evaluation=ids[[r['split']=='held_component_development' for r in w.panel]]
        mode=cfg['mode'];raw,valid,alignments=w.delta(ids,mode,w.hidden)
        if not valid.all():raise ValueError('Missing source coverage')
        compact=cfg.get('compact_single_site',False)
        if compact:
            if any(len(a)!=1 for a in alignments):raise ValueError('Compact material requires exactly one aligned site')
            w.semantic_positions=np.asarray([a[0][0] for a in alignments])
            donor_positions=np.asarray([a[0][1] for a in alignments])
            if not np.array_equal(w.semantic_positions[w.donors],donor_positions):raise ValueError('Donor compact positions differ from alignment')
            raw=raw[ids,w.semantic_positions]
        xr=torch.as_tensor(raw,device=w.device);sae=None;xs=None;decoder=None
        controls=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
        for kind in cfg['methods']:
            torch.manual_seed(cfg['optimizer_seed'])
            if kind.startswith('native') or kind=='sae_mdbm':
                if sae is None:
                    sae=w.load_sae(cfg['source_seed'],w.semantic_positions if compact else None)
                    if compact:dz=sae['codes'][w.donors]-sae['codes']
                    else:dz,positions=code_difference(w,ids,sae['codes'],mode)
                    xs=torch.as_tensor(dz,device=w.device);decoder=torch.as_tensor(sae['decoder'],device=w.device)
                    del dz
                operator=(SemanticBinaryMask(sae['codes'].shape[-1],3) if kind=='sae_mdbm' else
                          SemanticNative(sae['codes'].shape[-1],3,cfg['weight_budget'],kind=='native_exclusive'))
                x,dec=xs,decoder
            elif kind=='mdbm':operator=MultiDBM(w.dim,3);x,dec=xr,None
            elif kind=='mdas':operator=MultiDAS(w.dim,3,cfg['das_rank']);x,dec=xr,None
            else:raise ValueError(kind)
            operator=fit_operator(w,operator.to(w.device),x,dec,kind)
            references={}
            for control in controls:
                with torch.no_grad():delta=expand_delta(w,evaluation,operator(x[evaluation],dec,torch.as_tensor(control,device=w.device))).cpu().numpy()
                name=''.join(map(str,control))
                references[name]=semantic_measure(w,evaluation,delta,kind,mode,control,seed=cfg['source_seed'] if kind.startswith('native') or kind=='sae_mdbm' else 0,
                    control_seen_in_fit=sum(control)==1)
            np.savez_compressed(w.run/(kind+'_held_outputs.npz'),row_ids=evaluation,**{k:v.astype(np.float32) for k,v in references.items()})
            w.progress('SEMANTIC_SOURCE_EVALUATED',method=kind)
            del operator
        groups={}
        for r in w.metrics:groups.setdefault((r['method'],r['operation'],r['task'],r['endpoint']),[]).append(r)
        cells=[dict(method=k[0],control=k[1],task=k[2],endpoint=k[3],n=len(rr),
            first_token_correct=float(np.mean([r['first_token_correct'] for r in rr])),
            expected_probability=float(np.mean([r['expected_probability'] for r in rr]))) for k,rr in groups.items()]
        write(w.run/'semantic_summary.json',dict(cells=cells))
    except Exception:error=traceback.format_exc()
    return w.finish(error)

if __name__=='__main__':raise SystemExit(main())
