"""Fit matched independent/common attribute families on source data only."""
from __future__ import annotations
import argparse
import json
import traceback
import numpy as np
from pathlib import Path
from run_causalgym_multisite import MultisiteWork,ROOT,write
from run_ravel_semantic_source import expand_delta
from run_ravel_source_coverage import semantic_measure


def balanced_label_loss(w,ids,lp,control):
    torch=w.torch;count=int(control.sum());loss=[]
    for j,i in enumerate(ids):
        row=w.panel[i];a=w.cfg['tasks'].index(row['task']);cause=bool(control[a]>.5)
        labels=row['donor_expected_ids'] if cause else row['expected_ids']
        weight=1. if count in [0,3] else (3/(2*count) if cause else 3/(2*(3-count)))
        loss.append(-torch.logsumexp(lp[j,labels],0)*weight)
    return torch.stack(loss).mean()


def calibration(w,operator,x,ids,controls):
    torch=w.torch;rows=[];operator.eval()
    with torch.no_grad():
        for c in controls:
            control=torch.as_tensor(c,device=w.device);ca=[];iso=[];loss=[]
            for jj in w.batches(ids):
                with torch.nn.utils.parametrize.cached():q=operator(x[jj],None,control)
                lp,_=w.forward(jj,expand_delta(w,jj,q),differentiable=True)
                loss.append(float(balanced_label_loss(w,jj,lp,control))*len(jj))
                for i,pred in zip(jj,lp.argmax(1).cpu().tolist()):
                    row=w.panel[i];cause=bool(control[w.cfg['tasks'].index(row['task'])]>.5)
                    (ca if cause else iso).append(pred in (row['donor_expected_ids'] if cause else row['expected_ids']))
            cmean=float(np.mean(ca)) if ca else None;imean=float(np.mean(iso)) if iso else None
            score=float(np.mean([a for a in [cmean,imean] if a is not None]))
            rows.append(dict(control=list(c),cause=cmean,iso=imean,score=score,loss=sum(loss)/len(ids)))
    single=float(np.mean([r['score'] for r in rows if sum(r['control'])==1]));family=float(np.mean([r['score'] for r in rows]));ce=float(np.mean([r['loss'] for r in rows]))
    return dict(single_score=single,family_score=family,loss=ce,cells=rows)


def fit(w,operator,x,kind):
    torch=w.torch;cfg=w.cfg;ids=np.arange(len(w.panel));fitids=ids[[r['split']=='fit' for r in w.panel]];calids=ids[[r['split']=='calibration' for r in w.panel]]
    controls=cfg['binary_controls'];train_controls=cfg.get('training_controls',controls);trace=[];best=None;beststep=None;bestkey=None
    def evaluate(step):
        nonlocal best,beststep,bestkey
        stats=calibration(w,operator,x,calids,controls);stats['step']=step;trace.append(stats)
        key=(-stats['single_score'],-stats['family_score'],stats['loss']) if cfg.get('selection_rule')=='single' else (-stats['family_score'],-stats['single_score'],stats['loss'])
        state={k:v.detach().cpu().clone() for k,v in operator.state_dict().items()}
        torch.save(state,w.run/f'{kind}_checkpoint_{step}.pt')
        if bestkey is None or key<bestkey:best,beststep,bestkey=state,step,key
        write(w.run/f'{kind}_fit_progress.json',dict(trace=trace,selected_step=beststep,selection_key=bestkey))
        w.progress('SEMANTIC_FAMILY_FIT',method=kind,step=step,calibration_family=stats['family_score'],calibration_single=stats['single_score'],selected_step=beststep)
    evaluate(0);opt=torch.optim.AdamW(operator.parameters(),lr=cfg['response_lr'],weight_decay=0)
    rng=np.random.default_rng(cfg['optimizer_seed']);order=list(w.batches(rng.permutation(fitids)))
    for step in range(1,cfg['response_steps']+1):
        if step>1 and (step-1)%len(order)==0:order=list(w.batches(rng.permutation(fitids)))
        jj=order[(step-1)%len(order)];control=torch.as_tensor(train_controls[(step-1)%len(train_controls)],device=w.device)
        operator.train();opt.zero_grad(set_to_none=True)
        with torch.nn.utils.parametrize.cached():q=operator(x[jj],None,control)
        lp,_=w.forward(jj,expand_delta(w,jj,q),differentiable=True);loss=balanced_label_loss(w,jj,lp,control)
        if not torch.isfinite(loss):raise ValueError('Nonfinite family objective')
        loss.backward();opt.step()
        if step in cfg['calibration_steps']:evaluate(step)
    operator.load_state_dict(best);operator.eval();torch.save(best,w.run/f'{kind}_state.pt')
    result=dict(method=kind,selected_step=beststep,selection_key=bestkey,trace=trace,selected=next(r for r in trace if r['step']==beststep),
        parameter_count=sum(p.numel() for p in operator.parameters()),training_controls=train_controls,calibration_controls=controls,
        selection_rule=cfg.get('selection_rule','family'),updates=cfg['response_steps'],lr=cfg['response_lr'])
    write(w.run/f'{kind}_fit.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);cfgpath=ap.parse_args().config;cfg=json.loads(cfgpath.read_text())
    sources=['scripts/run_semantic_family_source.py','scripts/run_ravel_semantic_source.py','scripts/run_ravel_source_coverage.py','scripts/run_causalgym_multisite.py',
        'scripts/run_causalgym_native_participation.py','scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/semantic_family.py','src/ccad/semantic_readout.py','src/ccad/ravel_controls.py','src/ccad/semantic_participation.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']
    w=MultisiteWork(cfg,cfgpath,sources);error=None
    try:
        from ccad.semantic_family import make_family,get_bases,apply_torch
        w.setup();torch=w.torch;ids=np.arange(len(w.panel));held=ids[[r['split']=='held_component_development' for r in w.panel]];cal=ids[[r['split']=='calibration' for r in w.panel]]
        raw,valid,align=w.delta(ids,cfg['mode'],w.hidden)
        if not valid.all() or any(len(a)!=1 for a in align):raise ValueError('Expected aligned single site')
        w.semantic_positions=np.asarray([a[0][0] for a in align]);donorpos=np.asarray([a[0][1] for a in align])
        if not np.array_equal(w.semantic_positions[w.donors],donorpos):raise ValueError('Donor alignment differs')
        xr=torch.as_tensor(raw[ids,w.semantic_positions],device=w.device);del raw
        source=w.load_sae(cfg['source_seed'],w.semantic_positions)
        with torch.no_grad():xd=torch.as_tensor(source['codes'][w.donors]-source['codes'],device=w.device)@torch.as_tensor(source['decoder'],device=w.device)
        summaries=[]
        for kind in cfg['methods']:
            torch.manual_seed(cfg['optimizer_seed']);common=kind.endswith('common');op=make_family(w.dim,3,cfg['das_rank'],common).to(w.device)
            x=xd if kind.startswith('decoded') else xr;result=fit(w,op,x,kind)
            with torch.no_grad():basis=get_bases(op).detach();flat=basis.reshape(-1,w.dim)
            error_orth=float((basis@basis.transpose(-1,-2)-torch.eye(cfg['das_rank'],device=w.device)).abs().max())
            cross=float((flat@flat.T-torch.eye(flat.shape[0],device=w.device)).abs().max())
            w.checks[kind+'_orthogonal']=error_orth<1e-4
            if common:w.checks[kind+'_disjoint']=cross<1e-4
            np.savez_compressed(w.run/f'{kind}_basis.npz',bases=basis.cpu().numpy())
            result.update(common_blocks=common,within_orthogonality_error=error_orth,full_orthogonality_error=cross,
                basis_path=f'{kind}_basis.npz',source_seed=cfg['source_seed'],stiefel_degrees_of_freedom=(w.dim*3*cfg['das_rank']-(3*cfg['das_rank'])*(3*cfg['das_rank']+1)//2) if common else 3*(w.dim*cfg['das_rank']-cfg['das_rank']*(cfg['das_rank']+1)//2))
            saved={};replay=[]
            for c in cfg['binary_controls']:
                control=torch.as_tensor(c,device=w.device)
                with torch.no_grad():
                    with torch.nn.utils.parametrize.cached():q=op(x[held],None,control)
                    replay.append(float((q-apply_torch(x[held],basis,control)).abs().max()))
                key=''.join(map(str,c));saved[key]=semantic_measure(w,held,expand_delta(w,held,q).cpu().numpy(),kind,cfg['mode'],c,
                    seed=cfg['source_seed'],operator_class='common orthogonal blocks' if common else 'independent ordered projectors',source_role='family_supervised')
            result['saved_basis_replay_max_delta_error']=max(replay);w.checks[kind+'_basis_replay']=max(replay)<1e-3
            np.savez_compressed(w.run/f'{kind}_held_outputs.npz',row_ids=held,**saved);write(w.run/f'{kind}_fit.json',result)
            summaries.append(result);write(w.run/'family_source_summary.json',dict(methods=summaries,scope=cfg['scope']))
            del op,basis,flat,q,saved
        # Frozen earlier singleton-trained choices get the same family calibration.
        for ref in cfg.get('reference_sources',[]):
            parent=ROOT/ref['run'];pc=json.loads(w.checked(parent/'config.resolved.json').read_text());torch.manual_seed(pc['optimizer_seed'])
            op=make_family(w.dim,3,pc['das_rank'],False).to(w.device)
            op.load_state_dict(torch.load(w.checked(parent/f"{ref['kind']}_state.pt"),map_location=w.device,weights_only=True));op.eval()
            x=xd if ref['kind']=='sae_mdas' else xr;stats=calibration(w,op,x,cal,cfg['binary_controls'])
            with torch.no_grad():basis=get_bases(op).detach().cpu().numpy()
            np.savez_compressed(w.run/f"{ref['label']}_basis.npz",bases=basis)
            summaries.append(dict(method=ref['label'],selected=stats,source_reference=ref,basis_path=f"{ref['label']}_basis.npz",source_seed=cfg['source_seed'],common_blocks=False))
            write(w.run/'family_source_summary.json',dict(methods=summaries,scope=cfg['scope']));del op,basis
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
