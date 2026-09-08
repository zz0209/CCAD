"""Fit reusable signed contribution maps without target semantic supervision.

Compare identity, ordinary/full ridge, and rank-constrained residual fits with
ordinary or operation-family output metrics. Atom correspondences and actual
language-model use are separate consumers of these same frozen inputs.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write
from ccad.projected_correspondence import family_metric,penalized_low_rank,balanced_document_weights


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    files=['scripts/fit_projected_family_correspondence.py','src/ccad/projected_correspondence.py','src/ccad/ravel_controls.py',
           'src/ccad/semantic_participation.py','scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_transfer.py',
           'scripts/run_r011s1_raw_hook_asset.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,files);error=None
    try:
        manifest=json.loads((w.run/'manifest.json').read_text())
        manifest.update(mean_constants_source_split=cfg['natural_run']+' natural document mean partition; decoded source and target means stored separately',
            threshold_source_split='Fixed rank/alpha grid before this fit; alpha selected only on natural calibration operation-family error')
        write(w.run/'manifest.json',manifest)
        import torch,scipy,psutil
        from scipy.sparse import csr_matrix
        from safetensors import safe_open
        from ccad.ravel_controls import MultiDAS
        torch.set_num_threads(cfg['cpu_threads']);w.torch=torch;w.device=torch.device('cpu')
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
            torch=torch.__version__,scipy=scipy.__version__,device='cpu',cpu_threads=cfg['cpu_threads'])
        parent=ROOT/cfg['source_run'];pc=json.loads(w.checked(parent/'config.resolved.json').read_text())
        ref=ROOT/cfg['natural_run'];rc=json.loads(w.checked(ref/'config.resolved.json').read_text())
        status_bytes=(parent/'status.json').read_bytes()
        (w.run/'source_parent_status_at_read.json').write_bytes(status_bytes)
        parent_status=json.loads(w.checked(w.run/'source_parent_status_at_read.json').read_text())['status']
        if parent_status!='PASS':
            fit_path=parent/(cfg['source_kind']+'_fit.json')
            output_path=parent/(cfg['source_kind']+'_held_outputs.npz')
            phase=json.loads((parent/'progress.json').read_text())
            phase_complete=(phase.get('method')==cfg['source_kind'] and phase['stage']=='SEMANTIC_SOURCE_EVALUATED') or (
                phase.get('method') in pc['methods'] and pc['methods'].index(phase['method'])>pc['methods'].index(cfg['source_kind']))
            if not (cfg.get('allow_completed_source_phase') and parent_status=='RUNNING' and phase_complete and fit_path.exists() and output_path.exists() and output_path.stat().st_size>0):
                raise ValueError('Source phase must complete before correspondence')
            write(w.run/'source_parent_phase_at_read.json',phase)
            completed=json.loads(w.checked(fit_path).read_text())
            if completed['trace'][-1]['step']!=pc['response_steps']:raise ValueError('Source fitting phase incomplete')
            w.checked(output_path)
        if json.loads(w.checked(ref/'status.json').read_text())['status']!='PASS':raise ValueError('Natural code producer must pass')
        for k in ['model_revision','layer','sae_root']:
            if pc[k]!=cfg[k] or rc[k]!=cfg[k]:raise ValueError('Source/reference identity differs: '+k)
        if pc['source_seed']!=cfg['source_seed'] or cfg['source_kind']!='sae_mdas':raise ValueError('Expected decoded source operator')
        state=torch.load(w.checked(parent/(cfg['source_kind']+'_state.pt')),map_location='cpu',weights_only=True)
        operator=MultiDAS(2048,len(cfg['tasks']),pc['das_rank']);operator.load_state_dict(state);operator.eval()
        rotations=np.stack([r.weight.detach().numpy().astype(float) for r in operator.rotations])
        w.checks['frozen_source_projectors_orthogonal']=max(float(np.max(np.abs(r@r.T-np.eye(r.shape[0])))) for r in rotations)<1e-4
        if not w.checks['frozen_source_projectors_orthogonal']:raise ValueError('Source projection rows are not orthogonal')
        projectors=np.stack([r.T@r for r in rotations]);del operator,state
        metric=family_metric(projectors);L,Lplus=metric['factor'],metric['inverse']
        np.savez_compressed(w.run/'source_family.npz',rotations=rotations.astype(np.float32),projectors=projectors,
            controls=np.asarray(metric['controls']),operators=np.stack(metric['operators']),metric_factor=L,metric_inverse=Lplus)
        write(w.run/'family_identity.json',dict(source_run=cfg['source_run'],source_kind=cfg['source_kind'],
            source_seed=cfg['source_seed'],source_selected=json.loads(w.checked(parent/(cfg['source_kind']+'_fit.json')).read_text())['selected_step'],
            ordered_attributes=cfg['tasks'],rank_per_control=pc['das_rank'],metric_rank=metric['effective_rank'],parent_status_at_read=parent_status,
            scope='Frozen supervised source projectors; target correspondence uses only separate natural paired documents. Signed readout, not native SAE masking.'))
        rows=np.load(w.checked(ref/'natural_rows.npz'));splits=rows['split'];docs=rows['document_ids']
        indices={s:np.flatnonzero(splits==s) for s in ['mean','discovery','calibration']}
        weights={s:balanced_document_weights(docs[indices[s]]) for s in indices}
        def codes(seed):
            a=np.load(w.checked(ref/f'seed{seed}_natural_codes.npz'));ix=a['indices'];v=a['activations']
            return csr_matrix((v.ravel(),ix.ravel(),np.arange(0,v.size+1,ix.shape[1])),shape=(len(ix),int(a['width'])))
        def decoder(seed):
            with safe_open(w.checked(Path(cfg['sae_root'])/f'seed_{seed}/sae.safetensors'),framework='np') as f:return f.get_tensor('W_dec')
        source_codes=codes(cfg['source_seed']);source_decoder=decoder(cfg['source_seed'])
        source=(source_codes@source_decoder).astype(float)
        mu_source=weights['mean']@source[indices['mean']];y=source-mu_source
        yfit=y[indices['discovery']];ycal=y[indices['calibration']];ycalL=ycal@L
        calw=weights['calibration'];den=float(np.sum(calw[:,None]*ycalL**2))
        fitw=weights['discovery'];root=np.sqrt(fitw)[:,None]
        selection=[]
        for target in cfg['target_seeds']:
            if target==0:
                target_codes=None;target_decoder=None;target_state=rows['hidden'].astype(float)
            else:
                target_codes=codes(target);target_decoder=decoder(target)
                target_state=(target_codes@target_decoder).astype(float)
            mu_target=weights['mean']@target_state[indices['mean']];x=target_state-mu_target
            xf=x[indices['discovery']];xc=x[indices['calibration']];xcL=xc@L
            wx=xf*root;wy=yfit*root;gram=wx.T@wx;cross=wx.T@wy
            residual_cross=cross-gram;unit=float(np.trace(gram)/gram.shape[0])
            candidates={};details=[]
            def score(name,alpha,left,right,identity,extra=None,dense=None):
                if dense is not None:pred=xc@dense;predL=pred@L
                else:
                    scores=xc@left;pred=scores@right;predL=scores@(right@L)
                if identity:pred=pred+xc;predL=predL+xcL
                loss=float(np.sum(calw[:,None]*(predL-ycalL)**2)/den)
                row=dict(method=name,alpha=alpha,natural_calibration_relative_family_error=loss,
                    natural_calibration_state_error=float(np.sum(calw[:,None]*(pred-ycal)**2)),**(extra or {}))
                details.append(row)
                if name not in candidates or loss<candidates[name]['meta']['natural_calibration_relative_family_error']:
                    candidates[name]=dict(meta=row,left=None if left is None else left.copy(),right=None if right is None else right.copy(),
                        identity=identity,dense=None if dense is None else dense.copy())
            score('direct_shared_space',None,np.empty((x.shape[1],0)),np.empty((0,x.shape[1])),True)
            for alpha in cfg['ridge_alphas']:
                if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Family fit budget exhausted')
                a=gram+(alpha*unit)*np.eye(gram.shape[0])
                full=np.linalg.solve(a,cross);correction=np.linalg.solve(a,residual_cross)
                score('unanchored_full_ridge',alpha,None,None,False,dense=full)
                score('anchored_full_ridge',alpha,None,None,True,dense=correction)
                # One decomposition at the largest requested rank; smaller
                # prefixes are the exact penalized RRR solutions too.
                for kind,coefficient,cr,identity,factor,inverse in [
                    ('anchored_state',correction,residual_cross,True,None,None),
                    ('anchored_family',correction,residual_cross,True,L,Lplus),
                    ('unanchored_family',full,cross,False,L,Lplus)]:
                    left,right,diag=penalized_low_rank(coefficient,cr,max(cfg['ranks']),factor,inverse)
                    for rank in cfg['ranks']:
                        score(f'{kind}_rank{rank}',alpha,left[:,:rank],right[:rank],identity,
                            dict(requested_rank=rank,effective_rank=min(rank,diag['effective_rank'])))
                w.progress('PROJECTED_NATURAL_CANDIDATES',target_seed=target,alpha=alpha,rss_bytes=psutil.Process().memory_info().rss)
            for name,result in candidates.items():
                arrays=dict(mean_source=mu_source,mean_target=mu_target,identity_prior=np.asarray(result['identity']))
                if result['dense'] is not None:arrays['dense']=result['dense'].astype(np.float32)
                else:arrays.update(left=result['left'].astype(np.float32),right=result['right'].astype(np.float32))
                path=w.run/f's{cfg["source_seed"]}_t{target}_{name}.npz';np.savez_compressed(path,**arrays)
                row=dict(source_seed=cfg['source_seed'],target_seed=target,path=str(path.relative_to(w.run)),**result['meta'])
                selection.append(row)
                w.record(kind='natural_correspondence',task='semantic_family',row_id=len(selection),component=f'source_{cfg["source_seed"]}_target_{target}',
                    method=name,seed=cfg['source_seed'],target_seed=target,operation='family_fit',split='natural_calibration',
                    natural_calibration_relative_family_error=result['meta']['natural_calibration_relative_family_error'],selected_alpha=result['meta']['alpha'])
            if target!=0:
                assignment=ROOT/cfg['atom_assignment_template'].format(source=cfg['source_seed'],target=target)
                matched=np.load(w.checked(assignment))['target_indices']
                if len(matched)!=source_codes.shape[1] or len(np.unique(matched))!=len(matched):raise ValueError('Incomplete PW-MCC assignment')
                sf=source_codes[indices['discovery']];tf=target_codes[indices['discovery']][:,matched]
                sm=np.asarray(source_codes[indices['mean']].T.dot(weights['mean'])).ravel()
                tm=np.asarray(target_codes[indices['mean']][:,matched].T.dot(weights['mean'])).ravel()
                es=np.asarray(sf.T.dot(fitw)).ravel();et=np.asarray(tf.T.dot(fitw)).ravel()
                covariance=np.asarray(sf.multiply(tf).T.dot(fitw)).ravel()-sm*et-tm*es+sm*tm
                variance=np.maximum(0,np.asarray(tf.multiply(tf).T.dot(fitw)).ravel()-2*tm*et+tm*tm)
                active=variance>max(float(variance.max())*1e-8,1e-12);unit_var=float(variance[active].mean())
                atom_best=None;atom_trace=[]
                for alpha in cfg['ridge_alphas']:
                    beta=np.divide(covariance,variance+alpha*unit_var,out=np.zeros_like(covariance),where=active)
                    pred=(target_codes[indices['calibration']][:,matched].multiply(beta)@source_decoder)-(tm*beta)@source_decoder
                    loss=float(np.sum(calw[:,None]*((pred-ycal)@L)**2)/den)
                    atom_trace.append(dict(alpha=alpha,natural_calibration_relative_family_error=loss))
                    if atom_best is None or loss<atom_best['loss']:atom_best=dict(alpha=alpha,loss=loss,beta=beta.copy())
                path=w.run/f's{cfg["source_seed"]}_t{target}_atom_pw_mcc.npz'
                np.savez_compressed(path,target_indices=matched,beta=atom_best['beta'].astype(np.float32),mean_source_codes=sm,mean_target_matched_codes=tm)
                row=dict(source_seed=cfg['source_seed'],target_seed=target,method='atom_pw_mcc',alpha=atom_best['alpha'],
                    natural_calibration_relative_family_error=atom_best['loss'],path=str(path.relative_to(w.run)),
                    source_members=len(matched),active_scalar_fits=int(active.sum()),negative_scales=int((atom_best['beta']<0).sum()))
                selection.append(row);write(w.run/f's{cfg["source_seed"]}_t{target}_atom_calibration.json',dict(candidates=atom_trace))
                w.record(kind='natural_correspondence',task='semantic_family',row_id=len(selection),component=f'source_{cfg["source_seed"]}_target_{target}',
                    method='atom_pw_mcc',seed=cfg['source_seed'],target_seed=target,operation='family_fit',split='natural_calibration',
                    natural_calibration_relative_family_error=atom_best['loss'],selected_alpha=atom_best['alpha'])
            write(w.run/f's{cfg["source_seed"]}_t{target}_calibration.json',dict(candidates=details))
            w.progress('PROJECTED_CORRESPONDENCE_COMPLETE',target_seed=target,selected=[r for r in selection if r['target_seed']==target])
            del target_codes,target_decoder,target_state,x,xf,xc,wx,wy,candidates
        write(w.run/'correspondence_selection.json',dict(rows=selection,config_scope=cfg['scope'],
            objective='Fit centered decoded source from decoded target. Anchored fits regress only source-target residual. Family metric sums all seven fixed binary operator errors. Rank-constrained ridge penalizes the transformed output coefficient; ordinary state and family fits are distinct constrained estimators. Same natural calibration family endpoint chooses alpha for every method.',
            interpretation='Low-rank correction factors induce signed many-to-many decoder contributions. No claim of unique semantic membership, target-native editing, new regression algorithm, or semantic performance follows from natural errors.'))
        w.checks['all_target_maps_written']=len(selection)==len(cfg['target_seeds'])*(3*len(cfg['ranks'])+3)+sum(t!=0 for t in cfg['target_seeds'])
        w.checks['finite_calibration']=all(np.isfinite(r['natural_calibration_relative_family_error']) for r in selection)
        w.checks['natural_audit_absent']=not np.any(splits=='audit')
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
