"""Give complete PW-MCC signed calibration for each frozen source family."""
from __future__ import annotations
import argparse,json,platform,sys,time,traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write
from ccad.projected_correspondence import balanced_document_weights


def family_error(error,bases,controls,weights):
    # Row-space calculation avoids a large output metric eigendecomposition.
    total=0.
    for c in controls:
        delta=np.zeros_like(error)
        for a in range(len(bases)):
            if c[a]:delta+=c[a]*((error-delta)@bases[a].T)@bases[a]
        total+=float(np.sum(weights[:,None]*delta**2))
    return total


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);path=ap.parse_args().config;cfg=json.loads(path.read_text())
    w=MultisiteWork(cfg,path,['scripts/fit_semantic_cycle_atoms.py','scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/projected_correspondence.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']);error=None
    try:
        import torch,scipy,psutil
        from scipy.sparse import csr_matrix
        from safetensors import safe_open
        torch.set_num_threads(cfg['cpu_threads']);w.torch=torch;w.device=torch.device('cpu')
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,device='cpu',cpu_threads=cfg['cpu_threads'])
        reference=ROOT/cfg['natural_run'];nr=json.loads(w.checked(reference/'config.resolved.json').read_text())
        for k in ['layer','sae_root','model_revision']:
            if nr[k]!=cfg[k]:raise ValueError('Natural reference differs: '+k)
        rows=np.load(w.checked(reference/'natural_rows.npz'));splits=rows['split'];docs=rows['document_ids']
        ids={s:np.flatnonzero(splits==s) for s in ['mean','discovery','calibration']}
        weights={s:balanced_document_weights(docs[ids[s]]) for s in ids}
        cache={}
        for seed in cfg['seeds']:
            a=np.load(w.checked(reference/f'seed{seed}_natural_codes.npz'));ix=a['indices'];v=a['activations']
            cache[seed]=csr_matrix((v.ravel(),ix.ravel(),np.arange(0,v.size+1,ix.shape[1])),shape=(len(ix),int(a['width'])))
        selections=[]
        for spec in cfg['source_families']:
            if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Atom fit budget exhausted')
            source,target=spec['source_seed'],spec['target_seed'];sc,tc=cache[source],cache[target]
            bases=np.load(w.checked(ROOT/spec['basis_path']))['bases'].astype(float)
            lo,hi=sorted([source,target]);match=np.load(w.checked(ROOT/cfg['assignment_template'].format(source=lo,target=hi)))['target_indices']
            if source>target:match=np.argsort(match)
            if len(np.unique(match))!=sc.shape[1]:raise ValueError('Assignment must cover complete dictionary')
            with safe_open(w.checked(Path(cfg['sae_root'])/f'seed_{source}/sae.safetensors'),framework='np') as f:d=f.get_tensor('W_dec')
            sf=sc[ids['discovery']];tf=tc[ids['discovery']][:,match];fitw=weights['discovery'];calw=weights['calibration']
            sm=np.asarray(sc[ids['mean']].T.dot(weights['mean'])).ravel();tm=np.asarray(tc[ids['mean']][:,match].T.dot(weights['mean'])).ravel()
            es=np.asarray(sf.T.dot(fitw)).ravel();et=np.asarray(tf.T.dot(fitw)).ravel()
            covariance=np.asarray(sf.multiply(tf).T.dot(fitw)).ravel()-sm*et-tm*es+sm*tm
            variance=np.maximum(0,np.asarray(tf.multiply(tf).T.dot(fitw)).ravel()-2*tm*et+tm*tm)
            active=variance>max(float(variance.max())*1e-8,1e-12);unit=float(variance[active].mean())
            ycal=(sc[ids['calibration']]@d)-(sm@d);den=family_error(ycal,bases,cfg['binary_controls'],calw)
            trace=[];best=None
            for alpha in cfg['ridge_alphas']:
                beta=np.divide(covariance,variance+alpha*unit,out=np.zeros_like(covariance),where=active)
                pred=tc[ids['calibration']][:,match].multiply(beta)@d-(tm*beta)@d
                loss=family_error(pred-ycal,bases,cfg['binary_controls'],calw)/den;trace.append(dict(alpha=alpha,relative_family_error=loss))
                if best is None or loss<best['loss']:best=dict(alpha=alpha,loss=loss,beta=beta.copy())
            filename=f's{source}_t{target}_atom_pw_mcc.npz';np.savez_compressed(w.run/filename,target_indices=match,beta=best['beta'].astype(np.float32),mean_source_codes=sm,mean_target_codes=tm)
            row=dict(source_seed=source,target_seed=target,path=filename,selected_alpha=best['alpha'],relative_family_error=best['loss'],trace=trace,source_members=len(match),negative_scales=int((best['beta']<0).sum()))
            selections.append(row);write(w.run/'atom_selection.json',dict(rows=selections,scope=cfg['scope']))
            w.record(kind='natural_atom_calibration',task='semantic_family',row_id=source,component=f's{source}_t{target}',method='atom_pw_mcc',seed=source,target_seed=target,operation='fit_signed_gain',split='natural_calibration',relative_family_error=best['loss'])
            w.progress('CYCLE_ATOM_FIT',source_seed=source,target_seed=target,relative_family_error=best['loss'],rss_bytes=psutil.Process().memory_info().rss)
        w.checks['audit_not_encoded']=not np.any(splits=='audit');w.checks['five_full_assignments']=len(selections)==5
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
