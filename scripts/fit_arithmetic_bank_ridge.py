"""Matched ridge control: retain the existing 64 read coordinates.

The full_activation array is stored in the existing full-code interface, with
exact zero coefficients outside the frozen write bank. All other arrays are
copied unchanged. Uses the original source fit states and existing ridge code.
"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,json,sys,traceback,platform
from pathlib import Path
import numpy as np
from fit_arithmetic_query_readouts import fit_full_code
from run_causalgym_multisite import MultisiteWork,ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());spec=cfg['readout_fit']
    w=MultisiteWork(cfg,args.config,['scripts/fit_arithmetic_bank_ridge.py','scripts/fit_arithmetic_query_readouts.py',
                                  'scripts/run_causalgym_multisite.py','src/ccad/artifacts.py'])
    error=None
    try:
        import torch
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
        parent=ROOT/spec['relation_run'];prior=ROOT/spec['copy_readouts_from'];cache=ROOT/spec['source_cache_identity']
        assert json.loads(w.checked(prior/'status.json').read_text())['status']=='PASS'
        pc=json.loads(w.checked(prior/'config.resolved.json').read_text())
        assert all(pc[k]==cfg[k] for k in ['model_revision','training_run','checkpoint_step'])
        assert all(pc['readout_fit'][k]==spec[k] for k in ['relation_run','source_cache_identity','ridge_grid','fold_seed'])
        panel=json.loads(w.checked(parent/'panel.json').read_text())
        with np.load(w.checked(cache/'states.npz')) as z:hidden=z['hidden']
        with np.load(w.checked(parent/'source_view_states.npz')) as z:hidden=np.concatenate([hidden,z['hidden']])
        assert len(hidden)==len(panel['rows'])
        tc=json.loads(w.checked(ROOT/cfg['training_run']/'config.resolved.json').read_text())
        sys.path.extend([tc['dictionary_source_dir'],tc['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        checkpoints=json.loads(w.checked(ROOT/cfg['training_run']/'checkpoints.json').read_text())['checkpoints']
        def encode(seed):
            checkpoint=next(r for r in checkpoints if r['seed']==seed and r['step']==cfg['checkpoint_step'])
            path=w.checked(checkpoint['path'])
            import hashlib
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==checkpoint['sha256']
            ae=AutoEncoderTopK(hidden.shape[-1],tc['dict_size'],tc['k'])
            ae.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            ae.to(cfg['encoder_device']);ae.eval();ae.requires_grad_(False)
            with torch.no_grad():
                code=ae.encode(torch.tensor(hidden.reshape(-1,hidden.shape[-1]),device=cfg['encoder_device']))
            return code.cpu().numpy().reshape(*hidden.shape[:-1],-1)
        for s in cfg['seeds']:
            t=s%5+1
            with np.load(w.checked(prior/f'readout_s{s}_t{t}.npz')) as z:payload={k:z[k] for k in z.files}
            pairs=payload['fit_pairs'];ii,jj=pairs.T;gate=payload['source_gate']
            assert all(panel['rows'][int(i)]['split']=='fit' for i in pairs.ravel())
            sc=encode(s);target=encode(t);source=sc[jj]-sc[ii];target=target[jj]-target[ii]
            roles=(np.arange(hidden.shape[1])[None]+np.array([1-panel['rows'][int(i)]['template'] for i in ii])[:,None]).ravel()
            for c,op in enumerate(['unit','tens']):
                si=payload[f'{op}_source_indices'];bank=payload[f'{op}_target_indices']
                y=source[...,si].reshape(-1,len(si))*gate[roles][:,si,c]
                coef,info=fit_full_code(target[...,bank].reshape(-1,len(bank)).astype(float),y.astype(float),roles,gate.shape[0],spec)
                full=np.zeros_like(payload[f'{op}_full_activation']);full[:,bank,:]=coef
                payload[f'{op}_full_activation']=full
                w.record(kind='readout_fit',task='arithmetic_source_fit',row_id=c,component=f'source{s}_target{t}',
                         method='bank_ridge',mode='64_input_same_ridge',seed=s,source_seed=s,target_seed=t,operation=op,roles=info)
            np.savez_compressed(w.run/f'readout_s{s}_t{t}.npz',**payload)
        w.checks.update(no_target_outputs=True,same_fit_pairs_and_bank=True,same_ridge_grid_and_folds=True)
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,encoder_device=cfg['encoder_device'])
    except Exception as exc:
        error=repr(exc);(w.run/'traceback.log').write_text(traceback.format_exc())
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
