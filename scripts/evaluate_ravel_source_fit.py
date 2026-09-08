"""Evaluate frozen source operators on original fit pairs and held city splits.

No retraining or checkpoint reselection. Held output agreement witnesses the
portable operator replay before interpreting fit-versus-held semantic behavior.
"""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write
from run_ravel_semantic_source import expand_delta
from run_ravel_source_coverage import semantic_measure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    sources=['scripts/evaluate_ravel_source_fit.py','scripts/run_ravel_semantic_source.py','scripts/run_ravel_source_coverage.py',
        'scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_participation.py','scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/semantic_participation.py','src/ccad/ravel_controls.py',
        'src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,sources);error=None
    try:
        parent=ROOT/cfg['source_run'];pc=json.loads(w.checked(parent/'config.resolved.json').read_text())
        if json.loads(w.checked(parent/'status.json').read_text())['status']!='PASS':raise ValueError('Source producer must complete')
        if any(cfg[k]!=pc[k] for k in ['model_revision','layer','sae_root','source_seed','mode','methods','das_rank']):
            raise ValueError('Frozen source identity/operator configuration differs')
        pp=json.loads(w.checked(parent/'panel.json').read_text())['rows']
        w.setup();torch=w.torch
        from ccad.ravel_controls import SemanticNative,MultiDBM,MultiDAS
        ids=np.arange(len(w.panel));key=lambda r:(r['entity'],r['donor_entity'],r['task'],r['template_index'])
        old={key(r):r for r in pp}
        for r in w.panel:
            q=old[key(r)]
            if any(r[k]!=q[k] for k in ['text','tokens','expected_ids','donor_expected_ids','split']):raise ValueError('Replay row identity differs')
        valid,aa=w.alignment(ids,cfg['mode'])
        if not valid.all() or any(len(a)!=1 for a in aa):raise ValueError('Expected single aligned site')
        w.semantic_positions=np.array([a[0][0] for a in aa]);donor_positions=np.array([a[0][1] for a in aa])
        if not np.array_equal(w.semantic_positions[w.donors],donor_positions):raise ValueError('Donor alignment differs')
        raw=w.hidden[w.donors,donor_positions]-w.hidden[ids,w.semantic_positions]
        xr=torch.as_tensor(raw,device=w.device);sae=w.load_sae(cfg['source_seed'],w.semantic_positions)
        xs=torch.as_tensor(sae['codes'][w.donors]-sae['codes'],device=w.device)
        decoder=torch.as_tensor(sae['decoder'],device=w.device);witness=[]
        held=np.array([i for i in ids if w.panel[i]['split']=='held_component_development'])
        for kind in cfg['methods']:
            torch.manual_seed(pc['optimizer_seed'])
            if kind.startswith('native') or kind=='sae_mdbm':
                g=np.load(w.checked(parent/f'{kind}_gates.npz'))['gates']
                operator=SemanticNative(g.shape[0],3,int((g>0).sum()),kind=='native_exclusive').to(w.device)
                with torch.no_grad():operator.gates.copy_(torch.as_tensor(g,device=w.device))
                x,d=xs,decoder
            else:
                operator=(MultiDBM(w.dim,3) if kind=='mdbm' else MultiDAS(w.dim,3,cfg['das_rank'])).to(w.device)
                operator.load_state_dict(torch.load(w.checked(parent/f'{kind}_state.pt'),map_location=w.device,weights_only=True))
                x,d=xr,None
            operator.eval();saved=np.load(w.checked(parent/f'{kind}_held_outputs.npz'))
            saved_index={int(i):j for j,i in enumerate(saved['row_ids'])}
            reference_order=np.array([saved_index[old[key(w.panel[i])]['row_id']] for i in held])
            for control in [[1,0,0],[0,1,0],[0,0,1]]:
                name=''.join(map(str,control))
                for split in ['fit','calibration','held_component_development']:
                    local=np.array([i for i in ids if w.panel[i]['split']==split])
                    # The producer materializes the entire96-row held delta
                    # before its32-row LM batches. Preserve that GEMM shape;
                    # a480-row GEMM can select different CUDA arithmetic.
                    with torch.no_grad():delta=expand_delta(w,local,operator(x[local],d,torch.as_tensor(control,device=w.device))).cpu().numpy()
                    output=semantic_measure(w,local,delta,kind,cfg['mode'],control,seed=cfg['source_seed'],frozen_source_run=cfg['source_run'])
                    if split=='held_component_development':held_output=output
                maximum=float(np.max(np.abs(held_output-saved[name][reference_order])))
                witness.append(dict(method=kind,control=name,max_full_logprob_error=maximum,rows=len(held)))
            w.progress('FROZEN_SOURCE_SPLITS_EVALUATED',method=kind,max_replay_error=max(r['max_full_logprob_error'] for r in witness if r['method']==kind))
            del operator,saved
        w.checks['frozen_source_held_replayed']=max(r['max_full_logprob_error'] for r in witness)<cfg['replay_logprob_tolerance']
        write(w.run/'replay_witness.json',dict(rows=witness,tolerance=cfg['replay_logprob_tolerance'],scope='Actual full-vocabulary held logprobs versus frozen producer; no source reselection'))
        groups={}
        for r in w.metrics:groups.setdefault((r['method'],r['split'],r['task'],r['endpoint']),[]).append(r)
        write(w.run/'semantic_summary.json',dict(cells=[dict(method=k[0],split=k[1],task=k[2],endpoint=k[3],n=len(rr),
            first_token_correct=float(np.mean([r['first_token_correct'] for r in rr]))) for k,rr in groups.items()]))
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
