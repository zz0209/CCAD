"""Translate an effective target readout into bounded target-native writes.

The desired update uses only target codes and frozen source projectors.
No target semantic label or true source edit enters support selection/NNLS.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write
from run_ravel_semantic_source import expand_delta
from run_ravel_source_coverage import semantic_measure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    files=['scripts/evaluate_projected_native_writer.py','scripts/run_ravel_semantic_source.py','scripts/run_ravel_source_coverage.py',
        'scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_participation.py','scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/ravel_controls.py','src/ccad/semantic_participation.py',
        'src/ccad/native_operation.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,files);error=None
    try:
        from safetensors import safe_open
        from ccad.ravel_controls import MultiDAS
        from ccad.native_operation import adaptive_writable_support,batched_project_native
        parent=ROOT/cfg['source_run'];pc=json.loads(w.checked(parent/'config.resolved.json').read_text())
        if json.loads(w.checked(parent/'status.json').read_text())['status']!='PASS':raise ValueError('Source must finish')
        if any(cfg[k]!=pc[k] for k in ['layer','model_revision','sae_root','source_seed','prepared_panel']):raise ValueError('Source identity differs')
        w.setup();torch=w.torch;ids=np.arange(len(w.panel));held=ids[[r['split']=='held_component_development' for r in w.panel]]
        valid,align=w.alignment(ids,cfg['mode'])
        if not valid.all() or any(len(a)!=1 for a in align):raise ValueError('Expected one aligned entity token')
        w.semantic_positions=np.asarray([a[0][0] for a in align]);dpos=np.asarray([a[0][1] for a in align])
        if not np.array_equal(w.semantic_positions[w.donors],dpos):raise ValueError('Donor alignment differs')
        source=w.load_sae(cfg['source_seed'],w.semantic_positions)
        with torch.no_grad():source_difference=torch.as_tensor(source['codes'][w.donors]-source['codes'],device=w.device)@torch.as_tensor(source['decoder'],device=w.device)
        operator=MultiDAS(w.dim,3,pc['das_rank']).to(w.device)
        operator.load_state_dict(torch.load(w.checked(parent/(cfg['source_kind']+'_state.pt')),map_location=w.device,weights_only=True));operator.eval()
        source_saved=np.load(w.checked(parent/(cfg['source_kind']+'_held_outputs.npz')))
        if not np.array_equal(source_saved['row_ids'],held):raise ValueError('Source held rows differ')
        controls=cfg['binary_controls']+cfg['fractional_controls'];references={};native_records=[];source_errors=[]
        name=lambda c:''.join(str(int(v)) if v in [0,1] else str(v) for v in c)
        def project(x,c):
            with torch.no_grad():return operator(x,None,torch.as_tensor(c,device=w.device))
        def evaluate(q,method,target,c,**extra):
            delta=expand_delta(w,held,q).detach().cpu().numpy()
            return semantic_measure(w,held,delta,method,cfg['mode'],c,seed=cfg['source_seed'],target_seed=target,
                reference=references.get(name(c)),control_is_binary=all(v in [0,1] for v in c),
                primary_fractional_endpoint='KL to the identical frozen source control; thresholded accuracy descriptive only',**extra)
        for c in controls:
            key=name(c)
            if c in cfg['binary_controls']:references[key]=source_saved[key]
            output=evaluate(project(source_difference[held],c),'frozen_source',cfg['source_seed'],c)
            if key in references:source_errors.append(float(np.max(np.abs(output-references[key]))))
            references[key]=output
        w.checks['source_replayed']=max(source_errors)<cfg['replay_logprob_tolerance']
        if not w.checks['source_replayed']:raise ValueError('Source replay differs')
        write(w.run/'source_replay_witness.json',dict(max_logprob_error=max(source_errors),tolerance=cfg['replay_logprob_tolerance']))
        del source,source_difference
        hbase=torch.as_tensor(w.hidden[held,w.semantic_positions[held]],device=w.device)
        for target in cfg['target_seeds']:
            sae=w.load_sae(target,w.semantic_positions);d=torch.as_tensor(sae['decoder'],device=w.device)
            z=torch.as_tensor(sae['codes'][held],device=w.device)
            with torch.no_grad():difference=torch.as_tensor(sae['codes'][w.donors[held]]-sae['codes'][held],device=w.device)@d
            path=Path(cfg['sae_root'])/f'seed_{target}/sae.safetensors'
            with safe_open(w.checked(path),framework='pt',device=str(w.device)) as f:
                ew,eb,db=[f.get_tensor(k) for k in ['encoder.weight','encoder.bias','b_dec']]
            for ci,c in enumerate(controls):
                key=name(c);q=project(difference,c);evaluate(q,'direct_shared_space',target,c,operation_class='signed contribution readout')
                # Encoder baseline preserves the original base residual when
                # its new nonnegative code is written through the real decoder.
                with torch.no_grad():
                    act,ix,_=w.encode_kernel(hbase+q-db,ew,eb,sae['cfg']['k'],sae['cfg']['activation'])
                    edited=torch.zeros_like(z).scatter_(1,ix,act);u_full=edited-z;realized=u_full@d
                evaluate(realized,'target_reencode',target,c,operation_class='native nonnegative code write')
                np.savez_compressed(w.run/f't{target}_{key}_reencode.npz',row_ids=held,code_increment=u_full.cpu().numpy())
                native_records.append(dict(target_seed=target,control=list(c),method='target_reencode',
                    changed_members=(u_full!=0).sum(1).cpu().tolist(),minimum_final_state=float(edited.min()),
                    squared_writer_error=((realized-q)**2).sum(1).cpu().tolist(),desired_energy=q.square().sum(1).cpu().tolist()))
                start=time.perf_counter()
                selected=adaptive_writable_support(q.cpu().numpy(),z.cpu().numpy(),sae['decoder'],max(cfg['writer_budgets']),
                    np.ones(len(sae['decoder']),bool),device=str(w.device))
                torch.cuda.synchronize();selection_seconds=time.perf_counter()-start
                supports=[(f'adaptive_native{budget}',selected[:,:budget]) for budget in cfg['writer_budgets']]
                rng=np.random.default_rng(cfg['random_support_seed']+100*target+ci);random=[]
                for row in z.cpu().numpy():
                    active=np.flatnonzero(row>0)
                    if len(active)>cfg['random_budget']:raise ValueError('Random baseline budget cannot retain base active coordinates')
                    available=np.flatnonzero(row==0);extra=rng.choice(available,cfg['random_budget']-len(active),replace=False)
                    random.append(np.concatenate([active,extra]))
                supports.append((f'random_base_native{cfg["random_budget"]}',np.asarray(random)))
                for method,support in supports:
                    begin=time.perf_counter();index=torch.as_tensor(support,device=w.device);local_z=z.gather(1,index);local_d=d[index]
                    with torch.no_grad():u,realized,diag=batched_project_native(q,local_z,local_d,max_steps=cfg['writer_steps'],tolerance=cfg['writer_tolerance'])
                    torch.cuda.synchronize();solve_seconds=time.perf_counter()-begin
                    w.checks[f'feasible_t{target}_{key}_{method}']=diag['minimum_final_state']>=-1e-7
                    evaluate(realized,method,target,c,operation_class='native nonnegative code write',candidate_members=support.shape[1])
                    np.savez_compressed(w.run/f't{target}_{key}_{method}.npz',row_ids=held,members=support,code_increment=u.cpu().numpy(),base_selected_codes=local_z.cpu().numpy())
                    native_records.append(dict(target_seed=target,control=list(c),method=method,candidate_members=support.shape[1],
                        changed_members=(u!=0).sum(1).cpu().tolist(),selection_seconds=selection_seconds if method.startswith('adaptive') else None,
                        selection_timing_scope='One shared max-budget adaptive path per control; do not sum its repeated value across prefixes',
                        solve_seconds=solve_seconds,desired_energy=q.square().sum(1).cpu().tolist(),**diag))
                    del local_d,local_z,u,index
                write(w.run/'native_writer_diagnostics.json',dict(rows=native_records,source_replay_max=max(source_errors),
                    scope='Target decoded difference projected by frozen source controls is the desired vector. Actual target decoder/native nonnegative state constraints; support/solve see no source truth or semantic endpoint. Approximate solves retain KKT residuals; support search is greedy, not globally optimal. TopK encoder-output closure after the realized write is not asserted.'))
                w.progress('PROJECTED_NATIVE_CONTROL',target_seed=target,control=c,selection_seconds=selection_seconds,
                    last_solve_seconds=solve_seconds,minimum_state=diag['minimum_final_state'])
            del sae,d,z,difference,ew,eb,db
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
