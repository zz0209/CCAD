"""Execute frozen semantic operation reuse through learned and direct maps.

Target maps/alphas are already fixed on natural data. The target RAVEL labels
are read only for evaluation. Fractional controls primarily test source KL;
thresholded categorical labels on them are descriptive and never selected on.
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
    files=['scripts/evaluate_projected_family_transfer.py','scripts/run_ravel_semantic_source.py','scripts/run_ravel_source_coverage.py',
        'scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_participation.py','scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/ravel_controls.py','src/ccad/semantic_participation.py',
        'src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,files);error=None
    try:
        parent=ROOT/cfg['source_run'];fitted=ROOT/cfg['correspondence_run']
        for p in [parent,fitted]:
            if json.loads(w.checked(p/'status.json').read_text())['status']!='PASS':raise ValueError('Source and correspondence must finish')
        pc=json.loads(w.checked(parent/'config.resolved.json').read_text())
        fc=json.loads(w.checked(fitted/'config.resolved.json').read_text())
        family=json.loads(w.checked(fitted/'family_identity.json').read_text())
        if family['source_run']!=cfg['source_run'] or family['source_kind']!=cfg['source_kind']:raise ValueError('Frozen source family differs')
        for k in ['layer','model_revision','sae_root','source_seed']:
            if cfg[k]!=pc[k] or cfg[k]!=fc[k]:raise ValueError('Model/SAE/source identity differs')
        if cfg['prepared_panel']!=pc['prepared_panel']:raise ValueError('Producer panel and exact operator materialization must match')
        if family['source_selected']!=json.loads(w.checked(parent/(cfg['source_kind']+'_fit.json')).read_text())['selected_step']:
            raise ValueError('Source checkpoint changed after correspondence')
        selection=json.loads(w.checked(fitted/'correspondence_selection.json').read_text())['rows']
        w.setup();torch=w.torch
        from ccad.ravel_controls import MultiDAS
        ids=np.arange(len(w.panel));held=ids[[r['split']=='held_component_development' for r in w.panel]]
        valid,align=w.alignment(ids,cfg['mode'])
        if not valid.all() or any(len(a)!=1 for a in align):raise ValueError('Expected one shared aligned entity token')
        w.semantic_positions=np.asarray([a[0][0] for a in align]);dpos=np.asarray([a[0][1] for a in align])
        if not np.array_equal(w.semantic_positions[w.donors],dpos):raise ValueError('Donor alignment differs')
        raw=torch.as_tensor(w.hidden[w.donors,dpos]-w.hidden[ids,w.semantic_positions],device=w.device)
        source=w.load_sae(cfg['source_seed'],w.semantic_positions)
        source_decoder=torch.as_tensor(source['decoder'],device=w.device)
        with torch.no_grad():decoded=torch.as_tensor(source['codes'][w.donors]-source['codes'],device=w.device)@source_decoder
        operator=MultiDAS(w.dim,3,pc['das_rank']).to(w.device)
        operator.load_state_dict(torch.load(w.checked(parent/(cfg['source_kind']+'_state.pt')),map_location=w.device,weights_only=True));operator.eval()
        natural_rotations=np.load(w.checked(fitted/'source_family.npz'))['rotations']
        rotation_error=max(float(np.max(np.abs(r.weight.detach().cpu().numpy()-natural_rotations[a]))) for a,r in enumerate(operator.rotations))
        w.checks['natural_fit_same_source_parameters']=rotation_error<1e-4
        write(w.run/'source_rotation_device_witness.json',dict(max_cpu_gpu_rotation_difference=rotation_error,tolerance=1e-4))
        saved=np.load(w.checked(parent/(cfg['source_kind']+'_held_outputs.npz')))
        if not np.array_equal(saved['row_ids'],held):raise ValueError('Held source order differs')
        binary=cfg['binary_controls'];fractional=cfg['fractional_controls'];controls=binary+fractional
        references={};teacher_deltas={};replay=[]
        def name(c):return ''.join(str(int(v)) if v in [0,1] else str(v) for v in c)
        def run_operation(difference,control,reverse=False):
            c=torch.as_tensor(control,device=w.device)
            with torch.no_grad():
                if not reverse:return operator(difference,None,c)
                delta=torch.zeros_like(difference)
                for a in [2,1,0]:
                    if c[a]!=0:
                        r=operator.rotations[a].weight
                        delta=delta+c[a]*((difference-delta)@r.T)@r
                return delta
        def evaluate(q,method,target,c,reference=None,**extra):
            with torch.no_grad():delta=expand_delta(w,held,q).cpu().numpy()
            key=name(c);is_binary=all(v in [0,1] for v in c)
            output=semantic_measure(w,held,delta,method,cfg['mode'],c,seed=cfg['source_seed'],target_seed=target,reference=reference,
                control_is_binary=is_binary,primary_fractional_endpoint='KL to same frozen source control; thresholded token accuracy is descriptive only',**extra)
            if key in teacher_deltas:
                difference=q.detach().cpu().numpy()-teacher_deltas[key]
                for j,i in enumerate(held):
                    w.record(kind='hook_error',task=w.panel[i]['task'],row_id=int(i),component=w.panel[i]['component'],
                        mode=cfg['mode'],method=method,seed=cfg['source_seed'],target_seed=target,operation=key,control=list(c),
                        split=w.panel[i]['split'],squared_error=float(np.sum(difference[j].astype(float)**2)),
                        source_energy=float(np.sum(teacher_deltas[key][j].astype(float)**2)))
            return output
        for c in controls:
            key=name(c);q=run_operation(decoded[held],c)
            teacher_deltas[key]=q.cpu().numpy()
            prior=saved[key] if c in binary else None
            output=evaluate(q,'frozen_source',cfg['source_seed'],c,prior)
            references[key]=output
            if prior is not None:replay.append(dict(control=key,max_logprob_error=float(np.max(np.abs(output-prior)))))
        w.checks['source_held_replayed']=max(r['max_logprob_error'] for r in replay)<cfg['replay_logprob_tolerance']
        write(w.run/'source_replay_witness.json',dict(rows=replay,tolerance=cfg['replay_logprob_tolerance']))
        if not w.checks['source_held_replayed']:raise ValueError('Frozen source replay differs')
        for c in binary:
            if sum(c)>1:evaluate(run_operation(decoded[held],c,True),'source_reverse_order',cfg['source_seed'],c,references[name(c)])
        raw_operator=MultiDAS(w.dim,3,pc['das_rank']).to(w.device)
        raw_operator.load_state_dict(torch.load(w.checked(parent/'mdas_state.pt'),map_location=w.device,weights_only=True));raw_operator.eval()
        for c in controls:
            with torch.no_grad():q=raw_operator(raw[held],None,torch.as_tensor(c,device=w.device))
            evaluate(q,'raw_supervised_das',0,c,references[name(c)])
        del raw_operator,decoded
        for target in cfg['target_seeds']:
            if target==0:base_difference=raw[held];target_codes=None
            else:
                target_sae=w.load_sae(target,w.semantic_positions)
                target_codes=torch.as_tensor(target_sae['codes'][w.donors[held]]-target_sae['codes'][held],device=w.device)
                with torch.no_grad():base_difference=target_codes@torch.as_tensor(target_sae['decoder'],device=w.device)
                del target_sae
            maps=[r for r in selection if r['target_seed']==target]
            for meta in maps:
                start=time.perf_counter();a=np.load(w.checked(fitted/meta['path']));method=meta['method']
                with torch.no_grad():
                    if method=='atom_pw_mcc':
                        if target_codes is None:raise ValueError('Atom map requires target SAE codes')
                        prediction=(target_codes[:,a['target_indices']]*torch.as_tensor(a['beta'],device=w.device))@source_decoder
                    else:
                        if 'dense' in a:prediction=base_difference@torch.as_tensor(a['dense'],device=w.device)
                        else:prediction=(base_difference@torch.as_tensor(a['left'],device=w.device))@torch.as_tensor(a['right'],device=w.device)
                        if bool(a['identity_prior']):prediction=prediction+base_difference
                for c in controls:
                    evaluate(run_operation(prediction,c),method,target,c,references[name(c)],
                        natural_selected_alpha=meta['alpha'],map_path=meta['path'])
                w.progress('PROJECTED_MAP_EXECUTED',target_seed=target,method=method,evaluation_wall_seconds=time.perf_counter()-start)
                del a,prediction
            for c in binary:
                if sum(c)==1:
                    wrong=np.roll(c,1).tolist()
                    evaluate(run_operation(base_difference,wrong),'wrong_attribute_shared_space',target,c,references[name(c)])
            del base_difference,target_codes
        semantic=[r for r in w.metrics if r['kind']=='semantic'];groups={}
        for r in semantic:groups.setdefault((r['method'],r['target_seed'],tuple(r['control']),r['endpoint']),[]).append(r)
        write(w.run/'transfer_summary.json',dict(cells=[dict(method=k[0],target_seed=k[1],control=list(k[2]),endpoint=k[3],n=len(rr),
            first_token_correct=float(np.mean([r['first_token_correct'] for r in rr])),mean_source_kl=float(np.mean([r['kl_to_reference'] for r in rr]))) for k,rr in groups.items()],
            scope=cfg['scope']))
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
