"""Source-defined contribution partitions and measured cross-seed partial effects."""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from run_f4_source_reference_causal import (ROOT, np, write, jsonl, sha256,
    fixed_support_ridge, readout_atom_order, source_dose_scale, compare,
    HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor)
from run_r011s1_raw_hook_asset import entry, aggregate
from ccad.artifacts import validate_run_directory
from f4_probability_endpoints import endpoint_positions, probability_metrics


def partition_source(z, mean, decoder, basis, ids, weights, count, mode='head_tail'):
    ids=np.asarray(ids,dtype=int)
    if len(ids)!=2*count or len(set(ids.tolist()))!=len(ids):raise ValueError('Expected two equal source groups')
    # Sorting IDs first makes readout_atom_order's tie rule refer to actual IDs.
    ids=np.sort(ids);scalar=decoder[ids]@basis
    order,energy=readout_atom_order(z[:,ids],scalar,weights)
    if mode not in ('head_tail','interleaved'):raise ValueError('Unknown source partition')
    groups=[ids[order[::2]],ids[order[1::2]]] if mode=='interleaved' else [ids[order[:count]],ids[order[count:]]]
    y=np.column_stack([(z[:,g]-mean[g])@(decoder[g]@basis) for g in groups])
    return groups,y,dict(atom_ids=ids.tolist(),energies=energy.tolist(),groups=[g.tolist() for g in groups])


def additive_ridge(x,y,weights,ridge):
    # Separate scalar solves use exactly the existing full-map ridge kernel.
    columns=[];diagnostics=[]
    for k in range(2):
        w,d=fixed_support_ridge(x,y[:,k],weights,ridge,center=False)
        columns.append(w);diagnostics.append(d)
    w=np.column_stack(columns)
    full,_=fixed_support_ridge(x,y.sum(axis=1),weights,ridge,center=False)
    error=float(np.linalg.norm(w.sum(axis=1)-full)/max(np.linalg.norm(full),1e-20))
    if error>1e-8:raise ValueError('Ridge component additivity failed')
    return w,dict(components=diagnostics,full_sum_relative_error=error)


def family_scale(coordinates,basis,masked_hook,cap):
    deltas=[coordinates.sum(axis=1)[:,None]*basis,coordinates[:,0,None]*basis,coordinates[:,1,None]*basis]
    return min(source_dose_scale(d,masked_hook,cap) for d in deltas)


def shared_support(z,beta,weights,budget):
    """One fixed latent interface for both source parts, not two separate budgets."""
    z=np.asarray(z,dtype=float);beta=np.asarray(beta,dtype=float);w=np.asarray(weights,dtype=float);w=w/w.sum()
    if beta.shape!=(z.shape[1],2) or not 0<budget<=len(beta):raise ValueError('Invalid joint support shapes')
    variance=np.sum(w[:,None]*(z-w@z)**2,axis=0);energy=variance*np.sum(beta**2,axis=1)
    keep=np.lexsort((np.arange(len(beta)),-energy))[:budget];masked=np.zeros_like(beta);masked[keep]=beta[keep]
    return masked,dict(support=keep.tolist(),budget=budget,union_budget=budget,energy_sum=float(energy.sum()),retained_energy=float(energy[keep].sum()))


def joint_single_atom(z,y,weights,ridge):
    """Best ONE atom for both outputs under weighted conditional-variation error."""
    z=np.asarray(z,dtype=float);y=np.asarray(y,dtype=float);w=np.asarray(weights,dtype=float);w=w/w.sum()
    if y.shape!=(len(z),2) or ridge<=0:raise ValueError('Invalid joint atom inputs')
    x=z-w@z;target=y-w@y;variance=np.sum(w[:,None]*x*x,axis=0);cross=x.T@(w[:,None]*target)
    coefficients=np.divide(cross,variance[:,None]*(1+ridge),out=np.zeros_like(cross),where=variance[:,None]>0)
    source_energy=float(np.sum(w[:,None]*target**2))
    errors=source_energy-2*np.sum(coefficients*cross,axis=1)+variance*np.sum(coefficients**2,axis=1)
    atom=int(np.argmin(errors));beta=np.zeros_like(coefficients);beta[atom]=coefficients[atom]
    return beta,dict(atom=atom,coefficients=coefficients[atom].tolist(),weighted_error=float(errors[atom]),source_conditional_energy=source_energy,
        candidate_count=z.shape[1],objective='sum of two output conditional-variation squared errors; same single atom for both',ridge_fraction=ridge)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());compact=cfg.get('compact_shared_interface',False);confirmation=cfg.get('confirmation_inputs');execution=cfg.get('joint_sparse_execution');run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    compact_methods=('short_full','short_shared16','long_shared16','short_single','long_single')
    now=datetime.now(timezone.utc).isoformat();started=time.perf_counter();write(run/'config.resolved.json',cfg)
    files=['scripts/run_f4_source_components.py','scripts/run_f4_source_reference_causal.py',
        'scripts/f4_probability_endpoints.py','scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/hook_transport.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']
    code=[]
    for rel in files:
        p=ROOT/rel;dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.source.components.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose=cfg['purpose'],milestone='M4',evidence_level=cfg['evidence_level'],started_utc=now,project_root=str(ROOT),
        config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,
        audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original independent mean',
        threshold_source_split='frozen parent coefficients/groups; new source-only inputs' if confirmation else 'source-only discovery group ranking; exposed cases unchanged',statistics_unit='query/document/shared-seed dependencies',
        device='cuda:0',seeds=cfg['target_seeds'],resource_lease='cpu-heavy -> gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now))
    for name in ('metrics.raw.jsonl','stderr.log','stdout.log'):(run/name).touch()
    inputs=[];seen={};checks={};records=[];fits=[];forwards=0;error=None;env={};timers={}
    def checked(path,expected=None):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        if path not in seen:seen[path]=sha256(path);inputs.append(entry(path,'CCAD frozen asset','component_input'))
        if expected and seen[path]!=expected:raise ValueError('Input identity changed: '+str(path))
        return path
    def load(path,expected=None):return json.loads(checked(path,expected).read_text())
    def mmap(meta):return np.memmap(checked(meta['path'],meta['sha256']),dtype='<u2' if meta['dtype']=='uint16' else '<f4',mode='r',shape=tuple(meta['shape']))
    try:
        checked(args.config);checked('.aris/compute/local-r006b1-env-spec.json','3129a184d787ae9be38ac6d8d97dbf5087e5c838c112473fe45f3862064bb60f')
        assert cfg['audit_opened'] is False and cfg['target_seeds']==[1,2,3,4,5]
        frozen_parts={};frozen_arrays={};compact_coefficients={}
        if compact:
            frozen=cfg['frozen_components'];parent=ROOT/frozen['path']
            parent_fits=load(parent/'component_fits.json',frozen['fits_sha256'])
            frozen_parts={(r['source_seed'],r['source_atom']):r for r in parent_fits if r['kind']=='source_partition'}
            with np.load(checked(parent/'component_coefficients.npz',frozen['coefficients_sha256']),allow_pickle=False) as arrays:frozen_arrays={k:np.array(arrays[k]) for k in arrays.files}
            parent_coords=np.load(checked(parent/'component_coordinates.npz',frozen['coordinates_sha256']),allow_pickle=False)
            parent_cases=load(parent/'compiled_cases.json',frozen['cases_sha256'])
        old={};fitconfigs={};manifests={};saes={}
        for label in ('old_fit','new_fit'):
            spec=cfg[label];root=ROOT/spec['path'];fc=load(root/'config.resolved.json',spec['config_sha256']);fitconfigs[label]=fc
            with np.load(checked(root/'fit_coefficients.npz',spec['coefficients_sha256']),allow_pickle=False) as saved:
                for r in jsonl(checked(root/'metrics.raw.jsonl',spec['metadata_sha256'])):
                    if r['configuration']=='long128':old[r['source_seed'],r['source_atom'],r['target_seed']]=(r,np.array(saved[r['array_key']]))
            lc=next(c for c in fc['long_configurations'] if c['name']=='long128');asset=load(lc['asset_config'])
            manifest=load(Path(asset['bulk_output_dir'])/'asset_manifest.json',lc['asset_manifest_sha256'])
            for item in asset['saes']:manifests[item['seed']]=manifest;saes[item['seed']]=(item,asset)
        fresh=confirmation or fitconfigs['new_fit'];base=load(fresh['source_asset_config']);ev=load(fresh['evaluation_asset_config'])
        refs={p:load(fresh['reference_config_template'].format(panel=p)) for p in fresh['panels']};ref=refs['original']
        source_manifest=load(Path(base['bulk_asset_dir'])/'asset_manifest.json',base['asset_manifest_sha256'])
        eval_manifest=load(Path(ev['bulk_output_dir'])/'asset_manifest.json',ref['asset_manifest_sha256'])
        raw_manifest=load(Path(base['raw_hook_asset_dir'])/'raw_hook_manifest.json',base['raw_hook_manifest_sha256'])
        eval_raw_manifest=load(Path(ref['raw_hook_asset_dir'])/'raw_hook_manifest.json',ref['raw_hook_manifest_sha256'])
        raw={r['split']:mmap(r) for r in raw_manifest['splits'] if r['split'] in ('mean','discovery')} if not confirmation else {}
        eval_raw=mmap(next(r for r in eval_raw_manifest['splits'] if r['split']=='calibration'))
        rawmean=np.asarray(raw['mean'],dtype=float).mean(axis=0) if not confirmation else None
        tm=load(ref['token_manifest_path'],ref['token_manifest_sha256']);ti=tm['outputs']['calibration'];length=base['context_length']
        tokenpath=checked(ROOT/'runs'/ref['paired_corpus_run']/ti['path'],ti.get('sha256'))
        tokens=np.memmap(tokenpath,dtype='<u2',mode='r').reshape(-1,length)
        f=np.load(checked(ref['factors_path'],ref['factors_sha256']),allow_pickle=False)
        fi={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(f['source_seed'],f['source_atom'],f['target_seed']))}
        selections={p:load(fresh['source_selections'][p]['path'],fresh['source_selections'][p]['sha256']) for p in fresh['panels']}
        cases=[dict(panel=p,source_seed=r['source_seed'],source_atom=r['source_atom'],**r['entry']) for p,sel in selections.items() for r in sel['choices'] if r['entry'] and r['source_scope']['selected']]
        assert len(cases)==cfg['expected_cases'] and sum(len(s['choices']) for s in selections.values())==cfg['expected_requests']
        write(run/'case_selection.json',selections)
        sparse={};dec={};means={}
        def init_codes(name,seed,manifest,splits):
            dm=next(r for r in manifest['decoders'] if r['seed']==seed);dec[name,seed]=np.asarray(mmap(dict(dm,dtype='float32')),dtype=float)
            for split in splits:
                sm=next(r for r in manifest['splits'] if r['split']==split);parts={m['dtype']:mmap(m) for m in sm['files'] if m['seed']==seed};sparse[name,seed,split]=parts
            if 'mean' in splits:
                m=sparse[name,seed,'mean'];ii=m['uint16'];aa=m['float32'];means[name,seed]=np.bincount(ii.ravel(),weights=aa.ravel(),minlength=base['num_latents'])/len(ii)
        def dense(name,seed,split,rows):
            m=sparse[name,seed,split];z=np.zeros((len(rows),base['num_latents']));np.add.at(z,(np.arange(len(rows))[:,None],m['uint16'][rows]),m['float32'][rows]);return z
        queries=sorted({(e['source_seed'],e['source_atom']) for e in cases})
        for s in sorted({s for s,a in queries}):
            init_codes('source',s,source_manifest,() if confirmation else ('mean','discovery'));init_codes('evalsource',s,eval_manifest,('calibration',))
        if not confirmation:
            for t in cfg['target_seeds']:init_codes('long',t,manifests[t],('mean','discovery'))
        if compact:
            for t in cfg['target_seeds']:
                init_codes('short',t,source_manifest,() if confirmation else ('mean','discovery'));init_codes('evalshort',t,eval_manifest,('calibration',))
        coefficients={};query_data={};fit_arrays={}
        if confirmation:
            # No discovery rows or mean are materialized; every output coefficient is restored.
            fit_arrays={k:np.array(v) for k,v in frozen_arrays.items()};fits=parent_fits
            for s,a in queries:
                groups=[np.asarray(g,dtype=int) for g in frozen_parts[s,a]['groups']]
                targets=[t for t in cfg['target_seeds'] if t!=s]
                b=f['source_basis'][fi[s,a,targets[0]],:,0].astype(float)
                query_data[s,a]=(groups,b)
                for t in targets:
                    assert np.array_equal(f['source_basis'][fi[s,a,t],:,0],b)
                    assert sorted(np.concatenate(groups).tolist())==sorted(old[s,a,t][0]['source_candidate_ids'])
                    coefficients[s,a,t]=frozen_arrays[f'long_{s}_{a}_{t}']
                    for method in ('short_full','short_shared16','long_shared16','short_single','long_single'):
                        prefix='short' if method=='short_full' else method
                        compact_coefficients[s,a,t,method]=frozen_arrays[f'{prefix}_{s}_{a}_{t}']
                coefficients[s,a,'raw']=frozen_arrays[f'raw_{s}_{a}']
            checks['all_parent_coefficients_restored_no_fitting']=True
        else:
            for s,a in queries:
                targets=[t for t in cfg['target_seeds'] if t!=s];r,_=old[s,a,targets[0]];rr=np.asarray(r['discovery_rows']);ww=np.asarray(r['discovery_weights']);b=f['source_basis'][fi[s,a,targets[0]],:,0].astype(float)
                groups,y,selection=partition_source(dense('source',s,'discovery',rr),means['source',s],dec['source',s],b,r['source_candidate_ids'],ww,cfg['source_group_size'],cfg.get('partition_mode','head_tail'))
                if compact:
                    frozen=frozen_parts[s,a]
                    assert selection['groups']==frozen['groups'] and rr.tolist()==frozen['discovery_rows'] and ww.tolist()==frozen['discovery_weights']
                query_data[s,a]=(groups,b);fits.append(dict(kind='source_partition',source_seed=s,source_atom=a,discovery_rows=rr.tolist(),discovery_weights=ww.tolist(),**selection))
                for t in targets:
                    prior,priorbeta=old[s,a,t]
                    assert prior['discovery_rows']==r['discovery_rows'] and prior['discovery_weights']==r['discovery_weights'] and prior['source_candidate_ids']==r['source_candidate_ids']
                    assert np.array_equal(f['source_basis'][fi[s,a,t],:,0],b)
                    z=dense('long',t,'discovery',rr);x=(z-means['long',t])@dec['long',t]
                    if compact:beta=np.array(frozen_arrays[f'long_{s}_{a}_{t}']);diag=dict(restored_parent=True,new_fit=False)
                    else:
                        w,diag=additive_ridge(x,y,ww,cfg['ridge_fraction']);beta=dec['long',t]@w
                    rel=float(np.linalg.norm(beta.sum(axis=1)-priorbeta)/max(np.linalg.norm(priorbeta),1e-20))
                    if rel>1e-5:raise ValueError('Long full coefficient replay failed')
                    coefficients[s,a,t]=beta;fit_arrays[f'long_{s}_{a}_{t}']=beta
                    fits.append(dict(kind='long',source_seed=s,source_atom=a,target_seed=t,full_beta_replay_relative=rel,**diag))
                    if compact:
                        zs=dense('short',t,'discovery',rr);xs=(zs-means['short',t])@dec['short',t]
                        ws,ds=additive_ridge(xs,y,ww,cfg['ridge_fraction']);bs=dec['short',t]@ws
                        reference_short=dec['short',t]@f['query_target'][fi[s,a,t],:,0].astype(float)
                        rs=float(np.linalg.norm(bs.sum(axis=1)-reference_short)/max(np.linalg.norm(reference_short),1e-20))
                        if rs>1e-5:raise ValueError('Short component full-map replay failed')
                        compact_coefficients[s,a,t,'short_full']=bs;fit_arrays[f'short_{s}_{a}_{t}']=bs
                        fits.append(dict(kind='short_full',source_seed=s,source_atom=a,target_seed=t,full_beta_replay_relative=rs,**ds))
                        for name,zdata,bb in [('long',z,beta),('short',zs,bs)]:
                            masked,info=shared_support(zdata,bb,ww,cfg['readout_budget'])
                            atom,ad=joint_single_atom(zdata,y,ww,cfg['ridge_fraction'])
                            for method,value,diagnostic in ((name+'_shared16',masked,info),(name+'_single',atom,ad)):
                                compact_coefficients[s,a,t,method]=value;fit_arrays[f'{method}_{s}_{a}_{t}']=value
                                fits.append(dict(kind=method,source_seed=s,source_atom=a,target_seed=t,**diagnostic))
                if compact:w=np.array(frozen_arrays[f'raw_{s}_{a}']);diag=dict(restored_parent=True,new_fit=False)
                else:w,diag=additive_ridge(np.asarray(raw['discovery'][rr],dtype=float)-rawmean,y,ww,cfg['ridge_fraction'])
                reference=f['raw_target'][fi[s,a,targets[0]],:,0].astype(float)
                rel=float(np.linalg.norm(w.sum(axis=1)-reference)/max(np.linalg.norm(reference),1e-20))
                if rel>1e-5:raise ValueError('Raw full coefficient replay failed')
                for t in targets:assert np.array_equal(f['raw_target'][fi[s,a,t],:,0],reference)
                coefficients[s,a,'raw']=w;fit_arrays[f'raw_{s}_{a}']=w
                fits.append(dict(kind='raw',source_seed=s,source_atom=a,full_coefficient_replay_relative=rel,**diag))
        np.savez_compressed(run/'component_coefficients.npz',**fit_arrays);write(run/'component_fits.json',fits)
        if confirmation:
            assert sha256(run/'component_fits.json')==frozen['fits_sha256']
            with np.load(run/'component_coefficients.npz',allow_pickle=False) as restored:
                assert set(restored.files)==set(frozen_arrays) and all(np.array_equal(restored[k],v) for k,v in frozen_arrays.items())
        if execution:
            # Optional frozen sparse interfaces share the existing source/case/
            # dose consumer. No discovery fitting or support selection here.
            assert compact and confirmation
            fitroot=ROOT/execution['path'];ec=load(fitroot/'config.resolved.json',execution['config_sha256'])
            assert ec['audit_opened'] is False and not ec['pilot'] and ec['support_budget']==cfg['readout_budget']
            erows=jsonl(checked(fitroot/'metrics.raw.jsonl',execution['metadata_sha256']));expected={(s,a,t,n) for s,a in queries for t in cfg['target_seeds'] if t!=s for n in ('short','long')}
            assert {(r['source_seed'],r['source_atom'],r['target_seed'],r['configuration']) for r in erows}==expected and len(erows)==len(expected)
            with np.load(checked(fitroot/'sparse_coefficients.npz',execution['coefficients_sha256']),allow_pickle=False) as saved:
                for r in erows:
                    s,a,t=r['source_seed'],r['source_atom'],r['target_seed'];method=r['configuration']+'_joint_sparse';key=r['array_key']
                    beta=np.array(saved[key]);intercept=np.array(saved[key+'_intercept'])
                    assert beta.shape==(base['num_latents'],2) and intercept.shape==(2,) and np.isfinite(beta).all() and np.isfinite(intercept).all()
                    assert np.flatnonzero(np.linalg.norm(beta,axis=1)>0).tolist()==r['shared_support'] and 0<len(r['shared_support'])<=cfg['readout_budget'] and r['selected']['converged']
                    assert ec['parent_run']==frozen['path'] and ec['parent_fits_sha256']==frozen['fits_sha256'] and ec['parent_coefficients_sha256']==frozen['coefficients_sha256']
                    compact_coefficients[s,a,t,method]=beta
            write(run/'execution_fits.json',erows);compact_methods+=('short_joint_sparse','long_joint_sparse')
            checks['sparse_interfaces_restored_no_fits']=True
        # All fitting/group selection has finished before task-target encoding and endpoints.
        rowids=np.array(sorted({seq*length+p for e in cases for seq,ps in ((e['sequence'],e['intervention_positions']),(e['donor_sequence'],e['donor_positions'])) for p in ps}));rowindex={int(r):i for i,r in enumerate(rowids)}
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SPARSIFY_DISABLE_TRITON='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        sys.path[:0]=[ev['sparsify_source_dir'],ev['sparsify_overlay_dir']]
        import torch
        import transformers
        from sparsify.sparse_coder import SparseCoder
        torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);z_eval={}
        for t,(item,asset) in saes.items():
            weight=checked(Path(item['path'])/'sae.safetensors',item['sha256']);checked(weight.parent/'cfg.json')
            sae=SparseCoder.load_from_disk(weight.parent,device='cuda:0').eval()
            with torch.no_grad():out=sae.encode(torch.tensor(np.asarray(eval_raw[rowids]),device='cuda:0'))
            z=np.zeros((len(rowids),base['num_latents']));np.add.at(z,(np.arange(len(rowids))[:,None],out.top_indices.cpu().numpy()),out.top_acts.cpu().numpy());z_eval[t]=z;del sae,out
        compiled=[];coordinates={};basisarrays={}
        for case_id,e in enumerate(cases):
            s,a=e['source_seed'],e['source_atom'];groups,b=query_data[s,a];rr=np.array(e['intervention_positions'])+e['sequence']*length;dd=np.array(e['donor_positions'])+e['donor_sequence']*length
            zd=dense('evalsource',s,'calibration',rr)-dense('evalsource',s,'calibration',dd)
            source=np.column_stack([zd[:,g]@(dec['source',s][g]@b) for g in groups]);coords={}
            def embed(value):
                result=np.zeros((length,2));result[e['intervention_positions']]=value;return result
            coords['source']=embed(source);coords['raw']=embed((np.asarray(eval_raw[rr],dtype=float)-np.asarray(eval_raw[dd],dtype=float))@coefficients[s,a,'raw'])
            for t in cfg['target_seeds']:
                if t!=s:
                    zdlong=z_eval[t][[rowindex[int(r)] for r in rr]]-z_eval[t][[rowindex[int(r)] for r in dd]]
                    coords[f'long_{t}']=embed(zdlong@coefficients[s,a,t])
                    if confirmation:coords[f'long_full_{t}']=coords[f'long_{t}']
                    if compact:
                        zdshort=dense('evalshort',t,'calibration',rr)-dense('evalshort',t,'calibration',dd)
                        for method in compact_methods:
                            coords[f'{method}_{t}']=embed((zdshort if method.startswith('short') else zdlong)@compact_coefficients[s,a,t,method])
            masked=np.zeros((length,base['hook_hidden_size']));masked[e['intervention_positions']]=eval_raw[rr]
            scale=family_scale(coords['source'],b,masked,cfg['maximum_source_hook_fraction'])
            oldscale=source_dose_scale(coords['source'].sum(axis=1)[:,None]*b,masked,cfg['maximum_source_hook_fraction'])
            compiled.append(dict(case_id=case_id,**e,common_family_scale=scale,old_full_scale=oldscale,masked_hook_norm=float(np.linalg.norm(masked))))
            basisarrays[f'basis_{case_id}']=b
            for name,c in coords.items():coordinates[f'case_{case_id}_{name}']=c
        if compact and (not confirmation or cfg.get('check_parent_coordinates',False)):
            assert compiled==parent_cases
            for key in parent_coords.files:
                value=basisarrays[key] if key.startswith('basis_') else coordinates[key]
                if not np.array_equal(value,parent_coords[key]):raise ValueError('Frozen source/full/raw coordinates changed')
            checks['parent_cases_dose_and_coordinates_exact']=True
        np.savez_compressed(run/'component_coordinates.npz',**coordinates,**basisarrays);write(run/'compiled_cases.json',compiled)
        write(run/'inputs.json',dict(inputs=inputs));timers['preparation_seconds']=time.perf_counter()-started
        if timers['preparation_seconds']>cfg['fit_budget_seconds']:raise TimeoutError('Preparation budget exceeded')
        checks['full_coefficient_replay']=True;checks['source_only_groups_before_target_encoding']=True
        print(json.dumps(dict(stage='COMPONENTS_FROZEN',cases=len(compiled),maps=len(coefficients),seconds=timers['preparation_seconds'])),flush=True)
        if cfg.get('preparation_only',False):
            checks['preparation_only_no_lm_forwards']=forwards==0
            env.update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
        else:
            loading=time.perf_counter();model=transformers.AutoModelForCausalLM.from_pretrained(base['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0')
            model.config.use_cache=False;timers['model_load_seconds']=time.perf_counter()-loading
            contract=HookPointContract(base['hook_module_path'],5,'resid_post',base['hook_hidden_size']);hook=model.get_submodule(base['hook_module_path'])
            def forward(batch,delta=None):
                nonlocal forwards
                captured={}
                def intervene(_m,_i,out):
                    h=extract_primary_hook_tensor(out,contract);captured['hook']=h.detach().cpu().numpy()
                    return replace_primary_hook_tensor(out,h-delta,contract) if delta is not None else None
                handle=hook.register_forward_hook(intervene)
                try:
                    with torch.no_grad():logits=model(batch,use_cache=False).logits[0].cpu().numpy()
                finally:handle.remove()
                forwards+=1;return logits,captured['hook'][0]
            numeric=time.perf_counter();noop=[];replay=[]
            for e in compiled:
                case_id=e['case_id'];b=basisarrays[f'basis_{case_id}'];scale=e['common_family_scale'];seq=e['sequence'];batch=torch.tensor(np.asarray(tokens[seq:seq+1],dtype=np.int64),device='cuda:0')
                baseline,live=forward(batch);zero,_=forward(batch,torch.zeros((1,length,base['hook_hidden_size']),device='cuda:0'))
                noop.append(float(np.max(np.abs(baseline-zero))));replay.append(float(np.linalg.norm(live-eval_raw[seq*length:(seq+1)*length])/max(np.linalg.norm(live),1e-20)))
                positions=endpoint_positions(tokens[seq],e['intervention_positions']);source_logits={};source_deltas={}
                def delta(name,component):
                    c=coordinates[f'case_{case_id}_{name}'];value=c.sum(axis=1) if component=='full' else c[:,0 if component=='A' else 1]
                    return value[:,None]*b*scale
                for component in ('full','A','B'):
                    d=delta('source',component);source_deltas[component]=d;source_logits[component],_=forward(batch,torch.tensor(d[None],dtype=torch.float32,device='cuda:0'))
                if execution:
                    methods=['source']+[f'{name}_joint_sparse_{t}' for t in cfg['target_seeds'] if t!=e['source_seed'] for name in ('short','long')]
                elif confirmation:
                    methods=['source','raw']+[f'{name}_{t}' for t in cfg['target_seeds'] if t!=e['source_seed']
                        for name in ('short_full','short_shared16','long_full','long_shared16','short_single','long_single')]
                else:
                    methods=(['source']+[f'{name}_{t}' for t in cfg['target_seeds'] if t!=e['source_seed'] for name in ('short_full','short_shared16','long_shared16','short_single','long_single')]
                        if compact else ['source','raw']+[f'long_{t}' for t in cfg['target_seeds'] if t!=e['source_seed']])
                for name in methods:
                    for component in ('full','A','B'):
                        d=delta(name,component);out=source_logits[component] if name=='source' else forward(batch,torch.tensor(d[None],dtype=torch.float32,device='cuda:0'))[0]
                        row=dict(**e,method=name,component=component,hook=compare(source_deltas[component],d),
                            source_hook_fraction=float(np.linalg.norm(source_deltas[component])/e['masked_hook_norm']),candidate_hook_fraction=float(np.linalg.norm(d)/e['masked_hook_norm']),
                            probability_endpoints={scope:probability_metrics(baseline,source_logits[component],out,tokens[seq],ps) for scope,ps in positions.items()})
                        records.append(row)
                        with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as sink:sink.write(json.dumps(row,sort_keys=True)+'\n')
                if time.perf_counter()-numeric>cfg['causal_budget_seconds']:raise TimeoutError('Causal budget exceeded')
            timers['causal_seconds']=time.perf_counter()-numeric
            # Same cached batch4 versus live batch1 tolerance as the original F4 consumer.
            checks.update(noop_exact=max(noop)==0,cached_hook_replay=max(replay)<=1e-4,expected_forwards=forwards==cfg['expected_forwards'],all_cases=len(compiled)==cfg['expected_cases'],all_rows=len(records)==cfg.get('expected_rows',18*cfg['expected_cases']))
            write(run/'numerical_checks.json',dict(noop=noop,cached_hook_replay=replay))
            env.update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    env.update(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),numpy=np.__version__)
    write(run/'environment.json',env);write(run/'inputs.json',dict(inputs=inputs))
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,model_forwards=forwards,rows=len(records),wall_seconds=time.perf_counter()-started,timings=timers,
        metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_f4_source_components.py',generator_script_sha256=sha256(Path(__file__)),scope_limit=cfg['scope'])
    write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    validation=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=validation.ok,errors=list(validation.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=validation.ok,contract_errors=list(validation.errors))),flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__':raise SystemExit(main())
