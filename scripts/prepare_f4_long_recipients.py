"""Fixed-source natural-text transfer into existing longer-trained SAE recipients."""
import argparse
import copy
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from run_f4_source_reference_causal import (ROOT, np, write, jsonl, sha256,
    fixed_support_ridge, best_single_atom, readout_atom_order)
from run_r011s1_raw_hook_asset import entry, aggregate
from ccad.artifacts import validate_run_directory


def restore_fits(records, arrays, width, budget):
    """Restore signed coefficients and frozen supports; never fit or rank."""
    fits={}
    for r in records:
        key=(r['configuration'],r['source_seed'],r['source_atom'],r['target_seed'])
        beta=np.asarray(arrays[r['array_key']],dtype=float)
        keep=np.asarray(r['top_atoms'],dtype=int);atom=r['single_atom']
        if (key in fits or beta.shape!=(width,) or not np.isfinite(beta).all()
                or len(keep)!=budget or len(set(keep.tolist()))!=budget
                or np.any(keep<0) or np.any(keep>=width)
                or not 0<=atom['atom']<width or not np.isfinite(atom['coefficient'])):
            raise ValueError('Invalid or duplicated frozen recipient fit')
        fits[key]=dict(beta=beta.copy(),keep=keep.copy(),atom=copy.deepcopy(atom))
    return fits


def apply_recipient_fit(z, fit):
    keep=fit['keep'];atom=fit['atom']
    return {'target':z@fit['beta'], 'top16':z[:,keep]@fit['beta'][keep],
            'single_atom':z[:,atom['atom']]*atom['coefficient']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    started=time.perf_counter();now=datetime.now(timezone.utc).isoformat();write(run/'config.resolved.json',cfg)
    source_files=['scripts/prepare_f4_long_recipients.py','scripts/run_f4_source_reference_causal.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/hook_transport.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']
    code=[]
    for rel in source_files:
        path=ROOT/rel;dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        code.append(dict(path=rel,sha256=sha256(path),bytes=path.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.long.recipient.fit.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose=cfg.get('purpose','Fixed-source recipient representation/complexity development'),milestone='M4',evidence_level=cfg.get('evidence_level','exposed_development_cross_configuration'),
        started_utc=now,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),
        source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original mean',
        threshold_source_split='fixed source cases; training-document exclusions before new target encoding',statistics_unit='query/document/seed dependencies',
        device='cpu and cuda:0',seeds=cfg['target_seeds'],resource_lease='cpu-heavy -> gpu-0 resource_manager.run',resource_lease_reason='small ridge fits and cached-hook SAE encoding; zero LM forward'))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now));inputs=[];records=[];checks={};error=None;env={};output_configs=[]
    def checked(path,expected=None,role='input'):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        digest=sha256(path)
        if expected and digest!=expected:raise ValueError('Input identity changed: '+str(path))
        inputs.append(entry(path,'CCAD existing asset',role));return path
    try:
        checked(args.config,role='frozen_config');spec=checked('.aris/compute/local-r006b1-env-spec.json','3129a184d787ae9be38ac6d8d97dbf5087e5c838c112473fe45f3862064bb60f','environment')
        base=json.loads(checked(cfg['source_asset_config']).read_text());evasset=json.loads(checked(cfg['evaluation_asset_config']).read_text())
        configs={'short':base}
        for r in cfg['long_configurations']:
            c=json.loads(checked(r['asset_config']).read_text());c['bulk_asset_dir']=c['bulk_output_dir'];c['asset_manifest_sha256']=r['asset_manifest_sha256'];configs[r['name']]=c
            if any(c[k]!=evasset[k] for k in ('model_revision','hook_module_path','num_latents','hook_hidden_size','context_length')):raise ValueError('Model/hook shape mismatch')
            if c['token_manifest_sha256']!=base['token_manifest_sha256']:raise ValueError('Paired corpus differs')
        training=json.loads(checked(cfg['long_training_documents']).read_text())['documents']
        train_ids={r['document_id'] for r in training};train_hashes={r['text_sha256'] for r in training}
        paired=jsonl(checked(cfg['original_paired_documents']));evaluation=jsonl(checked(cfg['evaluation_documents']))
        overlap=lambda rows:[r['document_id'] for r in rows if r['document_id'] in train_ids or r['text_sha256'] in train_hashes]
        if overlap(paired):raise ValueError('Paired fit data overlap long training')
        excluded=set(overlap(evaluation));panels={};case_rows=[];exclusions=[];active_requested=0;requested=0
        saved_spec=cfg.get('saved_recipient_fit')
        if saved_spec and cfg.get('require_no_evaluation_overlap') and excluded:raise ValueError('Fresh corpus overlaps long training')
        for label in cfg['panels']:
            ref=json.loads(checked(cfg['reference_config_template'].format(panel=label)).read_text())
            selection_spec=cfg.get('source_selections',{}).get(label,ref.get('case_replay'))
            selection=copy.deepcopy(json.loads(checked(selection_spec['path'],selection_spec['sha256']).read_text()))
            requested+=len(selection['choices'])
            for choice in selection['choices']:
                e=choice['entry'];bad=sorted(set(e['document_ids']+e['donor_document_ids'])&excluded) if e else []
                active=bool(e and choice['source_scope']['selected'])
                active_requested+=int(active)
                if active and bad:
                    exclusions.append(dict(panel=label,source_seed=choice['source_seed'],source_atom=choice['source_atom'],condition=choice['condition'],entry=e,excluded_document_ids=bad))
                    choice['excluded_matched_entry']=e;choice['entry']=None;choice['matching_status']='EXCLUDED_LONG_SAE_TRAINING_DOCUMENT'
                elif active:
                    targets=[t for t in cfg['target_seeds'] if t!=choice['source_seed']]
                    case_rows.append(dict(panel=label,source_seed=choice['source_seed'],source_atom=choice['source_atom'],targets=targets,**e))
            assert sum(r['panel']==label for r in case_rows)==cfg['expected_remaining_cases'][label]
            selection['long_training_exclusion_rule']=cfg['exclusion_rule'];sp=run/f'{label}_case_selection.json';write(sp,selection);panels[label]=(ref,sp)
        write(run/'case_exclusions.json',dict(total_eval_documents=len(evaluation),long_training_overlap_documents=sorted(excluded),excluded_cases=exclusions,retained_cases=case_rows,paired_overlap=[],rule=cfg['exclusion_rule']))
        queries=sorted({(r['source_seed'],r['source_atom']) for r in case_rows});ref0=next(iter(panels.values()))[0]
        f=np.load(checked(ref0['factors_path'],ref0['factors_sha256']),allow_pickle=False)
        fi={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(f['source_seed'],f['source_atom'],f['target_seed']))}
        surface={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in jsonl(checked(ref0['surface_path'],ref0['surface_sha256'])) if r['rank']==1 and r['query_role']=='anchor'}
        families={}
        for ref,_ in panels.values():
            spec=ref['readout_ablation']['saved_readout']
            for r in json.loads(checked(spec['path'],spec['sha256']).read_text())['families']:families[r['source_seed'],r['source_atom'],r['target_seed']]=r
        manifests={};dec={};means={};sparse={}
        for name,c in ([] if saved_spec else configs.items()):
            manifest=json.loads(checked(Path(c['bulk_asset_dir'])/'asset_manifest.json',c['asset_manifest_sha256']).read_text());manifests[name]=manifest
            seeds=sorted(set(cfg['target_seeds'])|{s for s,a in queries}) if name=='short' else cfg['target_seeds']
            for seed in seeds:
                d=next(r for r in manifest['decoders'] if r['seed']==seed)
                dec[name,seed]=np.asarray(np.memmap(checked(d['path'],d['sha256']),dtype='<f4',mode='r',shape=tuple(d['shape'])),dtype=float)
                for split in ('mean','discovery'):
                    sm=next(r for r in manifest['splits'] if r['split']==split);parts={}
                    for item in sm['files']:
                        if item['seed']==seed:
                            parts[item['dtype']]=np.memmap(checked(item['path'],item['sha256']),dtype='<u2' if item['dtype']=='uint16' else '<f4',mode='r',shape=tuple(item['shape']))
                    sparse[name,seed,split]=(parts['uint16'],parts['float32'])
                ii,aa=sparse[name,seed,'mean'];means[name,seed]=np.bincount(ii.ravel(),weights=aa.ravel(),minlength=c['num_latents'])/len(ii)
        def dense(name,seed,rows):
            ii,aa=sparse[name,seed,'discovery'];z=np.zeros((len(rows),configs[name]['num_latents']));np.add.at(z,(np.arange(len(rows))[:,None],ii[rows]),aa[rows]);return z
        fits={};fit_arrays={};replay=[]
        if saved_spec:
            records=jsonl(checked(saved_spec['metadata_path'],saved_spec['metadata_sha256'],'frozen_fit_metadata'))
            with np.load(checked(saved_spec['coefficients_path'],saved_spec['coefficients_sha256'],'frozen_coefficients'),allow_pickle=False) as saved_arrays:
                fits=restore_fits(records,saved_arrays,base['num_latents'],cfg['readout_budget'])
                fit_arrays={r['array_key']:fits[r['configuration'],r['source_seed'],r['source_atom'],r['target_seed']]['beta'] for r in records}
            if any((name,s,a,t) not in fits for s,a in queries for t in cfg['target_seeds'] if t!=s for name in configs):raise ValueError('Missing frozen fit for requested source/target')
            checks['saved_fit_identity_and_support']=True
        for s,a in ([] if saved_spec else queries):
            targets=[t for t in cfg['target_seeds'] if t!=s];first=targets[0];family=families[s,a,first]
            rr=np.array(family['discovery_rows'],dtype=int);ww=np.array(family['discovery_weights']);b=f['source_basis'][fi[s,a,first],:,:1].astype(float)
            ids=surface[s,a,first]['source_candidate_ids'];y=((dense('short',s,rr)[:,ids]-means['short',s][ids])@dec['short',s][ids]@b)[:,0]
            for t in targets:
                for name in configs:
                    z=dense(name,t,rr);x=(z-means[name,t])@dec[name,t]
                    w,diag=fixed_support_ridge(x,y,ww,cfg['ridge_fraction'],center=False)
                    if name=='short':
                        old=f['query_target'][fi[s,a,t],:,0].astype(float);rel=float(np.linalg.norm(w-old)/max(np.linalg.norm(old),1e-20));replay.append(rel)
                        if rel>1e-5:raise ValueError('Short full map does not replay original')
                        w=old # exact original coefficients for the common baseline
                    beta=dec[name,t]@w;order,energy=readout_atom_order(z,beta,ww);keep=order[:cfg['readout_budget']]
                    atom=best_single_atom(z-means[name,t],y,ww,cfg['ridge_fraction'],conditional_variation=True)
                    fits[name,s,a,t]=dict(beta=beta,keep=keep,atom=atom)
                    arrkey=f'{name}_{s}_{a}_{t}';fit_arrays[arrkey]=beta
                    records.append(dict(configuration=name,source_seed=s,source_atom=a,target_seed=t,discovery_rows=rr.tolist(),discovery_weights=ww.tolist(),source_candidate_ids=ids,full_fit=diag,top_atoms=keep.tolist(),total_readout_term_energy=float(energy.sum()),top_readout_term_energy=float(energy[keep].sum()),single_atom=atom,array_key=arrkey))
                if time.perf_counter()-started>cfg['budget_seconds']:raise TimeoutError('Preparation budget exceeded')
        np.savez_compressed(run/'fit_coefficients.npz',**fit_arrays)
        # Only encode hook rows that appear in the fixed retained interventions.
        rm=json.loads(checked(Path(ref0['raw_hook_asset_dir'])/'raw_hook_manifest.json',ref0['raw_hook_manifest_sha256']).read_text())
        rawmeta=next(r for r in rm['splits'] if r['split']=='calibration');h=np.memmap(checked(rawmeta['path'],rawmeta['sha256']),dtype='<f4',mode='r',shape=tuple(rawmeta['shape']))
        length=base['context_length'];rowids=np.array(sorted({seq*length+p for e in case_rows for seq,positions in ((e['sequence'],e['intervention_positions']),(e['donor_sequence'],e['donor_positions'])) for p in positions}),dtype=int)
        rowindex={int(r):i for i,r in enumerate(rowids)};evcodes={};encoding=[]
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SPARSIFY_DISABLE_TRITON='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        sys.path[:0]=[evasset['sparsify_source_dir'],evasset['sparsify_overlay_dir']]
        import torch
        import transformers
        from sparsify.sparse_coder import SparseCoder
        torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True)
        numeric=time.perf_counter();evmanifest=json.loads(checked(Path(evasset['bulk_output_dir'])/'asset_manifest.json',ref0['asset_manifest_sha256']).read_text())
        for name,c in configs.items():
            saecfg=evasset if name=='short' else c
            for t in cfg['target_seeds']:
                item=next(r for r in saecfg['saes'] if r['seed']==t);weight=checked(Path(item['path'])/'sae.safetensors',item['sha256'],'sae_checkpoint');checked(weight.parent/'cfg.json',role='sae_config')
                sae=SparseCoder.load_from_disk(weight.parent,device='cuda:0').eval();z=np.zeros((len(rowids),base['num_latents']))
                for begin in range(0,len(rowids),cfg['encoding_batch_rows']):
                    ix=rowids[begin:begin+cfg['encoding_batch_rows']]
                    with torch.no_grad():out=sae.encode(torch.tensor(np.asarray(h[ix]),device='cuda:0'))
                    ii=out.top_indices.cpu().numpy();aa=out.top_acts.cpu().numpy();np.add.at(z,(np.arange(begin,begin+len(ix))[:,None],ii),aa)
                if name=='short':
                    sm=next(r for r in evmanifest['splits'] if r['split']=='calibration');parts={}
                    for item in sm['files']:
                        if item['seed']==t:parts[item['dtype']]=np.memmap(checked(item['path'],item['sha256']),dtype='<u2' if item['dtype']=='uint16' else '<f4',mode='r',shape=tuple(item['shape']))
                    oldz=np.zeros_like(z);np.add.at(oldz,(np.arange(len(rowids))[:,None],parts['uint16'][rowids]),parts['float32'][rowids])
                    rel=float(np.linalg.norm(z-oldz)/max(np.linalg.norm(oldz),1e-20));checks[f'short_encoding_replay_{t}']=rel<1e-5
                    z=oldz # exact original cached codes for baseline scalar predictor
                else:rel=None
                evcodes[name,t]=z;encoding.append(dict(configuration=name,seed=t,rows=len(rowids),nonzero_l0=float(np.count_nonzero(z)/len(z)) if len(z) else None,short_replay_relative=rel))
                del sae
        newmethods=['short_single_atom']+[f'{name}_{method}' for name in configs if name!='short' for method in ('target','top16','single_atom')]
        arrays={};indexrows=[]
        for e in case_rows:
            s,a=e['source_seed'],e['source_atom'];rr=[rowindex[e['sequence']*length+p] for p in e['intervention_positions']];dd=[rowindex[e['donor_sequence']*length+p] for p in e['donor_positions']]
            for t in e['targets']:
                for name in configs:
                    z=evcodes[name,t][rr]-evcodes[name,t][dd];values=apply_recipient_fit(z,fits[name,s,a,t])
                    for method,value in values.items():
                        if name=='short' and method!='single_atom':continue
                        key=f'candidate_{len(arrays)}';coord=np.zeros((length,1));coord[e['intervention_positions'],0]=value;arrays[key]=coord
                        indexrows.append(dict(e,target_seed=t,method=f'{name}_{method}',array_key=key))
        arraypath=run/'candidate_coordinates.npz';np.savez_compressed(arraypath,**arrays)
        for label,(ref,sp) in panels.items():
            indexpath=run/f'{label}_candidate_index.json';write(indexpath,dict(rows=[r for r in indexrows if r['panel']==label],factors_sha256=ref['factors_sha256'],surface_sha256=ref['surface_sha256'],sequence_records_sha256=ref['sequence_records_sha256'],case_selection_sha256=sha256(sp),scale='unscaled_source_basis_coordinates'))
            new=copy.deepcopy(ref);new.pop('source_preparation_only',None);new['run_id']=cfg.get('causal_run_template','F4_long_recipient_{panel}_dev_v1_20260905').format(panel=label);new['case_replay']=dict(path=str(sp),sha256=sha256(sp),selected_only=True,export_details=False)
            new['target_seed_subset']=cfg['target_seeds'];new['methods']=['target','raw','readout_top16']+newmethods;new['expected_evaluated_cases']=cfg['expected_remaining_cases'][label]
            new['saved_candidate_coordinates']=dict(index_path=str(indexpath),index_sha256=sha256(indexpath),arrays_path=str(arraypath),arrays_sha256=sha256(arraypath),methods=newmethods)
            new['scope_limit']=cfg['scope'];new['evidence_level']=cfg.get('evidence_level','exposed_development_cross_configuration');new['budget']=cfg['budget'];new['intervention']='Original source reference/common dose; only recipient SAE and its fit/readout vary. Long training exclusions frozen before target encoding.'
            new['resource_lease']='cpu-heavy -> gpu-0 resource_manager.run'
            if cfg.get('probability_endpoints'):new['probability_endpoints']=cfg['probability_endpoints']
            dest=run/f'{label}_causal_config.json';write(dest,new);output_configs.append(str(dest))
        write(run/'encoding_summary.json',dict(rows=encoding,sae_encoding_seconds=time.perf_counter()-numeric,unique_hook_rows=rowids.tolist(),model_forwards=0))
        checks.update(training_fit_disjoint=True,training_evaluation_disjoint=True,all_cases_accounted=len(case_rows)+len(exclusions)==active_requested,finite_candidates=all(np.isfinite(x).all() for x in arrays.values()))
        if not saved_spec:checks['all_short_map_replay']=max(replay,default=0)<1e-5
        if cfg.get('expected_requested_conditions') is not None:checks['all_requests_retained']=requested==cfg['expected_requested_conditions']
        if cfg.get('coordinate_replay_witness'):
            witness=cfg['coordinate_replay_witness']
            with np.load(checked(witness['path'],witness['sha256'],'old_coordinate_replay'),allow_pickle=False) as old:
                checks['old_coordinates_exact_replay']=set(old.files)==set(arrays) and all(np.array_equal(old[k],v) for k,v in arrays.items())
        checks['within_preparation_budget']=time.perf_counter()-started<=cfg['budget_seconds']
        env.update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),sae=evasset['sparsify_commit'])
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    env.update(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),numpy=np.__version__)
    write(run/'inputs.json',dict(inputs=inputs));write(run/'environment.json',env)
    (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,fit_records=len(records),new_fits=0 if cfg.get('saved_recipient_fit') else len(records),model_forwards=0,wall_seconds=time.perf_counter()-started,causal_configs=output_configs,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/prepare_f4_long_recipients.py',generator_script_sha256=sha256(Path(__file__)))
    write(run/'metrics.summary.json',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));write(run/'stdout.log',summary)
    if not (run/'stderr.log').exists():(run/'stderr.log').write_text('')
    validation=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=validation.ok,errors=list(validation.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=validation.ok,contract_errors=list(validation.errors))),flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__':raise SystemExit(main())
