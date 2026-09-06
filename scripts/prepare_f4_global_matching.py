"""Full-dictionary one-to-one matching, lifted to fixed source-group operations."""
import argparse
import copy
import itertools
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from run_f4_source_reference_causal import ROOT,write,jsonl,sha256
from run_r011s1_raw_hook_asset import aggregate,entry
from ccad.artifacts import validate_run_directory

METHODS=['global_matching_geometric','global_matching_pair_calibrated']


def full_assignment(source,target):
    source=np.asarray(source,dtype=float);target=np.asarray(target,dtype=float)
    if source.shape!=target.shape or source.ndim!=2 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Expected equal finite full dictionaries')
    ns=np.linalg.norm(source,axis=1);nt=np.linalg.norm(target,axis=1)
    denom=ns[:,None]*nt[None,:]
    cosine=np.divide(source@target.T,denom,out=np.zeros_like(denom),where=denom>0)
    rows,cols=linear_sum_assignment(np.abs(cosine),maximize=True)
    assert np.array_equal(rows,np.arange(len(source))) and len(set(cols))==len(target)
    signed=cosine[rows,cols]
    scale=np.divide(np.sign(signed)*nt[cols],ns,out=np.zeros_like(ns),where=ns>0)
    return cols,scale,dict(mean_absolute_cosine=float(np.abs(signed).mean()),
                           minimum_absolute_cosine=float(np.abs(signed).min()),
                           negative_matches=int(np.sum(signed<0)),zero_norm_source=int(np.sum(ns==0)),
                           zero_norm_target=int(np.sum(nt==0)),zero_cosine_matches=int(np.sum(signed==0)))


def pair_calibration(source,target,weights,ridge_fraction):
    """Independent scalar regressions; no cross-pair mixing or outcome fitting."""
    source=np.asarray(source,dtype=float);target=np.asarray(target,dtype=float);weights=np.asarray(weights,dtype=float)
    if source.shape!=target.shape or source.ndim!=2 or weights.shape!=(len(source),) or np.any(weights<0) or weights.sum()<=0 or ridge_fraction<0:
        raise ValueError('Invalid calibration input')
    if not all(np.isfinite(x).all() for x in [source,target,weights]):raise ValueError('Nonfinite calibration')
    weights=weights/weights.sum();sm=weights@source;tm=weights@target
    sc=source-sm;tc=target-tm
    variance=np.sum(weights[:,None]*tc**2,axis=0)
    covariance=np.sum(weights[:,None]*sc*tc,axis=0)
    slopes=np.divide(covariance,variance*(1+ridge_fraction),out=np.zeros_like(variance),where=variance>0)
    return slopes,dict(source_weighted_mean=sm.tolist(),target_weighted_mean=tm.tolist(),
                       target_variance=variance.tolist(),covariance=covariance.tolist(),
                       zero_target_variance=int(np.sum(variance==0)),ridge_fraction=ridge_fraction,
                       pair_weighted_residual=np.sum(weights[:,None]*(sc-tc*slopes)**2,axis=0).tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();write(run/'config.resolved.json',cfg);files=[]
    for rel in ['scripts/prepare_f4_global_matching.py','scripts/run_f4_source_reference_causal.py',
                'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
        p=ROOT/rel;dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        files.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=files,aggregate_sha256=aggregate(files),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.full.matching.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],
          milestone='M4',evidence_level='real_sae_development_comparator_preparation',started_utc=datetime.now(timezone.utc).isoformat(),
          project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(files),
          source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='original independent mean; shifts cancel in conditional covariance and donor differences',
          threshold_source_split='none; fixed source query/case rules',statistics_unit='seed pair and source query; not independent directions',
          device='CPU',seeds=cfg['seeds'],resource_lease='cpu-heavy resource_manager.run',resource_lease_reason=cfg['budget']))
    write(run/'status.json',dict(status='RUNNING'));(run/'stderr.log').write_text('');records=[];inputs=[];cache={};checks={};error=None
    def checked(path,digest=None,role='input'):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        if path not in cache:
            cache[path]=sha256(path);inputs.append(entry(path,'CCAD existing asset',role))
        if digest is not None and cache[path].lower()!=digest.lower():raise ValueError('Input identity changed: '+str(path))
        return path
    try:
        checked(args.config,role='frozen_config')
        base=json.loads(checked(cfg['asset_config']).read_text())
        manifest=json.loads(checked(Path(base['bulk_asset_dir'])/'asset_manifest.json',base['asset_manifest_sha256']).read_text())
        n=base['num_latents'];dec={}
        for seed in cfg['seeds']:
            d=next(x for x in manifest['decoders'] if x['seed']==seed)
            dec[seed]=np.asarray(np.memmap(checked(d['path'],d['sha256'],'full_decoder'),dtype='<f4',mode='r',shape=tuple(d['shape'])),dtype=float)
            assert dec[seed].shape[0]==n
        assignments={};scales={};matching_arrays={}
        for i,(s,t) in enumerate(itertools.combinations(cfg['seeds'],2)):
            pstart=time.perf_counter();mapping,scale,stats=full_assignment(dec[s],dec[t]);elapsed=time.perf_counter()-pstart
            inverse=np.empty(n,dtype=int);inverse[mapping]=np.arange(n)
            ns=np.linalg.norm(dec[s],axis=1);nt=np.linalg.norm(dec[t],axis=1)
            inverse_scale=np.divide(np.sign(scale[inverse])*ns[inverse],nt,out=np.zeros_like(nt),where=nt>0)
            for a,b,mp,sc in [(s,t,mapping,scale),(t,s,inverse,inverse_scale)]:
                assignments[a,b]=mp;scales[a,b]=sc;matching_arrays[f's{a}_t{b}_assignment']=mp;matching_arrays[f's{a}_t{b}_geometric_scale']=sc
            records.append(dict(kind='full_dictionary_assignment',source_seed=s,target_seed=t,width=n,wall_seconds=elapsed,**stats))
            print(json.dumps(records[-1]),flush=True)
            np.savez_compressed(run/'full_assignments.npz',**matching_arrays)
            if i==0 and elapsed>cfg['first_pair_budget_seconds']:raise RuntimeError('First pair exceeded its bounded budget; completed assignment retained')
            if time.perf_counter()-start>cfg['total_cpu_budget_seconds']:raise RuntimeError('Preparation budget exceeded; completed assignments retained')
        sparse={}
        def open_sparse(tag,m,split,seeds):
            part=next(x for x in m['splits'] if x['split']==split)
            for seed in seeds:
                arrays={}
                for item in part['files']:
                    if item['seed']==seed:
                        arrays[item['dtype']]=np.memmap(checked(item['path'],item['sha256'],f'{tag}_{split}_codes'),dtype='<u2' if item['dtype']=='uint16' else '<f4',mode='r',shape=tuple(item['shape']))
                sparse[tag,seed]=(arrays['uint16'],arrays['float32'])
        def dense(tag,seed,rows):
            ix,act=sparse[tag,seed];rows=np.asarray(rows,dtype=int);z=np.zeros((len(rows),n))
            np.add.at(z,(np.arange(len(rows))[:,None],ix[rows]),act[rows]);return z
        open_sparse('discovery',manifest,'discovery',cfg['seeds'])
        panels={};families={};coordinate_arrays={};coordinate_rows=[];coefficients={};details=[]
        for panel in cfg['panels']:
            ref=json.loads(checked(panel['reference_config'],panel['reference_config_sha256']).read_text())
            payload=json.loads(checked(ref['case_replay']['path'],ref['case_replay']['sha256']).read_text())
            family=ref['saved_atom_families'][0]
            old=json.loads(checked(family['config_path'],family['config_sha256']).read_text())
            assert old['asset_manifest_sha256']==base['asset_manifest_sha256']
            fitted=json.loads(checked(family['fit_path'],family['fit_sha256']).read_text())
            assert fitted['fit_split']=='discovery' and fitted['calibration_used_for_fit'] is False
            for r in fitted['fits']:families[r['source_seed'],r['source_atom'],r['target_seed']]=r
            panels[panel['label']]=(ref,payload)
        ref0=next(iter(panels.values()))[0]
        factors=np.load(checked(ref0['factors_path'],ref0['factors_sha256']),allow_pickle=False)
        fi={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(factors['source_seed'],factors['source_atom'],factors['target_seed']))}
        surface={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in jsonl(checked(ref0['surface_path'],ref0['surface_sha256'])) if r['rank']==1 and r['query_role']=='anchor'}
        query_keys=sorted(families)
        for s,a,t in query_keys:
            r=families[s,a,t];ids=np.asarray(surface[s,a,t]['source_candidate_ids'],dtype=int);matched=assignments[s,t][ids]
            rows=np.asarray(r['discovery_rows'],dtype=int);weights=np.asarray(r['discovery_weights'],dtype=float)
            b=factors['source_basis'][fi[s,a,t],:,:1].astype(float)
            source=dense('discovery',s,rows)[:,ids];target=dense('discovery',t,rows)[:,matched]
            slopes,diag=pair_calibration(source,target,weights,cfg['ridge_fraction'])
            source_weights=(dec[s][ids]@b)[:,0]
            geometric=source_weights*scales[s,t][ids];calibrated=source_weights*slopes
            key=f's{s}_a{a}_t{t}';coefficients[key+'_matched_atoms']=matched
            coefficients[key+'_geometric']=geometric;coefficients[key+'_pair_calibrated']=calibrated
            families[s,a,t]=dict(matched=matched,geometric=geometric,pair_calibrated=calibrated)
            record=dict(kind='query_group_readout',source_seed=s,source_atom=a,target_seed=t,source_candidate_ids=ids.tolist(),
                        matched_target_ids=matched.tolist(),discovery_rows=rows.tolist(),discovery_weights=weights.tolist(),
                        source_weights=source_weights.tolist(),geometric_coefficients=geometric.tolist(),
                        calibrated_coefficients=calibrated.tolist(),pair_slopes=slopes.tolist(),array_key=key,**diag)
            details.append(record)
            records.append(dict(kind='query_group_readout',source_seed=s,source_atom=a,target_seed=t,support=len(ids),
                                calibrated_nonzero=int(np.count_nonzero(calibrated)),source_group_discovery_variation=float(np.var(source@source_weights)),
                                zero_target_variance=diag['zero_target_variance']))
        np.savez_compressed(run/'group_coefficients.npz',**coefficients);write(run/'group_fit_details.json',details)
        evmanifest=json.loads(checked(Path(ref0['bulk_asset_dir'])/'asset_manifest.json',ref0['asset_manifest_sha256']).read_text())
        for d in evmanifest['decoders']:
            original=next(x for x in manifest['decoders'] if x['seed']==d['seed'])
            assert d['sha256']==original['sha256']
        open_sparse('evaluation',evmanifest,'calibration',cfg['seeds'])
        for label,(ref,payload) in panels.items():
            assert all(ref[k]==ref0[k] for k in ['factors_sha256','surface_sha256','asset_manifest_sha256','sequence_records_sha256'])
            cases=[]
            for choice in payload['choices']:
                e=choice['entry']
                if e is None:continue
                assert choice['source_scope']['supported'];s=choice['source_seed'];a=choice['source_atom'];length=base['context_length']
                cases.append(dict(source_seed=s,source_atom=a,**{k:e[k] for k in ['condition','sequence','donor_sequence']}))
                rr=np.asarray(e['intervention_positions'])+e['sequence']*length;dd=np.asarray(e['donor_positions'])+e['donor_sequence']*length
                for t in cfg['seeds']:
                    if t==s:continue
                    family=families[s,a,t];difference=dense('evaluation',t,rr)-dense('evaluation',t,dd)
                    for name in ['geometric','pair_calibrated']:
                        coord=np.zeros((length,1));coord[e['intervention_positions'],0]=difference[:,family['matched']]@family[name]
                        key=f'coordinate_{len(coordinate_arrays)}';coordinate_arrays[key]=coord
                        coordinate_rows.append(dict(panel=label,source_seed=s,source_atom=a,target_seed=t,method='global_matching_'+name,array_key=key,**e))
            assert len(cases)==cfg['expected_cases'][label]
            ref=copy.deepcopy(ref);ref.pop('saved_atom_families',None);ref.pop('frozen_rejected_requests',None)
            ref.update(run_id=f'F4_global_matching_{label}_v1_20260906',methods=METHODS,
                       expected_evaluated_cases=len(cases),frozen_evaluated_requests=cases,
                       selection='All original supported class-matched cases, selected/rejected labels unchanged; no new choice.',
                       intervention='Full-dictionary one-to-one source-group readout in unchanged source basis; geometric and pair-calibrated variants, common source cap0.1.',
                       evidence_level='exposed_document_global_one_to_one_comparator_development',scope_limit=cfg['scope_limit'],
                       budget=f'{len(cases)} cases*(3+4*2)={len(cases)*11} forwards; combined264, no new fit/data/audit during application.')
            ref['case_replay'].update(source_selection_scope='all_supported',selected_only=False,export_details=False)
            panels[label]=(ref,payload)
        arraypath=run/'candidate_coordinates.npz';np.savez_compressed(arraypath,**coordinate_arrays)
        output_configs=[]
        for label,(ref,payload) in panels.items():
            indexpath=run/f'{label}_candidate_index.json'
            write(indexpath,dict(rows=[r for r in coordinate_rows if r['panel']==label],factors_sha256=ref['factors_sha256'],
                  surface_sha256=ref['surface_sha256'],sequence_records_sha256=ref['sequence_records_sha256'],
                  case_selection_sha256=ref['case_replay']['sha256'],scale='unscaled_source_basis_coordinates'))
            ref['saved_candidate_coordinates']=dict(index_path=str(indexpath),index_sha256=sha256(indexpath),arrays_path=str(arraypath),arrays_sha256=sha256(arraypath),methods=METHODS)
            path=ROOT/f'configs/f4_global_matching_{label}_v1.json'
            if path.exists():raise FileExistsError(path)
            write(path,ref);output_configs.append(str(path))
        checks=dict(full_bijections=len(assignments)==20,query_group_count=len(details)==64,
                    expected_coordinates=len(coordinate_arrays)==24*4*2,finite_coordinates=all(np.isfinite(x).all() for x in coordinate_arrays.values()),
                    original_case_labels_preserved=True,audit_closed=True,no_model_forwards=True)
    except Exception:
        error=traceback.format_exc();(run/'stderr.log').write_text(error,encoding='utf-8')
    write(run/'inputs.json',dict(inputs=inputs))
    (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records),encoding='utf-8')
    summary=dict(status='PASS' if error is None and all(checks.values()) else 'FAIL',error=error,checks=checks,
                 rows=len(records),model_forwards=0,wall_seconds=time.perf_counter()-start,
                 metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope_limit=cfg['scope_limit'])
    write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary)
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
          platform=platform.platform(),torch='not_imported',transformers='not_imported',cuda='not_used',gpu='not_used',solver='SciPy modified Jonker-Volgenant full assignment'))
    write(run/'status.json',dict(status=summary['status'],error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    valid=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=valid.ok,errors=list(valid.errors)))
    print(json.dumps(dict(run=run.name,status=summary['status'],contract_ok=valid.ok,errors=list(valid.errors),wall_seconds=summary['wall_seconds'],error=error)))
    return int(summary['status']!='PASS' or not valid.ok)


if __name__=='__main__':raise SystemExit(main())
