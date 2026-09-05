"""One operation-matched full FCC/raw fit using existing discovery and runner."""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from run_f4_source_reference_causal import ROOT, np, write, jsonl, sha256, fixed_support_ridge
from f4_case_details import token_class
from run_r011s1_raw_hook_asset import entry, aggregate
from ccad.artifacts import validate_run_directory


def operation_pairs(positive, scores, coordinate, classes, docsets, length, count=128):
    positive=np.asarray(positive,dtype=int)
    candidates=np.flatnonzero((scores>0)&~np.isin(np.arange(len(scores)),positive))
    candidates=candidates[np.lexsort((candidates,-scores[candidates]))[:count]]
    unused=set(candidates.tolist());recipients=[];donors=[];missing=[]
    for r in positive[:count]:
        possible=[d for d in unused if classes(int(r))==classes(d) and not (docsets[int(r)//length]&docsets[d//length])]
        if not possible:missing.append(int(r));continue
        d=max(possible,key=lambda d:(abs(coordinate[r]-coordinate[d]),-d))
        recipients.append(int(r));donors.append(d);unused.remove(d)
    return np.asarray(recipients,dtype=int),np.asarray(donors,dtype=int),missing,candidates


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    task=json.loads(args.config.read_text());parent=ROOT/task['parent_asset_config'];assetcfg=json.loads(parent.read_text())
    cfg={k:assetcfg[k] for k in ('bulk_asset_dir','asset_manifest_sha256','raw_hook_asset_dir','raw_hook_manifest_sha256','source_seeds','num_latents','hook_hidden_size','k','context_length')};cfg.update(task)
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();now=datetime.now(timezone.utc).isoformat()
    write(run/'config.resolved.json',cfg)
    sources=[Path(__file__),ROOT/'scripts/run_f4_source_reference_causal.py',ROOT/'scripts/f4_case_details.py',ROOT/'scripts/run_r011s1_raw_hook_asset.py',ROOT/'src/ccad/hook_transport.py',ROOT/'src/ccad/activation_contract.py',ROOT/'src/ccad/artifacts.py']
    code=[]
    for source in sources:
        rel=source.relative_to(ROOT).as_posix();dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        code.append(dict(path=rel,sha256=sha256(source),bytes=source.stat().st_size,snapshot_path=dest.relative_to(run).as_posix()))
    ch=aggregate(code);write(run/'code_hashes.json',dict(files=code,aggregate_sha256=ch,snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.full.operation.fit.v1',run_id=cfg['run_id'],run_parent='F4',purpose='Operation-matched full FCC/raw, fixed source and interface',milestone='M4',evidence_level='discovery_fit_for_exposed_development',started_utc=now,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=ch,source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='independent mean cancels in donor differences',threshold_source_split='no thresholds selected; existing source case rule',statistics_unit='query and seed dependence',device='cpu',seeds=cfg['source_seeds'],resource_lease='cpu-heavy resource_manager.run',resource_lease_reason='Bounded original discovery scan and dual ridge'))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now));inputs=[];records=[];checks={};error=None;outputs=[];transformers_version='not_imported'
    def checked(path,expected=None,role='input'):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        if expected and sha256(path)!=expected:raise ValueError('Input changed: '+str(path))
        inputs.append(entry(path,'CCAD saved data',role));return path
    try:
        checked(parent,role='parent_config');checked(args.config,role='frozen_config')
        fpath=checked(cfg['parent_factors'],cfg['parent_factors_sha256'],'parent_factors')
        old=np.load(fpath,allow_pickle=False);factors={k:old[k].copy() for k in old.files}
        index={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(old['source_seed'],old['source_atom'],old['target_seed']))}
        surface={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in jsonl(checked(cfg['surface'],cfg['surface_sha256'],'surface')) if r['rank']==1 and r['query_role']=='anchor'}
        asset=Path(cfg['bulk_asset_dir']);manifest=json.loads(checked(asset/'asset_manifest.json',cfg['asset_manifest_sha256'],'discovery_manifest').read_text());sp=next(r for r in manifest['splits'] if r['split']=='discovery');nt=sp['tokens'];length=cfg['context_length']
        rawmeta=json.loads(checked(Path(cfg['raw_hook_asset_dir'])/'raw_hook_manifest.json',cfg['raw_hook_manifest_sha256'],'raw_manifest').read_text());rs=next(r for r in rawmeta['splits'] if r['split']=='discovery');raw=np.memmap(checked(rs['path'],rs['sha256'],'discovery_raw'),dtype='<f4',mode='r',shape=tuple(rs['shape']))
        ids={};acts={};dec={}
        for seed in cfg['source_seeds']:
            for item in sp['files']:
                if item['seed']!=seed:continue
                checked(item['path'],item['sha256'],'discovery_codes')
            ids[seed]=np.memmap(asset/'discovery'/f'seed_{seed}'/'top_indices.uint16.bin',dtype='<u2',mode='r',shape=(nt,cfg['k']))
            acts[seed]=np.memmap(asset/'discovery'/f'seed_{seed}'/'top_acts.float32.bin',dtype='<f4',mode='r',shape=(nt,cfg['k']))
            dm=next(r for r in manifest['decoders'] if r['seed']==seed)
            dec[seed]=np.asarray(np.memmap(checked(dm['path'],dm['sha256'],'decoder'),dtype='<f4',mode='r',shape=(cfg['num_latents'],cfg['hook_hidden_size'])),dtype=float)
        corpus=ROOT/'runs'/cfg['original_corpus_run'];tm=json.loads(checked(corpus/'artifacts/token_manifest.json',role='original_tokens_manifest').read_text())['outputs']['discovery']
        tokens=np.memmap(checked(corpus/tm['path'],tm['sha256'],'discovery_tokens'),dtype='<u2',mode='r');assert len(tokens)==nt
        seqs=json.loads(checked(corpus/'artifacts/sequence_records.json',assetcfg['sequence_records_sha256'],'discovery_document_identity').read_text())['sequences'];docsets={r['sequence_index']:set(r['document_ids']) for r in seqs if r['split']=='discovery'}
        basecfg=json.loads((ROOT/'configs/f4_probability_confirmation_codes_v1.json').read_text())
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import transformers
        transformers_version=transformers.__version__
        tokenizer=transformers.AutoTokenizer.from_pretrained(basecfg['model_local_dir'],local_files_only=True)
        classes={}
        def cls(row):
            token=int(tokens[row])
            if token not in classes:classes[token]=token_class(tokenizer.decode([token]))
            return classes[token]
        def dense(seed,rows):
            z=np.zeros((len(rows),cfg['num_latents']));np.add.at(z,(np.arange(len(rows))[:,None],ids[seed][rows]),acts[seed][rows]);return z
        families={};panelcfgs={};selected_queries=set()
        for panel in cfg['panels']:
            ref=json.loads(checked(panel['reference_config'],role='evaluation_reference').read_text());panelcfgs[panel['name']]=ref
            cp=ref['case_replay'];choices=json.loads(checked(cp['path'],cp['sha256'],'source_case_selection').read_text())['choices']
            selected_queries.update((r['source_seed'],r['source_atom']) for r in choices if r['entry'] is not None and r['source_scope']['selected'])
            rp=ref['readout_ablation']['saved_readout'];payload=json.loads(checked(rp['path'],rp['sha256'],'saved_discovery_rows').read_text())
            assert payload['discovery_manifest_sha256']==cfg['asset_manifest_sha256']
            for r in payload['families']:families[r['source_seed'],r['source_atom'],r['target_seed']]=r
        changed=[]
        for s,a in sorted(selected_queries):
            t0=next(t for t in cfg['source_seeds'] if t!=s);family=families[s,a,t0];surf=surface[s,a,t0];b=old['source_basis'][index[s,a,t0],:,:1].astype(float)
            coef=np.zeros(cfg['num_latents']);coef[surf['source_candidate_ids']]=(dec[s][surf['source_candidate_ids']]@b)[:,0]
            coordinate=np.zeros(nt);scores=np.zeros(nt)
            for begin in range(0,nt,4096):
                sl=slice(begin,begin+4096);ii=ids[s][sl];aa=acts[s][sl]
                pos=np.sum(np.where(ii==a,aa,0),axis=1,dtype=float);neg=np.sum(np.where(np.isin(ii,surf['negative_source_atoms']),aa,0),axis=1,dtype=float)
                scores[sl]=neg**2/(1+pos**2);coordinate[sl]=np.sum(aa*coef[ii],axis=1,dtype=float)
            positive=np.asarray(family['discovery_rows'],dtype=int)
            rr,dd,missing,pool=operation_pairs(positive,scores,coordinate,cls,docsets,length,cfg['maximum_pairs'])
            if len(rr)<2:
                if cfg.get('unavailable_fit_policy')!='retain_parent_and_report':
                    raise ValueError('No usable difference fit for query '+str((s,a)))
                for t in cfg['source_seeds']:
                    if t==s:continue
                    records.append(dict(source_seed=s,source_atom=a,target_seed=t,fit_status='PARENT_FALLBACK_NO_MATCHED_PAIRS',pairs=len(rr),distinct_rows=2*len(rr),recipient_rows=rr.tolist(),donor_rows=dd.tolist(),unmatched_positive_rows=missing,negative_candidate_rows=pool.tolist(),candidate_classes=[cls(int(d)) for d in pool],recipient_classes=[cls(int(r)) for r in positive[:cfg['maximum_pairs']]],fcc=None,raw=None))
                print(json.dumps(dict(query=[s,a],status='PARENT_FALLBACK_NO_MATCHED_PAIRS',pairs=len(rr),negative_pool=len(pool))),flush=True)
                continue
            wi={int(r):float(w) for r,w in zip(positive,family['discovery_weights'])};weights=np.array([wi[int(r)] for r in rr]);weights/=weights.sum();y=coordinate[rr]-coordinate[dd]
            xr=np.asarray(raw[rr],dtype=float)-np.asarray(raw[dd],dtype=float);wr,dr=fixed_support_ridge(xr,y,weights,cfg['ridge_fraction'],center=False)
            pair=dict(source_seed=s,source_atom=a,recipient_rows=rr.tolist(),donor_rows=dd.tolist(),source_weights=weights.tolist(),source_coordinate_difference=y.tolist(),unmatched_positive_rows=missing,negative_candidate_rows=pool.tolist(),distinct_rows=len(set(rr.tolist()+dd.tolist())),pairs=len(rr),classes=[cls(int(r)) for r in rr])
            for t in cfg['source_seeds']:
                if t==s:continue
                ix=index[s,a,t];assert np.array_equal(old['source_basis'][ix,:,:1],b)
                assert families[s,a,t]['discovery_rows']==family['discovery_rows'] and families[s,a,t]['discovery_weights']==family['discovery_weights']
                xt=(dense(t,rr)-dense(t,dd))@dec[t];wt,dt=fixed_support_ridge(xt,y,weights,cfg['ridge_fraction'],center=False)
                factors['query_target'][ix,:,0]=wt;factors['raw_target'][ix,:,0]=wr;changed.append(ix)
                records.append(dict(**pair,target_seed=t,fit_status='FITTED',fcc=dt,raw=dr,old_fcc_norm=float(np.linalg.norm(old['query_target'][ix,:,0])),old_raw_norm=float(np.linalg.norm(old['raw_target'][ix,:,0]))))
            print(json.dumps(dict(query=[s,a],pairs=len(rr),missing=len(missing),elapsed=time.perf_counter()-start)),flush=True)
            if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('CPU budget exceeded')
        checks['only_rank1_maps_changed']=True
        for key in old.files:
            if key not in ('query_target','raw_target'):np.testing.assert_array_equal(factors[key],old[key])
            else:
                expected=old[key].copy();expected[changed,:,0]=factors[key][changed,:,0];np.testing.assert_array_equal(expected,factors[key])
        checks['finite_maps']=all(np.isfinite(factors[k]).all() for k in ('query_target','raw_target'))
        checks['unique_paired_budget']=all(r['distinct_rows']<=cfg['maximum_distinct_endpoint_rows'] and r['distinct_rows']==2*r['pairs'] for r in records)
        checks['all_selected_queries_accounted']=len(records)==4*len(selected_queries)
        factorpath=run/'operation_matched_factors.npz';np.savez_compressed(factorpath,**factors)
        for name,ref in panelcfgs.items():
            ref=dict(ref);ref.pop('readout_ablation',None);ref['run_id']=f'F4_full_operation_{name}_dev_v1_20260905';ref['methods']=['target','raw'];ref['factors_path']=factorpath.relative_to(ROOT).as_posix();ref['factors_sha256']=sha256(factorpath)
            ref['evidence_level']='exposed_natural_text_operation_matched_development';ref['scope_limit']=cfg['scope'];ref['intervention']=cfg['objective'];ref['budget']='Original3cases33forwards plus expanded8cases88forwards=121; <=180s numerical plus90s expected loading/export, source/common scales unchanged.'
            ref['fit_provenance']=dict(path=str(run),config_sha256=sha256(run/'config.resolved.json'),data_split='original_discovery',parent_factors_sha256=cfg['parent_factors_sha256'],fallback_queries=[list(q) for q in sorted({(r['source_seed'],r['source_atom']) for r in records if r['fit_status']!='FITTED'})])
            dest=run/f'{name}_causal_config.json';write(dest,ref);outputs.append(str(dest))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    write(run/'inputs.json',dict(inputs=inputs));(run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    write(run/'metrics.summary.json',dict(status=status,error=error,checks=checks,fit_count=len(records),wall_seconds=time.perf_counter()-start,model_forwards=0,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/prepare_f4_full_operation_fit.py',generator_script_sha256=sha256(Path(__file__)),causal_configs=outputs))
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,platform=platform.platform(),device='cpu',cuda='not_used',torch='not_used',transformers=transformers_version,sae='saved codes only'))
    write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));write(run/'stdout.log',dict(status=status,error=error,outputs=outputs))
    if not (run/'stderr.log').exists():(run/'stderr.log').write_text('')
    contract=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=contract.ok,errors=list(contract.errors)))
    print(json.dumps(dict(status=status,error=error,checks=checks,contract_ok=contract.ok,outputs=outputs)),flush=True)
    return 0 if status=='PASS' and contract.ok else 1


if __name__=='__main__':raise SystemExit(main())
