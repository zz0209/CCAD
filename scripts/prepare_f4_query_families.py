"""Prepare new source-hash-query comparisons from original discovery assets only."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from run_f4_source_reference_causal import (
    ROOT, np, write, jsonl, sha256, validate_run_directory, source_hash_queries,
    fit_single_atoms, fit_ot_maps, prepare_readout_ablation,
)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();task=json.loads(args.config.read_text())
    ref=ROOT/task['reference_config_path']
    if sha256(ref)!=task['reference_config_sha256']: raise ValueError('Reference configuration changed')
    cfg=json.loads(ref.read_text());cfg.update(task)
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    started=datetime.now(timezone.utc).isoformat();begin=time.perf_counter()
    write(run/'config.resolved.json',cfg)
    sources=[Path(__file__),ROOT/'scripts/run_f4_source_reference_causal.py']+[
        ROOT/'src/ccad'/name for name in ('artifacts.py','activation_contract.py','ot_transport.py','nip_baselines.py','proposal.py')]
    code=[]
    for p in sorted(sources,key=lambda p:p.relative_to(ROOT).as_posix()):
        rel=p.relative_to(ROOT).as_posix();dest=run/'source_snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path=dest.relative_to(run).as_posix()))
    code_hash=hashlib.sha256(''.join(f"{r['path']}:{r['sha256']}\n" for r in code).encode()).hexdigest()
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=code_hash,snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.query.preparation.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose='New-query discovery readouts and fixed baseline fits, no model forwards',milestone='M4',
        evidence_level='discovery_only_parameter_preparation',started_utc=started,project_root=str(ROOT),
        config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=code_hash,source_snapshot_required=True,
        audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='mean',
        threshold_source_split='prior_development_frozen; no new thresholds',statistics_unit='source_query/seed',
        device='cpu',seeds=cfg['source_seeds'],resource_lease='cpu-heavy resource_manager.run',
        resource_lease_reason='Discovery scalar and OT fits; no GPU or evaluation data required'))
    write(run/'status.json',dict(status='RUNNING',updated_utc=started))
    paths={key:ROOT/cfg[key+'_path'] for key in ('surface','factors','source_census','query_panel')}
    paths['assets']=Path(cfg['bulk_asset_dir'])/'asset_manifest.json'
    paths['reference_config']=ref
    expected={key:cfg[key+'_sha256'] for key in ('surface','factors','source_census','query_panel')}
    expected.update(assets=cfg['asset_manifest_sha256'],reference_config=cfg['reference_config_sha256'])
    write(run/'inputs.json',dict(inputs=[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size,
        source='CCAD frozen discovery artifact',license_or_access_boundary='internal; no audit arrays',
        role=k) for k,p in paths.items()]))
    error=None;records=[];checks={}
    try:
        if any(sha256(p)!=expected[k] for k,p in paths.items()):raise ValueError('Input identity mismatch')
        surface={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in jsonl(paths['surface']) if r['query_role']=='anchor' and r['rank']==1}
        panel={(r['seed'],r['atom']):r for r in jsonl(paths['query_panel'])}
        available=sorted({(s,a) for s,a,t in surface})
        queries=source_hash_queries(available,panel,cfg['query_hash_offset'])
        old=source_hash_queries(available,panel,0)
        checks['new_queries_disjoint']=not(set(queries)&set(old))
        checks['expected_panel']=queries==[tuple(x) for x in cfg['expected_queries']]
        write(run/'query_panel_selection.json',dict(queries=queries,old_queries=old,
            hash_offset=cfg['query_hash_offset'],target_outcomes_read=False))
        factors=np.load(paths['factors'],allow_pickle=False)
        findex={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(factors['source_seed'],factors['source_atom'],factors['target_seed']))}
        means={s:np.zeros(cfg['num_latents']) for s in cfg['source_seeds']}
        for r in jsonl(paths['source_census']):means[r['seed']][r['atom']]=r['mean_code']
        asset=Path(cfg['bulk_asset_dir']);manifest=json.loads(paths['assets'].read_text())
        split_tokens={r['split']:r['tokens'] for r in manifest['splits']}
        dec={s:np.asarray(np.memmap(asset/'decoders'/f'seed_{s}.float32.bin',dtype='<f4',mode='r',
            shape=(cfg['num_latents'],cfg['hook_hidden_size'])),dtype=np.float64) for s in cfg['source_seeds']}
        for name,dynamic in [('single_atom_level',False),('single_atom_dynamic',True)]:
            fitcfg=dict(cfg,single_atom_fit=dict(cfg['single_atom_fit'],conditional_variation=dynamic))
            fits,payload=fit_single_atoms(fitcfg,queries,surface,factors,findex,means,dec,split_tokens)
            write(run/(name+'.json'),payload)
            checks[name+'_count']=len(fits)==32
            records.extend(dict(method=name,**r) for r in payload['fits'])
            print(json.dumps(dict(prepared=name,count=len(fits))),flush=True)
        reference_paths={'ot_reference_config':run/'config.resolved.json','ot_reference_fit':run/'single_atom_dynamic.json'}
        for name,spec in cfg['ot_families'].items():
            output=run/name;output.mkdir()
            fits=fit_ot_maps(dict(cfg,ot_fit=spec),queries,surface,factors,findex,dec,output,reference_paths)
            payload=json.loads((output/'ot_fits.json').read_text())
            checks[name+'_count']=len(fits)==32
            checks[name+'_finite']=all(np.isfinite(v).all() for v in fits.values())
            records.extend(dict(method=name,**r) for r in payload['fits'])
        readcfg=dict(cfg,saved_atom_families=[dict(method='single_atom_dynamic')])
        readpaths={'saved_atom_0_config':run/'config.resolved.json','saved_atom_0_fit':run/'single_atom_dynamic.json'}
        families=prepare_readout_ablation(readcfg,factors,findex,dec,run,readpaths)
        checks['readout_count']=len(families)==32
        checks['no_evaluation_arrays']=True
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    status='PASS' if checks and all(checks.values()) and error is None else 'FAIL'
    (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    write(run/'metrics.summary.json',dict(status=status,error=error,checks=checks,wall_seconds=time.perf_counter()-begin,
        model_forwards=0,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
        generator_script_path='scripts/prepare_f4_query_families.py',generator_script_sha256=sha256(Path(__file__)),
        scope_limit='Original discovery parameter preparation, not a causal result. New queries, fixed old hyperparameters.'))
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
        scipy='not_used; NumPy-only fitting kernels',platform=platform.platform(),device='cpu',cuda='not_used',
        torch='not_used',transformers='not_used',sae='frozen decoder/code assets'))
    write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    write(run/'stdout.log',dict(status=status,error=error))
    if not (run/'stderr.log').exists():(run/'stderr.log').write_text('')
    result=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=result.ok,errors=list(result.errors)))
    print(json.dumps(dict(run=str(run),status=status,error=error,contract_ok=result.ok,contract_errors=list(result.errors))))
    return 0 if status=='PASS' and result.ok else 1


if __name__=='__main__':raise SystemExit(main())
