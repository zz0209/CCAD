"""Development feedback for saved signed F4 maps against one source reference.

Optional single-atom fitting uses discovery only; no audit access. Wrong-query maps use their own source basis and are
also tested at the reference hook norm. Every effect is compared to the SAME
source-local projection, not to a method-specific self-reference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256, validate_run_directory


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compare(a, b):
    # float64 accumulation without changing model precision
    aa, bb, ab = float(np.sum(a*a, dtype=np.float64)), float(np.sum(b*b, dtype=np.float64)), float(np.sum(a*b, dtype=np.float64))
    return {"source_energy": aa, "candidate_energy": bb, "cross_energy": ab,
            "normalized_error": float(np.sum((a-b)**2, dtype=np.float64))/aa if aa > 0 else None,
            "bcc": 2*ab/(aa+bb) if aa+bb > 0 else None,
            "source_rms": float(np.sqrt(aa/a.size)), "candidate_rms": float(np.sqrt(bb/b.size))}


def content_mask(tokens, values, maximum_positions=4):
    """Input/source-only intervention positions, at least 16 after a boundary."""
    boundary=-1; eligible=[]
    for pos,token in enumerate(tokens):
        if int(token)==0: boundary=pos
        if pos-boundary>=16 and values[pos]>0: eligible.append(pos)
    chosen=sorted(eligible,key=lambda p:(-values[p],p))[:maximum_positions]
    mask=np.zeros(len(tokens));mask[chosen]=1
    return mask


def aligned_difference(recipient, donor, positions, donor_positions):
    """Swap contribution coordinates at source-selected aligned positions."""
    if len(positions)!=len(donor_positions):
        raise ValueError('Recipient and donor position counts must match')
    result=np.zeros_like(recipient)
    result[positions]=recipient[positions]-donor[donor_positions]
    return result


def source_dose_scale(source, masked_hook, maximum_fraction=None):
    """One source-defined scale, applied unchanged to every candidate."""
    if maximum_fraction is None:
        return 1.0
    if maximum_fraction<=0:
        raise ValueError('Source dose fraction must be positive')
    norm=float(np.linalg.norm(source))
    return min(1.0,maximum_fraction*float(np.linalg.norm(masked_hook))/norm) if norm else 1.0


def selection_document_ids(selection):
    return {d for q in selection['queries'] for e in q['sequences']
            for d in e['document_ids']+e.get('donor_document_ids',[])}


def best_single_atom(x, y, weights, ridge_fraction, conditional_variation=False):
    """All scalar predictors, same trace/min(shape) ridge rule as F4.

    Inputs are independently mean-centered codes; no fitted intercept. Each
    column is a separate one-dimensional model, NOT joint latent regression.
    Choose by discovery weighted prediction error, with atom-id tie breaking.
    Decoder norm cancels from this scalar trace-normalized ridge solution.
    """
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64);w=w/w.sum()
    if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or ridge_fraction<=0:
        raise ValueError('Invalid scalar regression inputs')
    source_conditional_mean=float(w@y);predictor_conditional_means=w@x
    if conditional_variation:
        # Weighted covariance = half the all-pairs weighted difference Gram.
        # Intercepts cancel in the actual donor difference intervention.
        x=x-predictor_conditional_means;y=y-source_conditional_mean
    second=np.sum(w[:,None]*x*x,axis=0);cross=x.T@(w*y)
    beta=np.divide(cross,second*(1+ridge_fraction),out=np.zeros_like(cross),where=second>0)
    error=np.sum(w[:,None]*(y[:,None]-x*beta)**2,axis=0)
    atom=int(np.argmin(error))
    return {'atom':atom,'coefficient':float(beta[atom]),'weighted_error':float(error[atom]),
            'source_second_moment':float(np.sum(w*y*y)),
            'predictor_second_moment':float(second[atom]),'ridge':float(ridge_fraction*second[atom]),
            'conditional_variation_objective':conditional_variation,
            'source_conditional_mean':source_conditional_mean,
            'predictor_conditional_mean':float(predictor_conditional_means[atom]),
            'candidate_count':x.shape[1]}


def fit_single_atoms(cfg, queries, surface, factors, findex, means, dec, split_tokens):
    """Materialize only <=256 discovery rows per query, never calibration."""
    spec=cfg['single_atom_fit'];asset=Path(cfg['bulk_asset_dir'])
    shape=(split_tokens['discovery'],cfg['k'])
    indices={s:np.memmap(asset/'discovery'/f'seed_{s}'/'top_indices.uint16.bin',dtype='<u2',mode='r',shape=shape) for s in cfg['source_seeds']}
    acts={s:np.memmap(asset/'discovery'/f'seed_{s}'/'top_acts.float32.bin',dtype='<f4',mode='r',shape=shape) for s in cfg['source_seeds']}
    def dense_rows(seed, rows):
        z=np.zeros((len(rows),cfg['num_latents']))
        np.add.at(z,(np.arange(len(rows))[:,None],indices[seed][rows]),acts[seed][rows])
        return z
    fits={};records=[]
    for s,a in queries:
        scores=np.zeros(shape[0])
        for begin in range(0,shape[0],4096):
            sl=slice(begin,begin+4096)
            values=np.sum(np.where(indices[s][sl]==a,acts[s][sl],0),axis=1,dtype=np.float64)
            scores[sl]=np.abs(values)**spec['condition_weight_power']
        positive=np.flatnonzero(scores>0)
        rows=positive[np.lexsort((positive,-scores[positive]))[:spec['max_condition_tokens']]]
        if not len(rows): raise ValueError('No source-positive discovery fit rows')
        weights=scores[rows]/scores[rows].sum()
        targets=[t for t in cfg['source_seeds'] if t!=s];t0=targets[0]
        ids=surface[s,a,t0]['source_candidate_ids']
        basis=factors['source_basis'][findex[s,a,t0],:,:1].astype(np.float64)
        y=((dense_rows(s,rows)[:,ids]-means[s][ids])@dec[s][ids]@basis)[:,0]
        for t in targets:
            if not np.array_equal(basis,factors['source_basis'][findex[s,a,t],:,:1]):
                raise ValueError('Single-atom comparison requires identical source basis')
            fit=best_single_atom(dense_rows(t,rows)-means[t],y,weights,spec['ridge_fraction'],spec.get('conditional_variation',False))
            fits[s,a,t]=fit
            records.append({'source_seed':s,'source_atom':a,'target_seed':t,**fit,
                            'discovery_rows':rows.tolist(),'discovery_weights':weights.tolist()})
    return fits,{'fit_split':'discovery','selection_endpoint':'weighted conditional variation error (equivalent to all-pair difference error)' if spec.get('conditional_variation') else 'weighted source-coordinate prediction error',
                 'calibration_used_for_fit':False,'rank':1,'intercept':'conditional fit intercept cancels in donor difference' if spec.get('conditional_variation') else 'none; independent mean split',
                 'ridge_rule':'fraction * predictor weighted second moment / min(n,1)',
                 'operation':'source-aligned single-atom map, not native deletion','fits':records}


def fit_ot_maps(cfg, queries, surface, factors, findex, dec, run, paths):
    from ccad.ot_transport import signed_ot_readout, discovery_document_partition, weighted_difference_error
    original=json.loads(paths['ot_reference_config'].read_text())
    identity_fields=('factors_sha256','source_census_sha256','query_panel_sha256','model_revision','hook_module_path','num_latents','hook_hidden_size')
    if any(original[k]!=cfg[k] for k in identity_fields):
        raise ValueError('OT reference model/basis/mean identity mismatch')
    asset=Path(original['bulk_asset_dir'])
    manifest_path=asset/'asset_manifest.json'
    if sha256(manifest_path)!=original['asset_manifest_sha256']:
        raise ValueError('OT discovery manifest changed')
    manifest=json.loads(manifest_path.read_text())
    for entry in manifest['decoders']:
        current=Path(cfg['bulk_asset_dir'])/'decoders'/f"seed_{entry['seed']}.float32.bin"
        if sha256(current)!=entry['sha256']:
            raise ValueError('OT discovery and evaluation decoder differ')
    tokens=next(r['tokens'] for r in manifest['splits'] if r['split']=='discovery')
    reference=json.loads(paths['ot_reference_fit'].read_text())
    if reference['fit_split']!='discovery' or reference['calibration_used_for_fit']:
        raise ValueError('OT fit-row provenance is not discovery-only')
    shape=(tokens,cfg['k'])
    indices={s:np.memmap(asset/'discovery'/f'seed_{s}'/'top_indices.uint16.bin',dtype='<u2',mode='r',shape=shape) for s in cfg['source_seeds']}
    acts={s:np.memmap(asset/'discovery'/f'seed_{s}'/'top_acts.float32.bin',dtype='<f4',mode='r',shape=shape) for s in cfg['source_seeds']}
    def dense(seed,rows):
        z=np.zeros((len(rows),cfg['num_latents']))
        np.add.at(z,(np.arange(len(rows))[:,None],indices[seed][rows]),acts[seed][rows])
        return z
    fit_spec=dict(cfg['ot_fit'])
    if cfg.get('ot_tuning'):
        spec=cfg['ot_tuning']
        sequences=json.loads(paths['ot_discovery_sequences'].read_text())['sequences']
        tuning_rows=[];partitions=[]
        for s,a in queries:
            prior=[r for r in reference['fits'] if (r['source_seed'],r['source_atom'])==(s,a)]
            row_ids=prior[0]['discovery_rows'];weights=np.asarray(prior[0]['discovery_weights'])
            if any(r['discovery_rows']!=row_ids or r['discovery_weights']!=prior[0]['discovery_weights'] for r in prior):
                raise ValueError('OT tuning rows are not target-independent')
            part=discovery_document_partition(row_ids,sequences,cfg['context_length'],salt=spec['document_hash_salt'])
            partitions.append(dict(source_seed=s,source_atom=a,discovery_rows=row_ids,**part))
            tr=part['train_indices'];va=part['validation_indices']
            if len(tr)<2 or len(va)<2:
                raise ValueError('Insufficient single-document discovery fit/validation rows')
            targets=[t for t in cfg['source_seeds'] if t!=s]
            ids=surface[s,a,targets[0]]['source_candidate_ids']
            basis=factors['source_basis'][findex[s,a,targets[0]],:,:1].astype(np.float64)
            coordinate=(dec[s][ids]@basis)[:,0];xs=dense(s,row_ids)[:,ids]
            for t in targets:
                xt=dense(t,row_ids)
                for epsilon in spec['regularization_grid']:
                    candidate=dict(fit_spec,regularization=epsilon)
                    beta,_,diagnostic=signed_ot_readout(xs[tr],xt[tr],coordinate,weights[tr],**candidate)
                    error=weighted_difference_error(xs[va]@coordinate,xt[va]@beta,weights[va])
                    tuning_rows.append(dict(source_seed=s,source_atom=a,target_seed=t,regularization=epsilon,
                                            train_rows=len(tr),validation_rows=len(va),fit_status=diagnostic['status'],**error))
                    with (run/'ot_tuning.raw.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(tuning_rows[-1],sort_keys=True)+'\n')
            write(run/'ot_tuning_partitions.json',partitions)
            print(json.dumps({'ot_tuning_query':[s,a],'completed_fits':len(tuning_rows)}),flush=True)
        scores=[]
        for epsilon in spec['regularization_grid']:
            per_query=[]
            for s,a in queries:
                values=[r['normalized_error'] for r in tuning_rows if (r['source_seed'],r['source_atom'],r['regularization'])==(s,a,epsilon) and r['normalized_error'] is not None]
                per_query.append(dict(source_seed=s,source_atom=a,valid_targets=len(values),error=float(np.median(values)) if values else None))
            valid=[r['error'] for r in per_query if r['error'] is not None]
            scores.append(dict(regularization=epsilon,mean_query_error=float(np.mean(valid)) if valid else None,valid_queries=len(valid),query_errors=per_query))
        if not all(r['valid_queries']==len(queries) for r in scores):
            raise ValueError('OT tuning has unsupported queries; do not silently select a subset')
        chosen=min(scores,key=lambda r:(r['mean_query_error'],r['regularization']))['regularization']
        fit_spec['regularization']=chosen
        write(run/'ot_tuning.json',{'partitions':partitions,'raw_validation':tuning_rows,'scores':scores,
              'chosen_regularization':chosen,'fit_spec':fit_spec,'selection':'lowest mean across query median across targets of held-out-discovery weighted all-pair-difference error',
              'scope':'Single-document contexts only during tuning; original fixed source basis is reused, not independently fitted in this inner split. No calibration endpoints. Refit chosen epsilon on original full256 discovery rows including packed contexts for matched final budget.'})
        print(json.dumps({'ot_selected_regularization':chosen,'scores':scores}),flush=True)
    fits={};records=[];arrays={}
    for s,a in queries:
        prior=[r for r in reference['fits'] if (r['source_seed'],r['source_atom'])==(s,a)]
        row_ids=prior[0]['discovery_rows'];weights=prior[0]['discovery_weights']
        if any(r['discovery_rows']!=row_ids or r['discovery_weights']!=weights for r in prior):
            raise ValueError('OT source-only discovery rows differ by target')
        if min(row_ids)<0 or max(row_ids)>=tokens:
            raise ValueError('OT fit rows outside discovery')
        targets=[t for t in cfg['source_seeds'] if t!=s];t0=targets[0]
        ids=surface[s,a,t0]['source_candidate_ids']
        basis=factors['source_basis'][findex[s,a,t0],:,:1].astype(np.float64)
        source_codes=dense(s,row_ids)[:,ids]
        coordinate=(dec[s][ids]@basis)[:,0]
        for t in targets:
            if not np.array_equal(basis,factors['source_basis'][findex[s,a,t],:,:1]):
                raise ValueError('OT requires target-independent source basis')
            coefficients,plan,diagnostics=signed_ot_readout(source_codes,dense(t,row_ids),coordinate,weights,**fit_spec)
            key=f's{s}_a{a}_t{t}';arrays[key]=coefficients;arrays[key+'_plan']=plan
            fits[s,a,t]=coefficients
            records.append(dict(source_seed=s,source_atom=a,target_seed=t,array_key=key,source_candidate_ids=ids,discovery_rows=row_ids,discovery_weights=weights,**diagnostics))
        print(json.dumps({'ot_fit_query':[s,a],'completed_maps':len(fits)}),flush=True)
    np.savez_compressed(run/'ot_fits.npz',**arrays)
    write(run/'ot_fits.json',{'fit_split':'discovery','calibration_used_for_fit':False,'rank':1,
          'reference_rows':'reused source-only rows/weights from saved atom fit; atom choices/coefficients unused',
          'method':'paired-correlation UOT signed readout; not SCOTM/Semantic OT/MAS/native deletion',
          'discovery_manifest_path':str(manifest_path),'discovery_manifest_sha256':sha256(manifest_path),
          'arrays_sha256':sha256(run/'ot_fits.npz'),'fits':records})
    return fits


def readout_atom_order(x, beta, weights):
    """Rank fixed readout terms by discovery conditional-variation energy."""
    weights=np.asarray(weights,dtype=np.float64);weights=weights/weights.sum()
    centered=x-weights@x
    energies=np.sum(weights[:,None]*centered**2,axis=0)*beta**2
    return np.lexsort((np.arange(len(beta)),-energies)),energies


def norm_match(delta, reference):
    norm=float(np.linalg.norm(delta))
    return delta*(float(np.linalg.norm(reference))/norm) if norm>0 else np.zeros_like(delta)


def prepare_readout_ablation(cfg, factors, findex, dec, run, paths):
    spec=cfg['readout_ablation']
    if cfg['ranks']!=[1] or not cfg.get('donor_difference') or spec['native_budget']!=16:
        raise ValueError('Readout structure consumer requires rank1 donor differences and native budget16')
    if any(not isinstance(m,int) or m<=0 or m>cfg['num_latents'] for m in spec['budgets']):
        raise ValueError('Invalid readout truncation budget')
    i=next(i for i,e in enumerate(cfg['saved_atom_families']) if e['method']==spec['discovery_reference_method'])
    oldcfg=json.loads(paths[f'saved_atom_{i}_config'].read_text())
    reference=json.loads(paths[f'saved_atom_{i}_fit'].read_text())
    asset=Path(oldcfg['bulk_asset_dir']);manifest_path=asset/'asset_manifest.json'
    if sha256(manifest_path)!=oldcfg['asset_manifest_sha256']:
        raise ValueError('Readout-ablation discovery manifest changed')
    manifest=json.loads(manifest_path.read_text())
    for entry in manifest['decoders']:
        if sha256(Path(cfg['bulk_asset_dir'])/'decoders'/f"seed_{entry['seed']}.float32.bin")!=entry['sha256']:
            raise ValueError('Readout-ablation discovery decoder mismatch')
    shape=(next(r['tokens'] for r in manifest['splits'] if r['split']=='discovery'),cfg['k'])
    families={};records=[]
    for record in reference['fits']:
        s,a,t=(record[k] for k in ('source_seed','source_atom','target_seed'))
        ids=np.asarray(record['discovery_rows']);weights=np.asarray(record['discovery_weights'])
        indices=np.memmap(asset/'discovery'/f'seed_{t}'/'top_indices.uint16.bin',dtype='<u2',mode='r',shape=shape)
        acts=np.memmap(asset/'discovery'/f'seed_{t}'/'top_acts.float32.bin',dtype='<f4',mode='r',shape=shape)
        x=np.zeros((len(ids),cfg['num_latents']))
        np.add.at(x,(np.arange(len(ids))[:,None],indices[ids]),acts[ids])
        beta=(dec[t]@factors['query_target'][findex[s,a,t],:,:1].astype(np.float64))[:,0]
        order,energies=readout_atom_order(x,beta,weights)
        seed=int.from_bytes(hashlib.sha256(f"{spec['sign_seed']}:{s}:{a}:{t}".encode()).digest()[:8],'little')
        signs=np.random.default_rng(seed).choice([-1.,1.],size=len(beta))
        families[s,a,t]={'beta':beta,'order':order,'signs':signs}
        top=order[:max(spec['budgets'])]
        records.append(dict(source_seed=s,source_atom=a,target_seed=t,discovery_rows=ids.tolist(),
                            discovery_weights=weights.tolist(),top_atoms=top.tolist(),top_atom_energies=energies[top].tolist(),
                            total_atom_energy=float(energies.sum()),nonzero_energy_atoms=int(np.sum(energies>0)),
                            sign_seed=seed,beta_sha256=hashlib.sha256(beta.tobytes()).hexdigest(),
                            sign_sha256=hashlib.sha256(signs.tobytes()).hexdigest()))
    write(run/'readout_ablation.json',{'fit_split':'discovery','calibration_used_for_ranking':False,
          'refitted':False,'ranking':'weighted conditional variance(z_j) * (decoder_j @ W)^2; atom ID breaks ties',
          'discovery_manifest_path':str(manifest_path),'discovery_manifest_sha256':sha256(manifest_path),
          'scope':'Truncation of the deployed signed readout, not optimal sparse refitting or optimal native support.',
          'spec':spec,'families':records})
    return families


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    task = json.loads(args.config.read_text(encoding="utf-8"))
    # Inherit asset identities, not historical execution gates.
    asset_cfg = json.loads((ROOT / task["asset_config"]).read_text(encoding="utf-8"))
    asset_keys = ('query_panel_path query_panel_sha256 source_census_path source_census_sha256 '
                  'bulk_asset_dir asset_manifest_sha256 raw_hook_asset_dir raw_hook_manifest_sha256 '
                  'paired_corpus_run token_manifest_path token_manifest_sha256 sequence_records_path '
                  'sequence_records_sha256 model_id model_revision model_local_dir model_license '
                  'hook_module_path next_module_path hook_hidden_size num_latents k context_length '
                  'source_seeds device cublas_workspace_config').split()
    cfg = {key: asset_cfg[key] for key in asset_keys}
    cfg.update(task)
    cfg['audit_opened']=False
    run = ROOT / "runs" / cfg["run_id"]
    run.mkdir(exist_ok=False)
    start = time.perf_counter(); now = datetime.now(timezone.utc).isoformat()
    write(run / "config.resolved.json", cfg)
    code_paths = [Path(__file__), ROOT / "src/ccad/activation_contract.py", ROOT / "src/ccad/artifacts.py"]
    if cfg.get('ot_fit'):
        code_paths.extend(ROOT/'src/ccad'/name for name in ('ot_transport.py','nip_baselines.py','proposal.py'))
    if cfg.get('source_scope'):
        code_paths.extend(ROOT/'scripts'/name for name in ('inspect_f4_atom_participation.py','summarize_f4_source_scope.py'))
    code = sorted([{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in code_paths], key=lambda x:x["path"])
    code_hash = hashlib.sha256("".join(f"{x['path']}:{x['sha256']}\n" for x in code).encode()).hexdigest()
    for entry,p in zip(code, sorted(code_paths,key=lambda p:p.relative_to(ROOT).as_posix())):
        dest=run/'source_snapshot'/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes());entry['snapshot_path']=dest.relative_to(run).as_posix()
    write(run / "code_hashes.json", {"files":code, "aggregate_sha256":code_hash, "snapshot_root":"source_snapshot"})
    write(run / "manifest.json", {"schema_version":"fcc.causal.development.v1", "run_id":cfg["run_id"], "run_parent":"R011-F4", "purpose":"Signed source-reference causal feedback", "milestone":"M4", "evidence_level":"real_sae_development", "started_utc":now, "project_root":str(ROOT), "config_hash":sha256(run / "config.resolved.json"), "code_snapshot_hash":code_hash, "audit_opened":False, "candidate_family_frozen":True, "mean_constants_source_split":"mean", "threshold_source_split":"development_no_selection_threshold", "statistics_unit":"query/seed/document; directions sharing seeds dependent", "device":cfg["device"], "seeds":cfg["source_seeds"], "resource_lease":"disk-d-io -> cpu-heavy -> gpu-0 resource_manager.run", "resource_lease_reason":"paired assets and bounded causal forward feedback", "git_head_at_run":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), "git_status_porcelain":subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).splitlines()})
    write(run / "status.json", {"status":"RUNNING", "updated_utc":now})
    manifest=json.loads((run/'manifest.json').read_text());manifest['source_snapshot_required']=True
    manifest['evidence_level']=cfg.get('evidence_level',manifest['evidence_level'])
    if cfg.get('resource_lease'):
        manifest['resource_lease']=cfg['resource_lease']
        manifest['resource_lease_reason']=cfg.get('resource_lease_reason','Compute leases matched to the actual workload')
    write(run/'manifest.json',manifest)
    (run / "stderr.log").write_text("",encoding="utf-8")
    rows=[]; error=None; status="FAIL"; summary={}
    try:
        paths={"surface":ROOT/cfg["surface_path"], "factors":ROOT/cfg["factors_path"], "census":ROOT/cfg["source_census_path"], "panel":ROOT/cfg["query_panel_path"], "sequences":ROOT/cfg["sequence_records_path"], "tokens":ROOT/cfg["token_manifest_path"], "assets":Path(cfg["bulk_asset_dir"])/"asset_manifest.json", "raw":Path(cfg["raw_hook_asset_dir"])/"raw_hook_manifest.json"}
        expected={"surface":cfg["surface_sha256"],"factors":cfg["factors_sha256"],"census":cfg["source_census_sha256"],"panel":cfg["query_panel_sha256"],"sequences":cfg["sequence_records_sha256"],"tokens":cfg["token_manifest_sha256"],"assets":cfg["asset_manifest_sha256"],"raw":cfg["raw_hook_manifest_sha256"]}
        for i,entry in enumerate(cfg.get('saved_atom_families',[])):
            for kind in ('fit','config'):
                key=f'saved_atom_{i}_{kind}'
                paths[key]=ROOT/entry[f'{kind}_path'];expected[key]=entry[f'{kind}_sha256']
        for i,entry in enumerate(cfg.get('saved_ot_families',[])):
            for kind in ('fit','config','arrays'):
                key=f'saved_ot_{i}_{kind}'
                paths[key]=ROOT/entry[f'{kind}_path'];expected[key]=entry[f'{kind}_sha256']
        if cfg.get('source_scope'):
            paths['source_scope']=ROOT/cfg['source_scope']['path'];expected['source_scope']=cfg['source_scope']['sha256']
        if cfg.get('ot_fit'):
            for kind in ('fit','config'):
                entry=cfg['ot_fit_reference'];key=f'ot_reference_{kind}'
                paths[key]=ROOT/entry[f'{kind}_path'];expected[key]=entry[f'{kind}_sha256']
            oldcfg=json.loads(paths['ot_reference_config'].read_text())
            paths['ot_discovery_manifest']=Path(oldcfg['bulk_asset_dir'])/'asset_manifest.json'
            expected['ot_discovery_manifest']=oldcfg['asset_manifest_sha256']
            if cfg.get('ot_tuning'):
                paths['ot_discovery_sequences']=ROOT/oldcfg['sequence_records_path']
                expected['ot_discovery_sequences']=oldcfg['sequence_records_sha256']
        if cfg.get('single_atom_fit'):
            paths['original_fit_config']=ROOT/cfg['original_fit_config_path']
            expected['original_fit_config']=cfg['original_fit_config_sha256']
        if cfg.get('exclude_selection_path'):
            paths['excluded_selection']=ROOT/cfg['exclude_selection_path']
            expected['excluded_selection']=cfg['exclude_selection_sha256']
        for i,entry in enumerate(cfg.get('additional_exclude_selections',[])):
            key=f'excluded_selection_extra_{i}'
            paths[key]=ROOT/entry['path'];expected[key]=entry['sha256']
        if any(sha256(p).lower()!=expected[k].lower() for k,p in paths.items()): raise ValueError("Input identity mismatch")
        if cfg.get('single_atom_fit'):
            original=json.loads(paths['original_fit_config'].read_text())
            spec=cfg['single_atom_fit']
            if any(spec[k]!=original[old] for k,old in [('ridge_fraction','ridge_fraction'),('condition_weight_power','condition_weight_power'),('max_condition_tokens','max_condition_tokens_per_split')]):
                raise ValueError('Single-atom fit must match original discovery weighting and ridge fraction')
        write(run/"inputs.json",{"inputs":[{"path":str(p.resolve()),"sha256":sha256(p),"bytes":p.stat().st_size,"source":"CCAD saved artifact","license_or_access_boundary":"internal","role":k} for k,p in paths.items()]})
        surface={(r["source_seed"],r["source_atom"],r["target_seed"]):r for r in jsonl(paths["surface"]) if r["query_role"]=="anchor" and r["rank"]==1}
        panel={(r["seed"],r["atom"]):r for r in jsonl(paths["panel"])}
        factors=np.load(paths["factors"],allow_pickle=False)
        findex={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(factors["source_seed"],factors["source_atom"],factors["target_seed"]))}
        available=sorted({(s,a) for s,a,t in surface})
        queries=[min([q for q in available if panel[q]["energy_stratum"]==st],key=lambda q:panel[q]["selection_hash"]) for st in range(8)]
        means={s:np.zeros(cfg["num_latents"]) for s in cfg["source_seeds"]}
        for r in jsonl(paths["census"]): means[r["seed"]][r["atom"]]=r["mean_code"]
        asset=Path(cfg["bulk_asset_dir"]); length=cfg["context_length"]; hidden=cfg["hook_hidden_size"]
        split_tokens={r['split']:int(r['tokens']) for r in json.loads(paths['assets'].read_text())['splits']}
        nt=split_tokens['calibration']
        dec={s:np.asarray(np.memmap(asset/"decoders"/f"seed_{s}.float32.bin",dtype="<f4",mode="r",shape=(cfg["num_latents"],hidden)),dtype=np.float64) for s in cfg["source_seeds"]}
        ot_fits={}
        if cfg.get('ot_fit'):
            if cfg['ranks']!=[1] or not cfg.get('donor_difference'):
                raise ValueError('OT readout requires rank1 donor differences')
            ot_fits=fit_ot_maps(cfg,queries,surface,factors,findex,dec,run,paths)
        single_fits={}
        atom_families={}
        if cfg.get('single_atom_fit'):
            if cfg['ranks']!=[1]: raise ValueError('Single-atom baseline currently supports rank1 only')
            if cfg['single_atom_fit'].get('conditional_variation') and not cfg.get('donor_difference'):
                raise ValueError('Conditional-variation atom coefficients require donor difference intervention')
            single_fits,fit_record=fit_single_atoms(cfg,queries,surface,factors,findex,means,dec,split_tokens)
            write(run/'single_atom_fits.json',fit_record)
            print(json.dumps({'single_atom_fits':len(single_fits),'fit_split':'discovery'}),flush=True)
            atom_families['single_atom']=single_fits
        for i,entry in enumerate(cfg.get('saved_atom_families',[])):
            oldcfg=json.loads(paths[f'saved_atom_{i}_config'].read_text())
            if cfg['ranks']!=[1] or not cfg.get('donor_difference'):
                raise ValueError('Saved atom replication requires rank1 donor differences')
            identity_fields=('factors_sha256','source_census_sha256','query_panel_sha256','model_revision','hook_module_path','num_latents','hook_hidden_size')
            if any(oldcfg[k]!=cfg[k] for k in identity_fields):
                raise ValueError('Saved atom model/basis/mean identity mismatch')
            payload=json.loads(paths[f'saved_atom_{i}_fit'].read_text())
            if payload['fit_split']!='discovery' or payload['calibration_used_for_fit'] or payload['rank']!=1:
                raise ValueError('Saved atom fitting boundary mismatch')
            atom_families[entry['method']]={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in payload['fits']}
        if atom_families:
            write(run/'atom_fit_reuse.json',{'refitted':False if not single_fits else True,'families':{name:len(fits) for name,fits in atom_families.items()},'saved_inputs':cfg.get('saved_atom_families',[])})
        readout_families=prepare_readout_ablation(cfg,factors,findex,dec,run,paths) if cfg.get('readout_ablation') else {}
        saved_ot_families={}
        for i,entry in enumerate(cfg.get('saved_ot_families',[])):
            oldcfg=json.loads(paths[f'saved_ot_{i}_config'].read_text())
            identity_fields=('factors_sha256','source_census_sha256','query_panel_sha256','model_revision','hook_module_path','num_latents','hook_hidden_size')
            if cfg['ranks']!=[1] or not cfg.get('donor_difference') or any(oldcfg[k]!=cfg[k] for k in identity_fields):
                raise ValueError('Saved OT source/model identity mismatch')
            payload=json.loads(paths[f'saved_ot_{i}_fit'].read_text())
            if payload['fit_split']!='discovery' or payload['calibration_used_for_fit'] or payload['arrays_sha256']!=expected[f'saved_ot_{i}_arrays']:
                raise ValueError('Saved OT fitting boundary/array identity mismatch')
            oldmanifest=Path(oldcfg['bulk_asset_dir'])/'asset_manifest.json'
            if sha256(oldmanifest)!=oldcfg['asset_manifest_sha256']:
                raise ValueError('Saved OT decoder manifest changed')
            for record in json.loads(oldmanifest.read_text())['decoders']:
                if sha256(asset/'decoders'/f"seed_{record['seed']}.float32.bin")!=record['sha256']:
                    raise ValueError('Saved OT and current decoder differ')
            with np.load(paths[f'saved_ot_{i}_arrays'],allow_pickle=False) as arrays:
                family={(r['source_seed'],r['source_atom'],r['target_seed']):np.array(arrays[r['array_key']],copy=True) for r in payload['fits']}
            if any(v.shape!=(cfg['num_latents'],) or not np.isfinite(v).all() for v in family.values()):
                raise ValueError('Invalid saved OT coefficient shape or value')
            if entry['method'] in saved_ot_families:
                raise ValueError('Duplicate saved OT method')
            saved_ot_families[entry['method']]=family
        if saved_ot_families:
            write(run/'ot_fit_reuse.json',{'refitted':False,'families':{k:len(v) for k,v in saved_ot_families.items()},'saved_inputs':cfg['saved_ot_families']})
        indices={s:np.memmap(asset/"calibration"/f"seed_{s}"/"top_indices.uint16.bin",dtype="<u2",mode="r",shape=(nt,cfg["k"])) for s in cfg["source_seeds"]}
        acts={s:np.memmap(asset/"calibration"/f"seed_{s}"/"top_acts.float32.bin",dtype="<f4",mode="r",shape=(nt,cfg["k"])) for s in cfg["source_seeds"]}
        def atom_values(seed, atoms):
            return np.sum(np.where(np.isin(indices[seed],atoms),acts[seed],0),axis=1,dtype=np.float64)
        def dense_seq(seed, seq):
            sl=slice(seq*length,(seq+1)*length); z=np.zeros((length,cfg["num_latents"]))
            np.add.at(z,(np.arange(length)[:,None],indices[seed][sl]),acts[seed][sl]); return z
        rm={r["split"]:r for r in json.loads(paths["raw"].read_text())["splits"]}
        raw={sp:np.memmap(rm[sp]["path"],dtype="<f4",mode="r").reshape(rm[sp]["shape"]) for sp in ("mean","calibration")}
        rawmean=np.mean(raw["mean"],axis=0,dtype=np.float64)
        seqs={r["sequence_index"]:r for r in json.loads(paths["sequences"].read_text())["sequences"] if r["split"]=="calibration"}
        tm=json.loads(paths["tokens"].read_text())["outputs"]["calibration"]
        token_path=ROOT/"runs"/cfg["paired_corpus_run"]/tm["path"]
        if sha256(token_path).lower()!=tm["sha256"]: raise ValueError("Token identity mismatch")
        tokens=np.memmap(token_path,dtype="<u2",mode="r").reshape(-1,length)
        excluded=set()
        for key,path in paths.items():
            if key.startswith('excluded_selection'):
                excluded.update(selection_document_ids(json.loads(path.read_text())))
        selections=[]
        for s,a in queries:
            targets=[(s-1+j)%5+1 for j in range(1,cfg.get('targets_per_query',2)+1)]
            base=surface[s,a,targets[0]]; positive=atom_values(s,[a]); negative=atom_values(s,base["negative_source_atoms"])
            pe=np.sum(positive.reshape(-1,length)**2,axis=1); ne=np.sum(negative.reshape(-1,length)**2,axis=1)
            used=set(excluded); chosen=[]
            # source-only selection, disjoint listed documents within each query
            for label,score in (("positive",pe),("negative",ne/(1+pe))):
                for idx in np.argsort(-score,kind="stable"):
                    docs=set(seqs[int(idx)]["document_ids"])
                    if not docs.intersection(used) and score[idx]>0:
                        chosen.append({"condition":label,"sequence":int(idx),"document_ids":sorted(docs),"positive_energy":float(pe[idx]),"negative_energy":float(ne[idx])}); used.update(docs)
                        if sum(x["condition"]==label for x in chosen)==cfg["documents_per_condition"]: break
            wrong=min([q for q in available if q[0]==s and q[1]!=a],key=lambda q:panel[q]["selection_hash"])[1]
            selections.append({"source_seed":s,"source_atom":a,"stratum":panel[s,a]["energy_stratum"],"targets":targets,"wrong_atom":wrong,"sequences":chosen})
        if cfg.get("local_content_positions"):
            for unit in selections:
                s,a=unit["source_seed"],unit["source_atom"]
                positive=atom_values(s,[a]); negatives=surface[s,a,unit["targets"][0]]["negative_source_atoms"]
                negative=atom_values(s,negatives)
                for entry in unit["sequences"]:
                    seq=entry["sequence"];sl=slice(seq*length,(seq+1)*length)
                    values=positive[sl] if entry["condition"]=="positive" else negative[sl]
                    entry["intervention_positions"]=np.flatnonzero(content_mask(tokens[seq],values,cfg["local_content_positions"])).tolist()
        if cfg.get('donor_difference'):
            for unit in selections:
                s,a=unit['source_seed'],unit['source_atom'];t0=unit['targets'][0]
                ids=surface[s,a,t0]['source_candidate_ids']
                b=factors['source_basis'][findex[s,a,t0],:,:min(cfg['ranks'])].astype(np.float64)
                coordinates={e['sequence']:dense_seq(s,e['sequence'])[:,ids]@dec[s][ids]@b for e in unit['sequences']}
                for entry in unit['sequences']:
                    positions=entry['intervention_positions'];n=len(positions)
                    candidates=[]
                    for donor in unit['sequences']:
                        if donor['condition']==entry['condition'] or len(donor['intervention_positions'])<n or not n:
                            continue
                        dp=donor['intervention_positions'][:n]
                        energy=float(np.sum((coordinates[entry['sequence']][positions]-coordinates[donor['sequence']][dp])**2))
                        candidates.append((energy,-donor['sequence'],dp,donor))
                    if candidates:
                        energy,_,dp,donor=max(candidates,key=lambda x:x[:2])
                        if set(entry['document_ids']) & set(donor['document_ids']):
                            raise ValueError('Donor and recipient documents must be disjoint')
                        entry.update(donor_sequence=donor['sequence'],donor_positions=dp,donor_document_ids=donor['document_ids'],donor_source_difference_energy=energy,donor_status='SELECTED_SOURCE_ONLY')
                    else:
                        entry.update(donor_sequence=entry['sequence'],donor_positions=positions,donor_document_ids=entry['document_ids'],donor_source_difference_energy=0.0,donor_status='NO_ELIGIBLE_PAIR')
        if selection_document_ids({'queries':selections}) & excluded:
            raise ValueError('Selected recipient/donor document intersects prior exclusion union')
        write(run/"selection.json",{"rule":"source-only hashes and activation energies; no F4 FOUND filtering","queries":selections,'excluded_document_count':len(excluded),'requested_sequences_per_condition':cfg['documents_per_condition'],'actual_sequences_per_query':[{"query":[x['source_seed'],x['source_atom']],"positive":sum(e['condition']=='positive' for e in x['sequences']),"negative":sum(e['condition']=='negative' for e in x['sequences'])} for x in selections]})
        if cfg.get('source_scope'):
            from inspect_f4_atom_participation import participation
            from summarize_f4_source_scope import selected
            rule=json.loads(paths['source_scope'].read_text())['rule'];scope_rows=[]
            for unit in selections:
                s,a=unit['source_seed'],unit['source_atom'];t0=unit['targets'][0]
                ids=surface[s,a,t0]['source_candidate_ids'];b=factors['source_basis'][findex[s,a,t0],:,:1].astype(np.float64)
                weights=dec[s][ids]@b
                for entry in unit['sequences']:
                    pos=entry['intervention_positions'];dp=entry['donor_positions'];seq=entry['sequence']
                    zd=dense_seq(s,seq)[pos][:,ids]-dense_seq(s,entry['donor_sequence'])[dp][:,ids]
                    stats=participation(zd,np.zeros(len(ids)),weights)
                    hooknorm=float(np.linalg.norm(raw['calibration'][np.asarray(pos,dtype=int)+seq*length].astype(np.float64)))
                    fraction=np.sqrt(stats['aggregate_energy'])/hooknorm if hooknorm else None
                    row=dict(source_seed=s,source_atom=a,rank=1,condition=entry['condition'],sequence=seq,**stats,
                             natural_source_hook_fraction=fraction,supported=bool(stats['aggregate_energy']>0 and entry['donor_status']=='SELECTED_SOURCE_ONLY'))
                    row['selected']=selected(row,rule);scope_rows.append(row)
            write(run/'source_scope_selection.json',{'rule':rule,'rule_sha256':expected['source_scope'],'endpoint_results_read':False,'rows':scope_rows,
                  'scope':'Frozen rule applied before loading model or materializing target-code rows for endpoint evaluation; all candidate pairs still evaluated.'})
            print(json.dumps({'source_scope_selected':sum(r['selected'] for r in scope_rows),'candidate_pairs':len(scope_rows)}),flush=True)
        print(json.dumps({'source_preflight':[{'query':[u['source_seed'],u['source_atom']],
              'conditions':{condition:{'pairs':len(group),'supported_pairs':sum(e.get('donor_status')=='SELECTED_SOURCE_ONLY' for e in group),
                    'median_natural_difference_energy':float(np.median([e.get('donor_source_difference_energy',0) for e in group])) if group else None}
                    for condition in ('positive','negative') for group in [[e for e in u['sequences'] if e['condition']==condition]]}}
              for u in selections]}),flush=True)
        os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",CUBLAS_WORKSPACE_CONFIG=cfg["cublas_workspace_config"])
        import torch
        import transformers
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        model=transformers.AutoModelForCausalLM.from_pretrained(cfg["model_local_dir"],local_files_only=True,dtype=torch.float32,attn_implementation="eager").eval().to(cfg["device"])
        contract=HookPointContract(cfg["hook_module_path"],5,"resid_post",hidden)
        hook=model.get_submodule(cfg["hook_module_path"]); nxt=model.get_submodule(cfg["next_module_path"])
        forwards=0; noop=[]; replay=[]
        torch.cuda.reset_peak_memory_stats()
        def forward(batch, delta=None):
            nonlocal forwards
            cap={}
            def intervene(m,i,out):
                h=extract_primary_hook_tensor(out,contract); cap["hook"]=h.detach().clone()
                return replace_primary_hook_tensor(out,h-delta,contract) if delta is not None else None
            def capture(m,i,out): cap["next_state"]=(out[0] if isinstance(out,tuple) else out).detach().clone()
            h1=hook.register_forward_hook(intervene); h2=nxt.register_forward_hook(capture)
            try:
                with torch.no_grad(): cap["next_logits"]=model(batch,use_cache=False).logits.detach()
            finally: h2.remove();h1.remove()
            forwards+=1;return cap
        rawfile=run/"metrics.raw.jsonl"
        with rawfile.open("w",encoding="utf-8") as sink:
            for unit in selections:
                s,a=unit["source_seed"],unit["source_atom"]; t0=unit["targets"][0]
                ids=surface[s,a,t0]["source_candidate_ids"]
                for entry in unit["sequences"]:
                    seq=entry["sequence"]; z={v:dense_seq(v,seq) for v in [s]+unit["targets"]}
                    if cfg.get('donor_difference'):
                        z={v:aligned_difference(z[v],dense_seq(v,entry['donor_sequence']),entry['intervention_positions'],entry['donor_positions']) for v in z}
                    local=(z[s][:,ids] if cfg.get('donor_difference') else z[s][:,ids]-means[s][ids])@dec[s][ids]
                    batch=torch.from_numpy(np.asarray(tokens[seq:seq+1],dtype=np.int64)).to(cfg["device"])
                    baseline=forward(batch); zero=forward(batch,torch.zeros_like(baseline["hook"]))
                    noop.append(max(float((baseline[e]-zero[e]).abs().max()) for e in ("next_state","next_logits")))
                    rawseq=np.asarray(raw["calibration"][seq*length:(seq+1)*length],dtype=np.float64)
                    live=baseline["hook"][0].cpu().numpy(); replay.append(float(np.linalg.norm(rawseq-live)/max(np.linalg.norm(live),1e-12)))
                    rawinput=rawseq-rawmean
                    if cfg.get('donor_difference'):
                        donor_start=entry['donor_sequence']*length
                        rawinput=aligned_difference(rawseq,np.asarray(raw['calibration'][donor_start:donor_start+length],dtype=np.float64),entry['intervention_positions'],entry['donor_positions'])
                    for rank in cfg["ranks"]:
                        b=np.asarray(factors["source_basis"][findex[s,a,t0],:,:rank],dtype=np.float64)
                        mask=np.zeros(length) if cfg.get("local_content_positions") else np.ones(length)
                        if cfg.get("local_content_positions"): mask[entry["intervention_positions"]]=1
                        source_natural=(local@b@b.T)*mask[:,None]
                        masked_hook_norm=float(np.linalg.norm(rawseq*mask[:,None]))
                        dose_scale=source_dose_scale(source_natural,rawseq*mask[:,None],cfg.get('maximum_source_hook_fraction'))
                        source=source_natural*dose_scale
                        ref=forward(batch,torch.tensor(source[None],dtype=torch.float32,device=cfg["device"]))
                        effects={e:(baseline[e]-ref[e]).cpu().numpy() for e in ("next_state","next_logits")}
                        if cfg.get("centered_logit_endpoint"):
                            effects["centered_logits"]=effects["next_logits"]-effects["next_logits"].mean(axis=-1,keepdims=True)
                        for t in unit["targets"]:
                            ix=findex[s,a,t]; bt=np.asarray(factors["source_basis"][ix,:,:rank],dtype=np.float64)
                            target=(z[t] if cfg.get('donor_difference') else z[t]-means[t])@dec[t]
                            widx=findex[s,unit["wrong_atom"],t]; wb=np.asarray(factors["source_basis"][widx,:,:rank],dtype=np.float64)
                            wrong=(target@np.asarray(factors["query_target"][widx,:,:rank],dtype=np.float64)@wb.T)*mask[:,None]
                            wrongscale=float(np.linalg.norm(source_natural)/max(np.linalg.norm(wrong),1e-12))
                            variants={"target":(target@np.asarray(factors["query_target"][ix,:,:rank],dtype=np.float64)@bt.T)*mask[:,None],
                                      "raw":(rawinput@np.asarray(factors["raw_target"][ix,:,:rank],dtype=np.float64)@bt.T)*mask[:,None],
                                      "wrong_query":wrong,"wrong_query_matched_energy":wrong*wrongscale}
                            if cfg.get('include_source_mean_only'):
                                variants['source_mean_only']=np.zeros_like(source) if cfg.get('donor_difference') else np.broadcast_to(-means[s][ids]@dec[s][ids]@b@b.T,source.shape)*mask[:,None]
                            for atom_method,fit_family in atom_families.items():
                                fit=fit_family[s,a,t];atom=fit['atom']
                                code=z[t][:,atom] if cfg.get('donor_difference') else z[t][:,atom]-means[t][atom]
                                variants[atom_method]=(code*fit['coefficient'])[:,None]@bt.T*mask[:,None]
                            if ot_fits:
                                variants['paired_correlation_uot']=(z[t]@ot_fits[s,a,t])[:,None]@bt.T*mask[:,None]
                            for ot_method,family in saved_ot_families.items():
                                variants[ot_method]=(z[t]@family[s,a,t])[:,None]@bt.T*mask[:,None]
                            if readout_families:
                                family=readout_families[s,a,t];beta=family['beta'];order=family['order']
                                for budget in cfg['readout_ablation']['budgets']:
                                    keep=order[:budget]
                                    variants[f'readout_top{budget}']=(z[t][:,keep]@beta[keep])[:,None]@bt.T*mask[:,None]
                                scrambled=(z[t]@(beta*family['signs']))[:,None]@bt.T*mask[:,None]
                                variants['readout_sign_scrambled_matched']=norm_match(scrambled,variants['target'])
                                keep=order[:cfg['readout_ablation']['native_budget']]
                                native=z[t][:,keep]@dec[t][keep]*mask[:,None]
                                variants['native_top16_difference']=native
                                variants['native_top16_difference_matched']=norm_match(native,source_natural)
                            if cfg.get('methods'):
                                variants={name:variants[name] for name in cfg['methods']}
                            for method,delta in variants.items():
                                delta=delta*dose_scale
                                out=forward(batch,torch.tensor(delta[None],dtype=torch.float32,device=cfg["device"]))
                                candidate_effects={e:(baseline[e]-out[e]).cpu().numpy() for e in ("next_state","next_logits")}
                                if cfg.get("centered_logit_endpoint"):
                                    candidate_effects["centered_logits"]=candidate_effects["next_logits"]-candidate_effects["next_logits"].mean(axis=-1,keepdims=True)
                                endpoints={e:compare(effects[e],candidate_effects[e]) for e in effects}
                                row={"source_seed":s,"source_atom":a,"target_seed":t,"stratum":unit["stratum"],"rank":rank,"method":method,**entry,"wrong_atom":unit["wrong_atom"],"wrong_norm_scale":wrongscale,"hook":compare(source,delta),"source_mean_projected_norm":float(np.linalg.norm(means[s][ids]@dec[s][ids]@b@b.T)),"target_mean_mapped_norm":float(np.linalg.norm(means[t]@dec[t]@np.asarray(factors["query_target"][ix,:,:rank],dtype=np.float64)@bt.T)),"endpoints":endpoints}
                                row['mean_terms_cancelled_in_intervention']=bool(cfg.get('donor_difference'))
                                row.update(common_source_dose_scale=dose_scale,source_natural_hook_energy=float(np.sum(source_natural**2)),recipient_masked_hook_norm=masked_hook_norm,source_hook_fraction=float(np.linalg.norm(source)/masked_hook_norm) if masked_hook_norm else None,candidate_hook_fraction=float(np.linalg.norm(delta)/masked_hook_norm) if masked_hook_norm else None)
                                rows.append(row);sink.write(json.dumps(row,sort_keys=True)+"\n");sink.flush()
                    print(json.dumps({"query":[s,a],"sequence":seq,"rows":len(rows),"forwards":forwards}),flush=True)
        method_names=['target','raw','wrong_query','wrong_query_matched_energy']+(['source_mean_only'] if cfg.get('include_source_mean_only') else [])
        method_names.extend(atom_families)
        if ot_fits: method_names.append('paired_correlation_uot')
        method_names.extend(saved_ot_families)
        if cfg.get('methods'): method_names=cfg['methods']
        checks={"noop":max(noop)<=1e-6,"raw_replay_relative":max(replay)<=1e-4,"eight_source_queries":len(selections)==8,"rows":len(rows)==sum(len(x["sequences"])*len(x["targets"])*len(cfg["ranks"])*len(method_names) for x in selections),"audit_closed":True}
        if cfg.get('maximum_source_hook_fraction'):
            checks['source_dose_bound']=all(r['source_hook_fraction'] is None or r['source_hook_fraction']<=cfg['maximum_source_hook_fraction']+1e-12 for r in rows)
        summary={"checks":checks,"model_forwards":forwards,"rows":len(rows),"wall_seconds":time.perf_counter()-start,"peak_allocated_vram_bytes":torch.cuda.max_memory_allocated(),"max_noop":max(noop),"max_replay_relative":max(replay),"by_method":{}}
        for condition in ("positive","negative"):
            for rank in cfg["ranks"]:
                for method in method_names:
                    group=[r for r in rows if r["condition"]==condition and r["rank"]==rank and r["method"]==method]
                    summary["by_method"][f"{condition}/r{rank}/{method}"]={e:{k:float(np.median([r["endpoints"][e][k] for r in group if r["endpoints"][e][k] is not None])) for k in ("normalized_error","bcc","source_rms")} for e in group[0]["endpoints"]}
        status="PASS" if all(checks.values()) else "FAIL"
        write(run/"environment.json",{"python":sys.executable,"python_version":platform.python_version(),"numpy":np.__version__,"scipy":"not_used","torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(),"platform":platform.platform()})
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}";(run/"stderr.log").write_text(traceback.format_exc(),encoding="utf-8")
        write(run/"environment.json",{"python":sys.executable,"error":error})
    if not (run/"metrics.raw.jsonl").exists(): (run/"metrics.raw.jsonl").write_text("",encoding="utf-8")
    summary.update(status=status,error=error,metrics_raw_sha256=sha256(run/"metrics.raw.jsonl"),generator_script_path="scripts/run_f4_source_reference_causal.py",generator_script_sha256=sha256(Path(__file__)),scope_limit="Calibration development, source-aligned transport; not native deletion or held-out confirmation")
    summary['scope_limit']=cfg.get('scope_limit',summary['scope_limit'])
    write(run/"metrics.summary.json",summary);write(run/"status.json",{"status":status,"error":error,"updated_utc":datetime.now(timezone.utc).isoformat()})
    write(run/"stdout.log",{"status":status,"rows":len(rows),"error":error})
    result=validate_run_directory(run);write(run/"contract_validation.json",{"ok":result.ok,"errors":list(result.errors)})
    print(json.dumps({"run":str(run),"status":status,"error":error,"contract_ok":result.ok,"contract_errors":list(result.errors)}))
    return 0 if status=="PASS" and result.ok else 1


if __name__=="__main__":
    raise SystemExit(main())
