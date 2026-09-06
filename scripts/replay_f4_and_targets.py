"""Apply frozen FCC/atom/raw maps to all revised-source lexical probe cases."""
import argparse,json,os,platform,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text())
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();write(run/'config.resolved.json',cfg)
    for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
    code=[]
    for rel in ['scripts/replay_f4_and_targets.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.and.targets.v1',run_id=cfg['run_id'],run_parent='F4',purpose='Frozen cross-seed prediction of a source-verified output contrast, with atom/raw controls',milestone='C2-C3-interpretability',evidence_level='source_selected_authored_cross_seed_feasibility',
        started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='independent mean; exact cancellation in paired difference',threshold_source_split='frozen source config; target behavior not used for selection',statistics_unit='one source query, four target seeds, four topics across two families; dependent',device=cfg['device'],seeds=[1,2,3,4,5],resource_lease='gpu-0 resource_manager.run',resource_lease_reason='224 bounded forwards and small cached-hook SAE encoding; no fitting or whole-disk work'))
    write(run/'status.json',dict(status='RUNNING'));inputs=[];rows=[];forwards=0
    def checked(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;e=entry(p,'frozen source run or historical map/weights','input','internal/Apache-2.0 model; no audit')
        if expected and e['sha256']!=expected:raise ValueError(f'Hash mismatch {p}')
        inputs.append(e);return p
    try:
        write(run/'resource_status_at_start.json',json.loads(subprocess.check_output([sys.executable,str(ROOT.parent/'.resource_manager/resource_manager.py'),'status','--resource','gpu-0'],text=True)))
        parent=ROOT/cfg['parent_run'];old=json.loads(checked(parent/'config.resolved.json',cfg['parent_config_sha256']).read_text());assets=json.loads(checked(old['asset_config']).read_text())
        saved=[json.loads(x) for x in checked(parent/'metrics.raw.jsonl',cfg['parent_raw_sha256']).read_text().splitlines()]
        arr=np.load(checked(parent/'source_arrays.npz',cfg['parent_arrays_sha256']),allow_pickle=False);auth=json.loads(checked(parent/'authored_inputs.json',cfg['parent_authored_sha256']).read_text())
        if len(saved)!=16 or len(auth['cases'])!=16:raise ValueError('Expected full 16-case source scope')
        factors=np.load(checked(old['factors_path'],old['factors_sha256']),allow_pickle=False);lookup={int(t):i for i,(s,a,t) in enumerate(zip(factors['source_seed'],factors['source_atom'],factors['target_seed'])) if s==old['source_seed'] and a==old['source_atom']}
        atom_payload=json.loads(checked(cfg['atom_fits_path'],cfg['atom_fits_sha256']).read_text())
        if atom_payload['fit_split']!='discovery' or atom_payload['calibration_used_for_fit']:raise ValueError('Atom fit boundary mismatch')
        atom={r['target_seed']:r for r in atom_payload['fits'] if r['source_seed']==old['source_seed'] and r['source_atom']==old['source_atom']}
        b=factors['source_basis'][lookup[cfg['target_seeds'][0]],:,0].astype(np.float64)
        if not all(np.array_equal(b,factors['source_basis'][lookup[t],:,0]) for t in cfg['target_seeds']):raise ValueError('Source basis changed by target')
        for t in cfg['target_seeds']:
            sp=next(p for p in assets['saes'] if p['seed']==t);checked(ROOT/sp['path']/'sae.safetensors',sp['sha256']);checked(ROOT/sp['path']/'cfg.json')
        checked(Path(assets['model_local_dir'])/'config.json');checked(Path(assets['model_local_dir'])/'tokenizer.json');checked('.aris/compute/local-r006b1-env-spec.json')
        import torch,transformers
        from sparsify import SparseCoder
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        model=transformers.AutoModelForCausalLM.from_pretrained(assets['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(cfg['device']);model.config.use_cache=False
        hooks=np.stack([arr[f'hook_{i}'] for i in range(16)]);coefficients={};target_codes={};raw_coefs={}
        for t in cfg['target_seeds']:
            sp=next(p for p in assets['saes'] if p['seed']==t);sae=SparseCoder.load_from_disk(ROOT/sp['path'],device=cfg['device']).eval()
            with torch.no_grad():enc=sae.encode(torch.tensor(hooks,device=cfg['device']))
            z=np.zeros((16,3072));np.add.at(z,(np.arange(16)[:,None],enc.top_indices.cpu().numpy()),enc.top_acts.cpu().numpy());target_codes[t]=z
            dec=sae.W_dec.detach().cpu().numpy().astype(np.float64);coefficients[t]=dec@factors['query_target'][lookup[t],:,0].astype(np.float64);raw_coefs[t]=factors['raw_target'][lookup[t],:,0].astype(np.float64)
            del sae
        write(run/'environment.json',dict(python=platform.python_version(),os=platform.platform(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),sae_framework='sparsify '+assets['sparsify_commit']))
        write(run/'inputs.json',dict(inputs=inputs));contract=HookPointContract(assets['hook_module_path'],5,'resid_post',768);module=model.get_submodule(assets['hook_module_path']);numeric=time.perf_counter();checks={};replay_arrays={};max_base=0.0;max_source=0.0;max_hook=0.0
        def forward(tokens,delta=None):
            nonlocal forwards
            if forwards>=cfg['maximum_forwards']:raise ValueError('Forward budget exceeded')
            captured=[]
            def hook(m,i,out):
                h=extract_primary_hook_tensor(out,contract);captured.append(h[0,-1].detach().cpu().numpy().copy())
                if delta is None:return out
                x=h.clone();x[0,-1]+=torch.tensor(delta,device=x.device,dtype=x.dtype);return replace_primary_hook_tensor(out,x,contract)
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad():lg=model(torch.tensor([tokens],device=cfg['device'])).logits[0,-1].cpu().numpy().astype(np.float64)
                forwards+=1;p=np.exp(lg-lg.max());return p/p.sum(),captured[0]
            finally:handle.remove()
        clause=auth['clause_token_ids']
        for i,c in enumerate(auth['cases']):
            original=saved[i]
            if any(original[k]!=c[k] for k in c):raise ValueError('Saved case changed')
            donor=original['donor_case_index'];scale=original['dose_scale'];scalar=original['source_difference_coordinate'];items=auth['item_token_ids'][c['pair_index']]
            base,h=forward(c['token_ids']);source,_=forward(c['token_ids'],-scalar*b*scale)
            max_hook=max(max_hook,float(np.max(abs(h-hooks[i]))));max_base=max(max_base,float(np.max(abs(base-arr[f'prob_{i}_baseline']))));max_source=max(max_source,float(np.max(abs(source-arr[f'prob_{i}_source_swap']))))
            basecontrast=float(np.log(base[clause].sum()/base[items].sum()));sourcecontrast=float(np.log(source[clause].sum()/source[items].sum())-basecontrast)
            if abs(sourcecontrast-original['effects']['source_swap']['contrast_delta'])>2e-6:raise ValueError('Source contrast does not replay')
            den=float(np.sum(source*(np.log(np.maximum(source,1e-300))-np.log(np.maximum(base,1e-300)))))
            replay_arrays[f'base_{i}']=base.astype(np.float32);replay_arrays[f'source_{i}']=source.astype(np.float32)
            for t in cfg['target_seeds']:
                dz=target_codes[t][i]-target_codes[t][donor];fit=atom[t]
                predicted={'fcc':float(dz@coefficients[t]),'atom':float(dz[fit['atom']]*fit['coefficient']),'raw':float((hooks[i].astype(np.float64)-hooks[donor].astype(np.float64))@raw_coefs[t])}
                for method,pred in predicted.items():
                    p,_=forward(c['token_ids'],-pred*b*scale);effect=float(np.log(p[clause].sum()/p[items].sum())-basecontrast)
                    kl=float(np.sum(source*(np.log(np.maximum(source,1e-300))-np.log(np.maximum(p,1e-300)))))
                    row=dict(c,run_id=cfg['run_id'],metric_version='v1',target_seed=t,method=method,predicted_coordinate=pred,source_coordinate=scalar,dose_scale=scale,source_contrast=sourcecontrast,candidate_contrast=effect,
                        contrast_error=effect-sourcecontrast,clause_mass=float(p[clause].sum()),item_mass=float(p[items].sum()),source_to_candidate_kl=kl,source_to_baseline_kl=den,normalized_kl_error=kl/den if den>1e-12 else None)
                    rows.append(row)
                    with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                    replay_arrays[f'candidate_{i}_{t}_{method}']=p.astype(np.float32)
            if time.perf_counter()-numeric>cfg['numeric_budget_seconds']:raise TimeoutError('Numeric budget exceeded')
        np.savez_compressed(run/'probabilities.npz',**replay_arrays)
        checks=dict(cached_hook_max_error=max_hook,baseline_float32_archive_max_error=max_base,source_float32_archive_max_error=max_source)
        summary=dict(forwards=forwards,numeric_seconds=time.perf_counter()-numeric,wall_seconds=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),checks=checks,
            methods={family:{m:dict(median_abs_contrast_error=float(np.median([abs(r['contrast_error']) for r in rows if r['family']==family and r['method']==m])),median_normalized_kl=float(np.median([r['normalized_kl_error'] for r in rows if r['family']==family and r['method']==m])),source_direction_agreement=sum(r['source_contrast']*r['candidate_contrast']>0 for r in rows if r['family']==family and r['method']==m),rows=sum(r['family']==family and r['method']==m for r in rows)) for m in ['fcc','atom','raw']} for family in ['ordinary','constrained']},
            aggregation='Descriptive pooled dependent target-case medians; per-case/per-target retained, not independent seed statistics',metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/replay_f4_and_targets.py',generator_script_sha256=code[0]['sha256'])
        write(run/'metrics.summary.json',summary)
        if max_hook>1e-5 or max_base>1e-7 or max_source>1e-7:raise ValueError('Source identity replay failed')
        write(run/'status.json',dict(status='PASS',forwards=forwards));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=v.errors))
        if not v.ok:raise ValueError(v.errors)
        print(json.dumps(summary))
    except Exception:
        (run/'stderr.log').write_text(traceback.format_exc());write(run/'status.json',dict(status='FAIL',forwards=forwards));raise


if __name__=='__main__':main()
