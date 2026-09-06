"""Fixed-basis symmetric dose diagnostic on all exposed natural and cases."""
import argparse,json,os,platform,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from f4_interpretation_endpoint import contrast


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);cfg=json.loads(ap.parse_args().config.read_text());start=time.perf_counter()
    run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);write(run/'config.resolved.json',cfg)
    for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
    code=[]
    for rel in ['scripts/probe_f4_context_response.py','scripts/f4_interpretation_endpoint.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.context.response.v1',run_id=cfg['run_id'],run_parent='F4',purpose='Separate context-dependent local output response from larger source-dose effects',milestone='C3',evidence_level='exposed_natural_context_development_diagnostic',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='not used; fixed original source direction only',threshold_source_split='fractions frozen before forwards, on exposed natural cases',statistics_unit='eight distinct documents, shared fixed source direction, four dose operations each',device=cfg['device'],seeds=[5],resource_lease='gpu-0 resource_manager.run',resource_lease_reason='48 bounded forwards; no fitting or SAE encoding'))
    write(run/'status.json',dict(status='RUNNING'));inputs=[];rows=[];forwards=0
    def checked(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;e=entry(p,'fixed source evidence or model asset','input','internal/Apache-2.0 model; no audit')
        if expected and e['sha256']!=expected:raise ValueError(f'Hash mismatch: {p}')
        inputs.append(e);return p
    try:
        write(run/'resource_status_at_start.json',json.loads(subprocess.check_output([sys.executable,str(ROOT.parent/'.resource_manager/resource_manager.py'),'status','--resource','gpu-0'],text=True)))
        parent=ROOT/cfg['parent_run'];old=json.loads(checked(parent/'config.resolved.json',cfg['parent_config_sha256']).read_text());assets=json.loads(checked(old['asset_config']).read_text())
        saved=[json.loads(x) for x in checked(parent/'metrics.raw.jsonl',cfg['parent_raw_sha256']).read_text().splitlines()]
        arr=np.load(checked(parent/'source_arrays.npz',cfg['parent_arrays_sha256']));auth=json.loads(checked(parent/'authored_inputs.json',cfg['parent_authored_sha256']).read_text())
        factors=np.load(checked(old['factors_path'],old['factors_sha256']));ix=np.flatnonzero((factors['source_seed']==old['source_seed'])&(factors['source_atom']==old['source_atom']));b=factors['source_basis'][ix[0],:,0].astype(float)
        if abs(np.linalg.norm(b)-1)>1e-6 or not all(np.array_equal(b,factors['source_basis'][i,:,0]) for i in ix):raise ValueError('Source direction identity differs')
        checked(Path(assets['model_local_dir'])/'config.json');checked('.aris/compute/local-r006b1-env-spec.json')
        selected=[i for i,c in enumerate(auth['cases']) if c['conjunction']==cfg['conjunction']]
        if len(selected)!=8:raise ValueError('Expected all eight and cases')
        import torch,transformers
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        model=transformers.AutoModelForCausalLM.from_pretrained(assets['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(cfg['device']);model.config.use_cache=False
        write(run/'environment.json',dict(python=platform.python_version(),os=platform.platform(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
        write(run/'inputs.json',dict(inputs=inputs));contract=HookPointContract(assets['hook_module_path'],5,'resid_post',768);module=model.get_submodule(assets['hook_module_path']);numeric=time.perf_counter();probs={};maxnoop=maxhook=maxbase=0.
        def forward(tokens,delta=None):
            nonlocal forwards
            if forwards>=cfg['maximum_forwards']:raise ValueError('Forward budget exceeded')
            hidden=[]
            def hook(m,i,out):
                h=extract_primary_hook_tensor(out,contract);hidden.append(h[0,-1].detach().cpu().numpy().copy())
                if delta is None:return out
                x=h.clone();x[0,-1]+=torch.tensor(delta,device=x.device,dtype=x.dtype);return replace_primary_hook_tensor(out,x,contract)
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad():lg=model(torch.tensor([tokens],device=cfg['device'])).logits[0,-1].cpu().numpy().astype(float)
                forwards+=1;p=np.exp(lg-lg.max());return p/p.sum(),hidden[0]
            finally:handle.remove()
        for i in selected:
            c=auth['cases'][i];base,h=forward(c['token_ids']);noop,_=forward(c['token_ids'],np.zeros_like(b));hn=float(np.linalg.norm(h.astype(float)));baseline=contrast(base,auth['clause_token_ids'],None)
            maxnoop=max(maxnoop,float(np.max(abs(noop-base))));maxhook=max(maxhook,float(np.max(abs(h-arr[f'hook_{i}']))));maxbase=max(maxbase,float(np.max(abs(base-arr[f'prob_{i}_baseline']))));probs[f'base_{i}']=base.astype(np.float32);effects={}
            for frac in cfg['fractions']:
                for sign in [-1,1]:
                    step=frac*sign;p,_=forward(c['token_ids'],step*hn*b);key=f'{step:+.4f}';probs[f'prob_{i}_{key}']=p.astype(np.float32);effects[key]=contrast(p,auth['clause_token_ids'],None)-baseline
            slopes={str(frac):(effects[f'{frac:+.4f}']-effects[f'{-frac:+.4f}'])/(2*frac*hn) for frac in cfg['fractions']}
            row=dict(c,run_id=cfg['run_id'],metric_version='v1',hook_norm=hn,original_source_effect=saved[i]['effects']['source_swap']['contrast_delta'],original_source_step=-saved[i]['source_difference_coordinate']*saved[i]['dose_scale'],effects=effects,central_directional_slopes=slopes)
            rows.append(row)
            with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            if time.perf_counter()-numeric>cfg['numeric_budget_seconds']:raise TimeoutError('Numeric budget exceeded')
        np.savez_compressed(run/'probabilities.npz',**probs)
        summary=dict(forwards=forwards,numeric_seconds=time.perf_counter()-numeric,wall_seconds=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),max_noop_error=maxnoop,max_cached_hook_error=maxhook,max_cached_baseline_error=maxbase,rows=[dict(pair_index=r['pair_index'],family=r['family'],original_source_effect=r['original_source_effect'],effects=r['effects'],slopes=r['central_directional_slopes']) for r in rows],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/probe_f4_context_response.py',generator_script_sha256=code[0]['sha256'])
        write(run/'metrics.summary.json',summary)
        if maxnoop>1e-7 or maxhook>1e-5 or maxbase>1e-7:raise ValueError('Baseline identity failed')
        write(run/'status.json',dict(status='PASS',forwards=forwards));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=v.errors))
        if not v.ok:raise ValueError(v.errors)
        print(json.dumps(summary))
    except Exception:
        (run/'stderr.log').write_text(traceback.format_exc());write(run/'status.json',dict(status='FAIL',forwards=forwards));raise


if __name__=='__main__':main()
