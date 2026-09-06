"""Bounded fixed-atom source activation and signed native-effect feasibility."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.artifacts import sha256, validate_run_directory
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();write(run/'config.resolved.json',cfg)
    for n in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/n).touch()
    code=[]
    for rel in ['scripts/run_native_explanation_probe.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='native.explanation.probe.v1',run_id=cfg['run_id'],run_parent='R011-NR1',
        purpose='Falsify source-native we-context and output-reference hypotheses on fixed new authored inputs',milestone='C2-C3-interpretability',
        evidence_level='authored_source_feasibility_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),
        config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,
        candidate_family_frozen=True,mean_constants_source_split='not_applicable_native_actual_code',threshold_source_split='fixed config before inference',
        statistics_unit='four authored templates and matched subject variants; not independent documents',device=cfg['device'],seeds=[cfg['source_seed']],
        resource_lease='gpu-0 resource_manager.run',resource_lease_reason='64 bounded model forwards; four CPU threads, no heavy CPU fit or whole-disk work'))
    write(run/'status.json',dict(status='RUNNING'));inputs=[];rows=[];arrays={};forwards=0
    def checked(path):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        inputs.append(entry(path,'existing locked source asset or fixed authored config','input','internal/Apache-2.0 model; no audit'))
        return path
    try:
        lease=json.loads(subprocess.check_output([sys.executable,str(ROOT.parent/'.resource_manager/resource_manager.py'),'status','--resource','gpu-0'],text=True))
        write(run/'resource_status_at_start.json',lease)
        old=json.loads(checked(cfg['parent_asset_config']).read_text()); spec=next(s for s in old['saes'] if s['seed']==cfg['source_seed'])
        sae_path=ROOT/spec['path'];assert sha256(checked(sae_path/'sae.safetensors'))==spec['sha256'];checked(sae_path/'cfg.json')
        checked(Path(old['model_local_dir'])/'config.json');checked(Path(old['model_local_dir'])/'tokenizer.json');checked('.aris/compute/local-r006b1-env-spec.json')
        import torch
        import transformers
        from sparsify import SparseCoder
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        model=transformers.AutoModelForCausalLM.from_pretrained(old['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(cfg['device'])
        model.config.use_cache=False
        tokenizer=transformers.AutoTokenizer.from_pretrained(old['model_local_dir'],local_files_only=True)
        sae=SparseCoder.load_from_disk(sae_path,device=cfg['device']).eval()
        write(run/'environment.json',dict(python=platform.python_version(),os=platform.platform(),numpy=np.__version__,torch=torch.__version__,
            transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),sae_framework='sparsify '+old['sparsify_commit']))
        contract=HookPointContract(old['hook_module_path'],5,'resid_post',768);module=model.get_submodule(old['hook_module_path'])
        first=[tokenizer.encode(t,add_special_tokens=False) for t in cfg['first_person_plural_tokens']]
        third=[tokenizer.encode(t,add_special_tokens=False) for t in cfg['third_person_plural_tokens']]
        if any(len(t)!=1 for t in first+third):raise ValueError('Fixed contrast must consist of single tokens')
        first=[t[0] for t in first];third=[t[0] for t in third];write(run/'contrast_tokens.json',dict(first=first,third=third))
        prepared=[dict(template=i,subject=s,text=t.format(subject=s),token_ids=tokenizer.encode(t.format(subject=s),add_special_tokens=False)) for i,t in enumerate(cfg['templates']) for s in cfg['subjects']]
        write(run/'authored_inputs.json',dict(cases=prepared));write(run/'inputs.json',dict(inputs=inputs));numeric=time.perf_counter()
        maxnoop=0.0
        for caseidx,case in enumerate(prepared):
            tokens=torch.tensor([case['token_ids']],device=cfg['device']);hidden=[]
            def forward(delta=None,capture=False):
                nonlocal forwards
                if forwards>=cfg['maximum_forwards']:raise ValueError('Forward budget exceeded')
                def hook(m,i,out):
                    h=extract_primary_hook_tensor(out,contract)
                    if capture:hidden.append(h.detach().clone())
                    if delta is None:return out
                    changed=h.clone();changed[:,-1,:]+=delta
                    return replace_primary_hook_tensor(out,changed,contract)
                handle=module.register_forward_hook(hook)
                try:
                    with torch.no_grad(): result=model(tokens).logits[0,-1].float().cpu().numpy()
                    forwards+=1;return result
                finally:handle.remove()
            logits={'baseline':forward(capture=True)};h=hidden[0][0,-1]
            with torch.no_grad(): enc=sae.encode(h[None]);mask=enc.top_indices[0]==cfg['source_atom'];act=enc.top_acts[0][mask].sum()
            vec=act*sae.W_dec[cfg['source_atom']].detach();hn=float(h.norm());natural=float(vec.norm())/hn
            scale=min(1.0,cfg['maximum_hook_fraction']/natural) if natural else 1.0;delta=vec*scale
            logits['noop']=forward(torch.zeros_like(delta));maxnoop=max(maxnoop,float(np.max(abs(logits['noop']-logits['baseline']))))
            logits['native_remove']=forward(-delta);logits['native_add']=forward(delta)
            probabilities={}
            for op,lg in logits.items():
                x=lg.astype(np.float64);x-=x.max();pp=np.exp(x);pp/=pp.sum();probabilities[op]=pp;arrays[f'case_{caseidx}_{op}']=pp.astype(np.float32)
            basep=probabilities['baseline'];source_effects={}
            for op,p in probabilities.items():
                effect=float(np.log(p[first].sum()/p[third].sum())-np.log(basep[first].sum()/basep[third].sum()))
                order=np.lexsort((np.arange(len(p)),-abs(p-basep)))[:8]
                source_effects[op]=dict(contrast_delta=effect,first_mass=float(p[first].sum()),third_mass=float(p[third].sum()),
                    kl_to_base=float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(basep,1e-300))))),
                    top_changed=[dict(id=int(j),token=tokenizer.decode([int(j)]),delta=float(p[j]-basep[j])) for j in order])
            row=dict(case,run_id=cfg['run_id'],metric_version='v1',activation=float(act),natural_hook_fraction=natural,dose_scale=scale,effects=source_effects)
            rows.append(row)
            with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
            if time.perf_counter()-numeric>cfg['numeric_budget_seconds']:raise TimeoutError('Numeric budget exceeded')
        np.savez_compressed(run/'probabilities.npz',**arrays)
        summary=dict(forwards=forwards,numeric_seconds=time.perf_counter()-numeric,wall_seconds=time.perf_counter()-start,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),max_noop_absolute_error=maxnoop,
            subjects={s:dict(activations=[r['activation'] for r in rows if r['subject']==s],
                remove_contrast=[r['effects']['native_remove']['contrast_delta'] for r in rows if r['subject']==s],
                add_contrast=[r['effects']['native_add']['contrast_delta'] for r in rows if r['subject']==s]) for s in cfg['subjects']},
            metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_native_explanation_probe.py',generator_script_sha256=code[0]['sha256'])
        write(run/'metrics.summary.json',summary)
        if maxnoop>1e-6:raise ValueError('Noop mismatch')
        write(run/'status.json',dict(status='PASS',forwards=forwards));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=v.errors))
        if not v.ok:raise ValueError(v.errors)
        print(json.dumps(summary))
    except Exception:
        (run/'stderr.log').write_text(traceback.format_exc());write(run/'status.json',dict(status='FAIL',forwards=forwards));raise


if __name__=='__main__':main()
