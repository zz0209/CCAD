"""One long-source hypothesis: discovery-only maps and measured donor effects."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_f4_source_reference_causal import ROOT,np,fixed_support_ridge
from run_r011s1_raw_hook_asset import entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from f4_probability_endpoints import log_prob


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'));run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();write(run/'config.resolved.json',cfg);code=[]
    for rel in ['scripts/run_long_source_correspondence.py','scripts/run_f4_source_reference_causal.py','scripts/run_r011s1_raw_hook_asset.py','scripts/f4_probability_endpoints.py','src/ccad/artifacts.py','src/ccad/activation_contract.py','src/ccad/hook_transport.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='long.source.correspondence.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],milestone='C2-C3',
        evidence_level='long_source_authored_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='independent paired mean; conditional intercept cancels in donor',
        threshold_source_split='source-only that discovery rows; no behavioral fitting',statistics_unit='six shared predicate pairs, one source-target seed pair',device='cuda:0',seeds=[1,2],
        resource_lease='cpu-heavy -> gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
    for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/name).touch()
    write(run/'status.json',dict(status='RUNNING'));inputs=[];rows=[];checks={};env={};forwards=0;error=None
    def checked(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        r=entry(p,'existing CCAD paired assets / standard sklearn BSD implementation','input')
        if expected and r['sha256']!=expected:raise ValueError('Changed input: '+str(p))
        inputs.append(r);return p
    def load(path):return json.loads(checked(path).read_text(encoding='utf-8-sig'))
    def mmap(meta):return np.memmap(checked(meta['path'],meta['sha256']),mode='r',dtype='<u2' if meta['dtype']=='uint16' else '<f4',shape=tuple(meta['shape']))
    try:
        checked(args.config);load('.aris/compute/local-r006b1-env-spec.json');load('.aris/compute/local-f4-sparse-env-spec.json')
        asset=load(cfg['asset_config']);manifest=load(Path(asset['bulk_output_dir'])/'asset_manifest.json');rawmanifest=load(cfg['raw_manifest'])
        tm=load(asset['token_manifest_path']);dm=tm['outputs']['discovery'];tokenpath=checked(ROOT/'runs'/asset['paired_corpus_run']/dm['path'],dm['sha256'])
        paired=[json.loads(s) for s in checked(ROOT/'runs'/asset['paired_corpus_run']/'artifacts/documents.jsonl').read_text().splitlines() if s]
        train=load(cfg['training_documents'])['documents']
        checks['disjoint_sae_training']=all(not({r[k] for r in paired}&{r[k] for r in train}) for k in ['document_id','text_sha256'])
        assert checks['disjoint_sae_training']
        import torch,transformers,sklearn
        from sklearn.linear_model import Lasso
        from sklearn.exceptions import ConvergenceWarning
        from sparsify import SparseCoder
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        tok=transformers.AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True)
        query_token=tok.encode(' that',add_special_tokens=False);assert len(query_token)==1
        tokens=np.memmap(tokenpath,dtype='<u2',mode='r');eligible=np.flatnonzero(tokens==query_token[0]);count=min(cfg['fit_rows'],len(eligible))
        selected=eligible[np.linspace(0,len(eligible)-1,count,dtype=int)];assert len(set(selected))==count and count>16
        dense={};means={};dec={}
        for seed in (1,2):
            dec[seed]=np.array(mmap(next(r for r in manifest['decoders'] if r['seed']==seed)),dtype=float)
            for split in ('mean','discovery'):
                spec=next(r for r in manifest['splits'] if r['split']==split);parts={r['dtype']:mmap(r) for r in spec['files'] if r['seed']==seed}
                ii,aa=parts['uint16'],parts['float32']
                if split=='mean':means[seed]=np.bincount(ii.ravel(),weights=aa.ravel(),minlength=3072)/len(ii)
                else:
                    z=np.zeros((count,3072));np.add.at(z,(np.arange(count)[:,None],ii[selected]),aa[selected]);dense[seed]=z
        rawmeta=next(r for r in rawmanifest['splits'] if r['split']=='discovery');raw=np.array(mmap(rawmeta)[selected],dtype=float)
        x=dense[2]-means[2];y=dense[1][:,cfg['source_atom']]-means[1][cfg['source_atom']];w=np.ones(count)/count
        beta={};diag={};beta['full'],diag['full']=fixed_support_ridge(x,y,w,cfg['ridge'])
        beta['raw'],diag['raw']=fixed_support_ridge(raw,y,w,cfg['ridge'])
        xc=x-x.mean(0);yc=y-y.mean();var=np.mean(xc*xc,axis=0);cross=xc.T@yc/count
        scalar=np.divide(cross,var*(1+cfg['ridge']),out=np.zeros_like(cross),where=var>0)
        losses=np.mean(yc*yc)-2*scalar*cross+scalar*scalar*var
        atom=int(np.argmin(losses));cos=dec[2]@dec[1][cfg['source_atom']]/np.linalg.norm(dec[2],axis=1)/np.linalg.norm(dec[1][cfg['source_atom']]);geom=int(np.argmax(abs(cos)))
        for name,j in [('best_atom',atom),('geometric_atom',geom)]:
            beta[name]=np.zeros(3072);beta[name][j]=scalar[j];diag[name]=dict(atom=j,coefficient=float(scalar[j]),training_error=float(losses[j]),decoder_cosine=float(cos[j]))
        sd=np.sqrt(var);active=sd>1e-12;xx=np.asfortranarray(xc[:,active]/sd[active]);alpha_max=float(np.max(abs(xx.T@yc/count)))
        solver=Lasso(fit_intercept=False,warm_start=True,tol=1e-7,max_iter=5000);path=[];best=None
        for factor in np.geomspace(1.,.01,40):
            solver.set_params(alpha=alpha_max*factor)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);solver.fit(xx,yc)
            ids=np.flatnonzero(active)[np.flatnonzero(solver.coef_)];row=dict(alpha_fraction=float(factor),support=len(ids),converged=not caught,iterations=int(solver.n_iter_))
            if not caught and 0<len(ids)<=cfg['support_budget']:
                b,d=fixed_support_ridge(x[:,ids],y,w,cfg['ridge']);row.update(error=d['weighted_error'],atom_ids=ids.tolist())
                if best is None or d['weighted_error']<best[0]:best=(d['weighted_error'],ids.copy(),b,d)
            path.append(row)
            if len(ids)>32:break
        if best is None:raise RuntimeError('No finite sparse candidate, numerical fit unresolved')
        beta['sparse16']=np.zeros(3072);beta['sparse16'][best[1]]=best[2];diag['sparse16']=dict(**best[3],support=best[1].tolist(),path=path)
        write(run/'fit_metadata.json',dict(source_atom=cfg['source_atom'],source_mean=float(means[1][cfg['source_atom']]),target_mean=means[2].tolist(),eligible_that_rows=len(eligible),selected_rows=selected.tolist(),methods=diag,
            source_conditional_variance=float(np.var(y)),operation='source-decoder aligned donor differences; atom baselines also aligned, not target-native deletion'))
        np.savez_compressed(run/'coefficients.npz',**beta,source_decoder=dec[1][cfg['source_atom']])
        checks['maps_saved_before_consumer']=True;fit_seconds=time.perf_counter()-start
        print(json.dumps(dict(stage='FITS_FROZEN',rows=count,eligible=len(eligible),best_atom=atom,geometric_atom=geom,sparse_support=best[1].tolist(),seconds=fit_seconds)),flush=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(asset['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.config.use_cache=False
        saes={}
        for item in asset['saes']:
            p=checked(Path(item['path'])/'sae.safetensors',item['sha256']);saes[item['seed']]=SparseCoder.load_from_disk(p.parent,device='cuda:0').eval()
        module=model.get_submodule(asset['hook_module_path']);contract=HookPointContract(asset['hook_module_path'],5,'resid_post',768)
        original=load(cfg['probe_config']);cases=[dict(pair=i,role=role,verb=verb,text=original['templates'][0].format(subject=verb)) for i,pair in enumerate(original['pairs']) for role,verb in zip(('report','attitude'),pair)]
        captured=[];prob={};encoded=[]
        def forward(tokens,delta=None):
            nonlocal forwards
            obs={}
            def hook(m,i,out):
                h=extract_primary_hook_tensor(out,contract);obs['h']=h[0,-1].detach().clone()
                if delta is None:return out
                changed=h.clone();changed[0,-1]+=torch.tensor(delta,device='cuda:0',dtype=h.dtype)
                return replace_primary_hook_tensor(out,changed,contract)
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad(): logits=model(tokens,use_cache=False).logits[0,-1].float().cpu().numpy()
            finally:handle.remove()
            forwards+=1;return logits,obs['h']
        batches=[]
        for i,c in enumerate(cases):
            ids=tok.encode(c['text'],add_special_tokens=False);assert ids[-1]==query_token[0];batches.append(torch.tensor([ids],device='cuda:0'))
            logits,h=forward(batches[-1]);prob[f'base_{i}']=np.exp(log_prob(logits[None])[0]);captured.append(h.cpu().numpy())
            zs={}
            with torch.no_grad():
                for s,sae in saes.items():
                    out=sae.encode(h[None]);z=np.zeros(3072);z[out.top_indices[0].cpu().numpy()]=out.top_acts[0].cpu().numpy();zs[s]=z
            encoded.append(zs)
        zero,_=forward(batches[0],np.zeros(768));checks['noop_exact']=np.array_equal(np.exp(log_prob(zero[None])[0]),prob['base_0'])
        direction=dec[1][cfg['source_atom']]
        for i,c in enumerate(cases):
            donor=i^1;truth=float(encoded[donor][1][cfg['source_atom']]-encoded[i][1][cfg['source_atom']]);source_delta=truth*direction
            scale=min(1.,cfg['max_hook_fraction']*np.linalg.norm(captured[i])/max(np.linalg.norm(source_delta),1e-30));source,_=forward(batches[i],scale*source_delta)
            ps=np.exp(log_prob(source[None])[0]);prob[f'source_{i}']=ps;pb=prob[f'base_{i}'];den=float(np.sum(ps*np.log(np.maximum(ps,1e-300)/np.maximum(pb,1e-300))))
            for method,b in beta.items():
                inp=captured[donor]-captured[i] if method=='raw' else encoded[donor][2]-encoded[i][2]
                predicted=float(inp@b);lg,_=forward(batches[i],scale*predicted*direction);pc=np.exp(log_prob(lg[None])[0]);prob[f'{method}_{i}']=pc
                kl=max(0.,float(np.sum(ps*np.log(np.maximum(ps,1e-300)/np.maximum(pc,1e-300)))))
                r=dict(case_id=i,**c,donor=donor,method=method,source_difference=truth,predicted_difference=predicted,preference_direction_matches=bool(truth*predicted>0),
                    scalar_squared_error=(predicted-truth)**2,source_delta_energy=float(truth**2),dose=scale,source_kl=den,candidate_kl=kl,normalized_kl_error=kl/den if den>1e-12 else None)
                rows.append(r)
                with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        write(run/'authored_inputs.json',dict(cases=cases));np.savez_compressed(run/'probabilities.npz',**{k:v.astype(np.float32) for k,v in prob.items()})
        checks.update(expected_forwards=forwards==85,all_cases=len(rows)==60,finite=all(np.isfinite(r['candidate_kl']) for r in rows))
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated())
        if time.perf_counter()-start>cfg['budget_seconds']:raise TimeoutError('Run exceeded wall budget')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    status='PASS' if not error and checks and all(checks.values()) else 'FAIL';write(run/'environment.json',env);write(run/'inputs.json',dict(inputs=inputs))
    summary=dict(status=status,error=error,checks=checks,forwards=forwards,rows=len(rows),wall_seconds=time.perf_counter()-start,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope=cfg['scope'])
    write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary);write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)));print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True)
    return 0 if status=='PASS' and v.ok else 1


if __name__=='__main__':raise SystemExit(main())
