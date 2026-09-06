"""Two non-collinear source contributions, shared support and operation reuse."""
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
    source_seed=int(cfg.get('source_seed',1));target_seed=int(cfg.get('target_seed',2));assert target_seed!=source_seed
    start=time.perf_counter();write(run/'config.resolved.json',cfg);code=[]
    for rel in ['scripts/run_long_source_pair.py','scripts/run_f4_source_reference_causal.py','scripts/fit_f4_joint_sparse.py','scripts/run_r011s1_raw_hook_asset.py','scripts/f4_probability_endpoints.py','src/ccad/artifacts.py','src/ccad/activation_contract.py','src/ccad/hook_transport.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='long.source.correspondence.v1',run_id=cfg['run_id'],run_parent='F4',purpose=cfg['purpose'],milestone='C2-C3',
        evidence_level=cfg.get('evidence_level','long_source_authored_development'),started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='independent paired mean; conditional intercept cancels in donor',
        threshold_source_split='source-only specified-token discovery rows; no behavioral fitting',statistics_unit='document/predicate pairs sharing one source and one target seed; reciprocal operations dependent',device='cuda:0',seeds=[source_seed,target_seed],
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
        if cfg.get('target_asset_config'):
            extra=load(cfg['target_asset_config'])
            for key in ('token_manifest_sha256','paired_corpus_run','model_revision','hook_module_path','num_latents','k'):
                assert asset[key]==extra[key], key
            em=load(Path(extra['bulk_output_dir'])/'asset_manifest.json')
            manifest['decoders']+=em['decoders']
            for spec in manifest['splits']:
                if spec['split'] not in ('mean','discovery'):continue
                match=next(r for r in em['splits'] if r['split']==spec['split'])
                spec['files']+=match['files']
            asset['saes']+=extra['saes']
        assert len([r for r in asset['saes'] if r['seed']==target_seed])==1
        import torch,transformers,sklearn
        from sparsify import SparseCoder
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        tok=transformers.AutoTokenizer.from_pretrained(asset['model_local_dir'],local_files_only=True)
        query_token=tok.encode(cfg.get('query_token',' that'),add_special_tokens=False);assert len(query_token)==1
        source_checkpoint_sae=None
        if cfg.get('source_checkpoint'):
            scp=cfg['source_checkpoint'];checked(cfg['checkpoint_registry'])
            source_weight=checked(Path(scp['path'])/'sae.safetensors',scp['sha256'])
            source_checkpoint_sae=SparseCoder.load_from_disk(source_weight.parent,device='cuda:0').eval()
            checks['source_checkpoint_loaded']=True
        if cfg.get('frozen_map'):
            freeze=cfg['frozen_map'];parent=Path(freeze['path'])
            pc=json.loads(checked(parent/'config.resolved.json',freeze['config_sha256']).read_text())
            for key in ('source_seed','target_seed','source_atoms','operators','max_hook_fraction','asset_config','target_asset_config'):
                assert cfg[key]==pc[key], key
            assert cfg.get('source_checkpoint')==pc.get('source_checkpoint')
            assert cfg.get('wrong_query_control')==pc.get('wrong_query_control')
            assert load(parent/'status.json')['status']=='PASS'
            for spec in cfg.get('confirmation_inputs',[]):checked(spec['path'],spec['sha256'])
            checked(cfg['evaluation_inputs'],cfg['evaluation_inputs_sha256'])
            with np.load(checked(parent/'coefficients.npz',freeze['coefficients_sha256']),allow_pickle=False) as ar:
                direction=np.array(ar['source_decoder']);beta={k:np.array(ar[k]) for k in ar.files if k!='source_decoder'}
            atoms=cfg['source_atoms'];dg=direction@direction.T;checkpoint_sae=None
            item=next(r for r in asset['saes'] if r['seed']==target_seed)
            if pc.get('target_checkpoint'):
                if cfg.get('use_frozen_target_checkpoint',False):
                    assert cfg['target_checkpoint']==pc['target_checkpoint']
                    cp=pc['target_checkpoint'];weight=checked(Path(cp['path'])/'sae.safetensors',cp['sha256'])
                    checkpoint_sae=SparseCoder.load_from_disk(weight.parent,device='cuda:0').eval()
                    checks['frozen_target_checkpoint_loaded']=True
                else:
                    assert item['sha256']==pc['target_checkpoint']['sha256']
            else:
                weight=Path(item['path'])/'sae.safetensors';weight=weight if weight.is_absolute() else ROOT/weight
                bound=next(r for r in load(parent/'inputs.json')['inputs'] if Path(r['path']).resolve()==weight.resolve())
                assert item['sha256']==bound['sha256']
            write(run/'fit_metadata.json',dict(mode='frozen_application',parent=freeze,fit_calls=0,means_cancel_in_differences=True))
            np.savez_compressed(run/'coefficients.npz',**beta,source_decoder=direction)
            checks['frozen_no_refit']=True;checks['maps_saved_before_consumer']=True
            print(json.dumps(dict(stage='FROZEN_MAP_LOADED',methods=list(beta),fit_calls=0)),flush=True)
        else:
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
            query_token=tok.encode(cfg.get('query_token',' that'),add_special_tokens=False);assert len(query_token)==1
            from fit_f4_joint_sparse import fit_joint
            atoms=cfg['source_atoms'];assert len(atoms)==2 and len(set(atoms))==2
            tokens=np.memmap(tokenpath,dtype='<u2',mode='r');eligible=np.flatnonzero(tokens==query_token[0]);count=min(cfg['fit_rows'],len(eligible))
            selected=eligible[np.linspace(0,len(eligible)-1,count,dtype=int)];assert count>16
            dense={};means={};dec={};checkpoint_sae=None
            if cfg.get('target_checkpoint'):
                cp=cfg['target_checkpoint'];checked(cfg['checkpoint_registry'])
                cpfile=checked(Path(cp['path'])/'sae.safetensors',cp['sha256']);checkpoint_sae=SparseCoder.load_from_disk(cpfile.parent,device='cuda:0').eval()
                dec[target_seed]=checkpoint_sae.W_dec.detach().cpu().numpy().astype(float)
                def encode_rows(hook_rows,sae=checkpoint_sae):
                    result=np.zeros((len(hook_rows),3072))
                    with torch.no_grad():
                        for off in range(0,len(hook_rows),512):
                            inp=torch.tensor(np.array(hook_rows[off:off+512]),device='cuda:0',dtype=torch.float32);en=sae.encode(inp)
                            ii=en.top_indices.cpu().numpy();aa=en.top_acts.cpu().numpy();np.add.at(result,(np.arange(off,off+len(ii))[:,None],ii),aa)
                    return result
                meanraw=mmap(next(r for r in rawmanifest['splits'] if r['split']=='mean'));total=np.zeros(3072)
                for off in range(0,len(meanraw),512):total+=encode_rows(meanraw[off:off+512]).sum(0)
                means[target_seed]=total/len(meanraw)
                discraw=mmap(next(r for r in rawmanifest['splits'] if r['split']=='discovery'));dense[target_seed]=encode_rows(discraw[selected])
                write(run/'target_encoding.json',dict(checkpoint=cp,mean_rows=len(meanraw),discovery_rows=len(selected),no_new_base_forward=True,no_full_code_cache_written=True))
            if source_checkpoint_sae is not None:
                assert checkpoint_sae is not None, 'Explicit source checkpoint requires explicit target checkpoint'
                dec[source_seed]=source_checkpoint_sae.W_dec.detach().cpu().numpy().astype(float)
                total=np.zeros(3072)
                for off in range(0,len(meanraw),512):total+=encode_rows(meanraw[off:off+512],source_checkpoint_sae).sum(0)
                means[source_seed]=total/len(meanraw);dense[source_seed]=encode_rows(discraw[selected],source_checkpoint_sae)
                write(run/'source_encoding.json',dict(checkpoint=cfg['source_checkpoint'],mean_rows=len(meanraw),discovery_rows=len(selected),no_new_base_forward=True))
            for seed in ((source_seed,) if checkpoint_sae is not None else (source_seed,target_seed)):
                if seed in dec:continue

                dec[seed]=np.array(mmap(next(r for r in manifest['decoders'] if r['seed']==seed)),dtype=float)
                for split in ('mean','discovery'):
                    spec=next(r for r in manifest['splits'] if r['split']==split);parts={r['dtype']:mmap(r) for r in spec['files'] if r['seed']==seed}
                    ii,aa=parts['uint16'],parts['float32']
                    if split=='mean':means[seed]=np.bincount(ii.ravel(),weights=aa.ravel(),minlength=3072)/len(ii)
                    else:
                        z=np.zeros((count,3072));np.add.at(z,(np.arange(count)[:,None],ii[selected]),aa[selected]);dense[seed]=z
            x=dense[target_seed]-means[target_seed];y=dense[source_seed][:,atoms]-means[source_seed][atoms];w=np.ones(count)/count
            direction=dec[source_seed][atoms];dg=direction@direction.T
            beta={};parents=[];baseline_diagnostics={}
            for j,parent in enumerate(cfg.get('parents',[])):
                path=ROOT/parent['path'];pc=load(path/'config.resolved.json');assert pc['source_atom']==atoms[j] and pc['source_seed']==source_seed
                if checkpoint_sae is None and not cfg.get('recompute_target_baselines'):assert pc['target_seed']==target_seed
                with np.load(checked(path/'coefficients.npz',parent['sha256']),allow_pickle=False) as arr:
                    assert np.array_equal(arr['source_decoder'],direction[j])
                    parents.append({k:np.array(arr[k]) for k in ('full','raw','best_atom','geometric_atom','sparse16')})
            if checkpoint_sae is None and not cfg.get('recompute_target_baselines'):
                for name in parents[0]:beta[name]=np.column_stack([r[name] for r in parents])
            else:
                if parents:beta['raw']=np.column_stack([r['raw'] for r in parents])
                else:
                    assert source_checkpoint_sae is not None
                    raw_x=np.array(discraw[selected],dtype=float)-np.mean(meanraw,axis=0,dtype=float)
                    beta['raw']=np.column_stack([fixed_support_ridge(raw_x,y[:,j],w,cfg['ridge'])[0] for j in range(2)])
                beta['full']=np.zeros((3072,2));beta['best_atom']=np.zeros((3072,2));beta['geometric_atom']=np.zeros((3072,2))
                xc=x-x.mean(0);yc=y-y.mean(0);var=np.mean(xc*xc,axis=0)
                for j in range(2):
                    beta['full'][:,j],fd=fixed_support_ridge(x,y[:,j],w,cfg['ridge']);cross=xc.T@yc[:,j]/count
                    scalar=np.divide(cross,var*(1+cfg['ridge']),out=np.zeros_like(cross),where=var>0);loss=np.mean(yc[:,j]**2)-2*scalar*cross+scalar**2*var
                    atom=int(np.argmin(loss));cos=dec[target_seed]@direction[j]/np.linalg.norm(dec[target_seed],axis=1)/np.linalg.norm(direction[j]);geom=int(np.argmax(abs(cos)))
                    beta['best_atom'][atom,j]=scalar[atom];beta['geometric_atom'][geom,j]=scalar[geom];baseline_diagnostics[str(atoms[j])]=dict(best_atom=atom,geometric_atom=geom,full=fd)
            beta['shared16'],intercept,diag=fit_joint(x,y,w,cfg['joint_fit'])
            if cfg.get('operation_energy_fit'):
                energy_cfg=dict(cfg['joint_fit'],output_metric='decoder_energy',output_metric_diagonal=np.diag(dg).tolist())
                beta['energy16'],energy_intercept,energy_diag=fit_joint(x,y,w,energy_cfg)
                write(run/'energy_support_selection.json',dict(fit=energy_diag,intercept=energy_intercept.tolist(),scope='Equal A/B/sum/difference expected hook-energy surrogate, not KL or worst-operator optimum'))
            support=np.flatnonzero(np.linalg.norm(beta['shared16'],axis=1)>0)
            supports={'same_support_ridge':support}
            if cfg.get('operation_energy_fit'):supports['energy_support_ridge']=np.flatnonzero(np.linalg.norm(beta['energy16'],axis=1)>0)
            for name,base in [('dynamic_pair_ridge','best_atom'),('geometric_pair_ridge','geometric_atom')]:
                supports[name]=np.flatnonzero(np.linalg.norm(beta[base],axis=1)>0)
            refits={}
            for name,ids in supports.items():
                beta[name]=np.zeros((3072,2));refits[name]=dict(support=ids.tolist(),outputs=[])
                for j in range(2):
                    b,d=fixed_support_ridge(x[:,ids],y[:,j],w,cfg['ridge']);beta[name][ids,j]=b;refits[name]['outputs'].append(d)
            if cfg.get('separate_support_fit'):
                from fit_f4_joint_sparse import select_separate_supports
                own,sepdiag=select_separate_supports(x,y,w,cfg['separate_support_fit'])
                union=np.unique(np.concatenate(own))
                checks['separate_union_budget']=len(union)<=cfg['support_budget']
                for name,ids_by_output in [('separate8_ridge',own),('union16_ridge',[union,union])]:
                    beta[name]=np.zeros((3072,2));refits[name]=dict(supports=[ids.tolist() for ids in ids_by_output],outputs=[])
                    for j,ids in enumerate(ids_by_output):
                        b,d=fixed_support_ridge(x[:,ids],y[:,j],w,cfg['ridge']);beta[name][ids,j]=b;refits[name]['outputs'].append(d)
                write(run/'separate_support_selection.json',sepdiag)
            if cfg.get('wrong_query_control'):
                assert cfg['wrong_query_control']=='swap_components_energy_matched_per_operator'
                beta['wrong_query']=beta['shared16'][:,::-1].copy()
            yc=y-y.mean(0);cov=yc.T@yc/count
            write(run/'fit_metadata.json',dict(baseline_diagnostics=baseline_diagnostics,source_atoms=atoms,source_seed=source_seed,target_seed=target_seed,source_decoder_gram=dg.tolist(),source_decoder_eigenvalues=np.linalg.eigvalsh(dg).tolist(),source_covariance=cov.tolist(),source_covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(),joint_fit=diag,shared_intercept=intercept.tolist(),refits=refits,source_mean=means[source_seed][atoms].tolist(),target_mean=means[target_seed].tolist(),selected_rows=selected.tolist(),eligible_query_rows=len(eligible),operation='source-decoder aligned two-component donor family, common dose across operators'))
            np.savez_compressed(run/'coefficients.npz',**beta,source_decoder=direction)
            checks['maps_saved_before_consumer']=True
            if cfg.get('calibration_fit_diagnostic'):
                cm=tm['outputs']['calibration'];ct=checked(ROOT/'runs'/asset['paired_corpus_run']/cm['path'],cm['sha256'])
                ctoken=np.memmap(ct,dtype='<u2',mode='r');ce=np.flatnonzero(ctoken==query_token[0]);nc=min(cfg['fit_rows'],len(ce));cr=ce[np.linspace(0,len(ce)-1,nc,dtype=int)]
                craw=mmap(next(r for r in rawmanifest['splits'] if r['split']=='calibration'))
                cx=encode_rows(craw[cr]);cy=encode_rows(craw[cr],source_checkpoint_sae)[:,atoms]
                stats={};csy=np.std(y,axis=0);op=np.asarray(list(cfg['operators'].values()));om=dg*(op.T@op/len(op))
                assert np.allclose(om,.75*np.diag(np.diag(dg))), 'This diagnostic assumes the symmetric four-operator family'
                for name,b in beta.items():
                    if name=='wrong_query':continue
                    design=np.asarray(craw[cr],dtype=float) if name=='raw' else cx
                    residual=cy-design@b;residual-=residual.mean(0);covr=residual.T@residual/nc
                    donor_gram=2*covr*dg
                    stats[name]=dict(support=int(np.count_nonzero(np.linalg.norm(b,axis=1))),component_variance=np.diag(covr).tolist(),standardized_error=float(np.sum(np.diag(covr)/csy**2)),mean_iid_donor_hook_error=float(2*np.sum(covr*om)),operator_iid_donor_hook_error={key:float(np.asarray(theta)@donor_gram@np.asarray(theta)) for key,theta in cfg['operators'].items()})
                write(run/'calibration_diagnostic.json',dict(rows=cr.tolist(),eligible_rows=len(ce),count=nc,methods=stats,scope='Existing calibration empirical iid donor covariance; centered residual removes intercept; no LM endpoint, no common dose, no document independence claim'))
                checks['calibration_no_base_forward']=forwards==0
            print(json.dumps(dict(stage='FITS_FROZEN',support=support.tolist(),source_decoder_eigenvalues=np.linalg.eigvalsh(dg).tolist(),source_covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(),seconds=time.perf_counter()-start)),flush=True)
        if cfg.get('fit_only'):
            checks['fit_only_no_base_forward']=forwards==0
            env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated())
        else:
            model=transformers.AutoModelForCausalLM.from_pretrained(asset['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0');model.config.use_cache=False
            saes={target_seed:checkpoint_sae} if checkpoint_sae is not None else {}
            if source_checkpoint_sae is not None:saes[source_seed]=source_checkpoint_sae
            for item in asset['saes']:
                if item['seed'] in saes:continue
                if item['seed'] not in (source_seed,target_seed):continue
                p=checked(Path(item['path'])/'sae.safetensors',item['sha256']);saes[item['seed']]=SparseCoder.load_from_disk(p.parent,device='cuda:0').eval()
            assert np.array_equal(saes[source_seed].W_dec.detach().cpu().numpy()[atoms].astype(float),direction)
            module=model.get_submodule(asset['hook_module_path']);contract=HookPointContract(asset['hook_module_path'],5,'resid_post',768)
            original=load(cfg['probe_config'])
            cases=load(cfg['evaluation_inputs'])['cases'] if cfg.get('evaluation_inputs') else [dict(pair=i,role=role,verb=verb,text=original['templates'][0].format(subject=verb)) for i,pair in enumerate(original['pairs']) for role,verb in zip(('report','attitude'),pair)]
            assert len(cases)>0 and len(cases)%2==0
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
                ids=c['token_ids'] if 'token_ids' in c else tok.encode(c['text'],add_special_tokens=False);assert ids[-1]==query_token[0];batches.append(torch.tensor([ids],device='cuda:0'))
                logits,h=forward(batches[-1]);prob[f'base_{i}']=np.exp(log_prob(logits[None])[0]);captured.append(h.cpu().numpy())
                zs={}
                with torch.no_grad():
                    for s,sae in saes.items():
                        out=sae.encode(h[None]);z=np.zeros(3072);z[out.top_indices[0].cpu().numpy()]=out.top_acts[0].cpu().numpy();zs[s]=z
                encoded.append(zs)
            zero,_=forward(batches[0],np.zeros(768));checks['noop_exact']=np.array_equal(np.exp(log_prob(zero[None])[0]),prob['base_0'])
            operators={k:np.asarray(v,dtype=float) for k,v in cfg['operators'].items()};gram_records=[]
            for i,c in enumerate(cases):
                donor=i^1;truth=encoded[donor][source_seed][atoms]-encoded[i][source_seed][atoms]
                deltas={name:(truth*theta)@direction for name,theta in operators.items()}
                common_dose=min(1.,cfg['max_hook_fraction']*np.linalg.norm(captured[i])/max(max(np.linalg.norm(v) for v in deltas.values()),1e-30))
                predicted={method:(captured[donor]-captured[i] if method=='raw' else encoded[donor][target_seed]-encoded[i][target_seed])@b for method,b in beta.items()}
                for method,pred in predicted.items():
                    if method=='wrong_query':continue
                    residual=(pred-truth)*common_dose;gram_records.append(dict(case_id=i,method=method,error_gram=(residual[:,None]*dg*residual[None,:]).tolist()))
                for op,theta in operators.items():
                    source,_=forward(batches[i],common_dose*deltas[op]);ps=np.exp(log_prob(source[None])[0]);prob[f'source_{op}_{i}']=ps
                    pb=prob[f'base_{i}'];den=max(0.,float(np.sum(ps*np.log(np.maximum(ps,1e-300)/np.maximum(pb,1e-300)))))
                    for method,pred in predicted.items():
                        control_scale=1.
                        if method=='wrong_query':
                            raw_energy=np.linalg.norm((pred*theta)@direction)
                            control_scale=np.linalg.norm(deltas[op])/raw_energy if raw_energy>1e-30 else 0.
                            pred=pred*control_scale
                        edit=common_dose*(pred*theta)@direction;lg,_=forward(batches[i],edit);pc=np.exp(log_prob(lg[None])[0]);prob[f'{method}_{op}_{i}']=pc
                        kl=max(0.,float(np.sum(ps*np.log(np.maximum(ps,1e-300)/np.maximum(pc,1e-300)))))
                        residual=(pred-truth)*common_dose;g=residual[:,None]*dg*residual[None,:];ve=float(np.sum((edit-common_dose*deltas[op])**2))
                        assert np.isclose(ve,theta@g@theta,rtol=1e-9,atol=1e-12)
                        r=dict(case_id=i,**c,donor=donor,operator=op,method=method,source_difference=truth.tolist(),predicted_difference=pred.tolist(),source_activation=encoded[i][source_seed][atoms].tolist(),common_dose=float(common_dose),vector_squared_error=ve,source_delta_energy=float(np.sum((common_dose*deltas[op])**2)),source_kl=den,candidate_kl=kl,normalized_kl_error=kl/den if den>1e-12 else None)
                        if method=='wrong_query':r['control_energy_scale']=float(control_scale);r['control_energy_matched']=bool(np.isclose(np.linalg.norm(edit),np.linalg.norm(common_dose*deltas[op]),rtol=1e-8,atol=1e-12))
                        rows.append(r)
                        with (run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            write(run/'error_grams.json',dict(records=gram_records,scope='Pointwise source-aligned vector error identity; excludes per-operator energy-matched wrong-query control which is not one linear operation family; empirical mean is not an unseen-context guarantee'))
            write(run/'authored_inputs.json',dict(cases=cases))
            if cfg.get('probability_storage')=='observed_and_probe_tokens':
                kept=sorted({int(c['observed_next_token_id']) for c in cases if c.get('observed_next_token_id') is not None}|set(cfg['probe_token_ids']))
                np.savez_compressed(run/'probabilities.npz',retained_token_ids=np.asarray(kept),**{k:v[kept].astype(np.float32) for k,v in prob.items()})
                write(run/'probability_storage.json',dict(scope='Exact token marginals retained; full-vocabulary KL computed before slicing and stored per row',retained_token_ids=kept,full_vocabulary_probabilities_saved=False))
            else:np.savez_compressed(run/'probabilities.npz',**{k:v.astype(np.float32) for k,v in prob.items()})
            checks.update(expected_forwards=forwards==len(cases)*(1+len(operators)*(1+len(beta)))+1,all_cases=len(rows)==len(cases)*len(beta)*len(operators),finite=all(np.isfinite(r['candidate_kl']) for r in rows))
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
