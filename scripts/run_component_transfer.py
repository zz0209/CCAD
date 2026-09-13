"""Transfer contrast-defined source components and their unfitted union.

The primary procedure discovers source components before fitting targets,
then freezes every map before evaluating old and new task subsets. The
optional raw-control mode reuses the exact frozen parent source/components.
"""
from __future__ import annotations
import os,sys,json,time,platform,argparse,traceback
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_causalgym_multisite import MultisiteWork,ROOT,write
from ccad.artifacts import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.config.read_text())
    source_files=['scripts/run_component_transfer.py','scripts/fit_component_correspondence.py','scripts/component_raw_controls.py','scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']
    if cfg.get('reuse_consumer_parent'):source_files.append('scripts/functional_reuse_consumer.py')
    if cfg.get('path_midpoints') or cfg.get('path_ensemble_parent'):source_files.append('scripts/functional_path_credit.py')
    if cfg.get('independent_consensus'):source_files.append('scripts/independent_functional_consensus.py')
    if cfg.get('source_path_export_only'):source_files.append('scripts/export_functional_source_paths.py')
    w=MultisiteWork(cfg,args.config,source_files)
    error=None
    def checked(path):return w.checked(ROOT/Path(path))
    def log(stage,**kw):
        w.progress(stage,**kw)
        with (w.run/'events.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,written_at_utc=datetime.now(timezone.utc).isoformat(),**kw))+'\n')
    def budget():
        if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Declared grammar-transfer budget exceeded')
    try:
        import numpy as np,torch,transformers
        w.torch=torch;w.device=torch.device('cuda:0');torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        if cfg.get('training_run'):
            tr=ROOT/cfg['training_run'];tc=json.loads(checked(tr/'config.resolved.json').read_text());rc=dict(checkpoint_step=cfg['checkpoint_step'])
            assert json.loads(checked(tr/'status.json').read_text())['status']=='PASS'
        else:
            base=ROOT/cfg['group_run'];gc=json.loads(checked(base/'config.resolved.json').read_text());ref=ROOT/gc['reference_run'];rc=json.loads(checked(ref/'config.resolved.json').read_text());tr=ROOT/rc['training_run'];tc=json.loads(checked(tr/'config.resolved.json').read_text())
            assert json.loads(checked(base/'status.json').read_text())['status']=='PASS'
        if cfg.get('sae_loader')!='sparsify':
            sys.path.append(tc['dictionary_source_dir']);sys.path.append(tc['dictionary_overlay_dir'])
            from dictionary_learning.trainers.top_k import AutoEncoderTopK
            from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
            for f in ['dictionary_learning/trainers/top_k.py','dictionary_learning/trainers/matryoshka_batch_top_k.py','LICENSE']:checked(Path(tc['dictionary_source_dir'])/f)
        for f in ['config.json','model.safetensors','tokenizer.json']:checked(Path(tc['model_local_dir'])/f)
        tokenizer=transformers.AutoTokenizer.from_pretrained(tc['model_local_dir'],local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(tc['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device);model.requires_grad_(False);model.config.use_cache=False
        module=model.get_submodule(tc['hook_module_path']);saes={};D={};dim=int(model.config.hidden_size)
        class SparsifyTopK(torch.nn.Module):
            def __init__(self,state,k):
                super().__init__();self.k=k
                for n,key in [('ew','encoder.weight'),('eb','encoder.bias'),('db','b_dec'),('dw','W_dec')]:self.register_buffer(n,state[key])
            def encode(self,x):
                v=torch.nn.functional.linear(x-self.db,self.ew,self.eb).relu();values,indices=v.topk(self.k,sorted=False)
                return torch.zeros_like(v).scatter_(-1,indices,values)
        cohorts=[(tr,tc,cfg['seeds'],rc['checkpoint_step'])]
        if cfg.get('target_training_run'):
            tt=checked(Path(cfg['target_training_run'])/'config.resolved.json').parent
            ttc=json.loads((tt/'config.resolved.json').read_text())
            assert json.loads(checked(tt/'status.json').read_text())['status']=='PASS'
            for key in ['model_id','model_revision','hook_module_path','dict_size','k','steps','learning_rate','token_manifest','batch_size_sequences','warmup_steps','decay_start','group_fractions']:
                assert tc[key]==ttc[key],key
            assert set(cfg['target_seeds']).isdisjoint(cfg['seeds'])
            cohorts.append((tt,ttc,cfg['target_seeds'],cfg['target_checkpoint_step']))
        for cohort,ct,seeds,step in cohorts:
            snapshots=json.loads(checked(cohort/'checkpoints.json').read_text())['checkpoints']
            for obj in cfg['objectives']:
                for seed in seeds:
                    snap=next(s for s in snapshots if s.get('objective',obj)==obj and s['seed']==seed and s['step']==step)
                    if cfg.get('sae_loader')=='sparsify':
                        from safetensors.torch import load_file
                        sp=checked(Path(snap['path'])/'sae.safetensors');assert sha256(sp)==snap['sha256']
                        scfg=json.loads(checked(Path(snap['path'])/'cfg.json').read_text());assert scfg['activation']=='topk' and not scfg['skip_connection']
                        ae=SparsifyTopK(load_file(str(sp),device=str(w.device)),scfg['k']);ae.eval();ae.requires_grad_(False)
                        saes[obj,seed]=ae;D[obj,seed]=ae.dw;assert ae.dw.shape[1]==dim
                        continue
                    sp=checked(snap['path']);assert sha256(sp)==snap['sha256']
                    state=torch.load(sp,map_location=w.device,weights_only=True)
                    ae=AutoEncoderTopK(dim,ct['dict_size'],ct['k']) if obj=='topk' else MatryoshkaBatchTopKSAE(dim,ct['dict_size'],ct['k'],state['group_sizes'].cpu().tolist())
                    ae=ae.to(w.device);ae.load_state_dict(state);ae.eval();ae.requires_grad_(False);saes[obj,seed]=ae;D[obj,seed]=ae.decoder.weight.T if obj=='topk' else ae.W_dec
        sys.path.append(cfg['scipy_overlay'])
        from scipy.optimize import linear_sum_assignment
        from fit_component_correspondence import fit,project_rows
        import scipy
        sm=json.loads(checked(cfg['data_manifest']).read_text());panel=[]
        checked(cfg['official_implementation_manifest'])
        for task in dict.fromkeys(cfg['source_tasks']+cfg['new_tasks']):
            fi=next(x for x in sm['files'] if x['task']==task);fp=checked(fi['path']);assert sha256(fp)==fi['sha256']
            for r in [json.loads(s) for s in fp.read_text().splitlines()]:
                rid=int(r['pairID']);split=None
                ranges=cfg.get('consumer_ranges') or {name:cfg[name+'_range'] for name in (['source_selection','fit','development'] if task in cfg['source_tasks'] else ['new'])}
                for name,(lo,hi) in ranges.items():
                    if lo<=rid<hi:split=name
                if split is None:continue
                good=[tokenizer.eos_token_id]+tokenizer.encode(r['sentence_good'])+[tokenizer.eos_token_id]
                bad=[tokenizer.eos_token_id]+tokenizer.encode(r['sentence_bad'])+[tokenizer.eos_token_id]
                prefix=[tokenizer.eos_token_id]+tokenizer.encode(r['one_prefix_prefix'])
                assert r['one_prefix_method'] and good[:len(prefix)]==prefix and bad[:len(prefix)]==prefix
                assert len(prefix)<min(len(good),len(bad)) and max(len(good),len(bad))<=cfg['max_length']
                panel.append(dict(task=task,row_id=rid,split=split,good=good,bad=bad,position=len(prefix)-1,
                                  sentence_good=r['sentence_good'],sentence_bad=r['sentence_bad']))
        write(w.run/'panel.json',dict(rows=panel,scope=cfg['scope']))
        manifest=json.loads((w.run/'manifest.json').read_text());manifest.update(schema_version='contrast.components.v1',mean_constants_source_split='Absolute native code deletions preserve original residual and decoder bias; no empirical centering',statistics_unit='Shared cyclic SAE seeds and original lexical pairs within fixed paradigms');write(w.run/'manifest.json',manifest)
        if cfg.get('independent_consensus'):
            manifest['statistics_unit']=cfg.get('consumer_statistics_unit','Independent target initializations conditional on the fixed five-source bank; generated sentence draws within three fixed grammars, shared across all targets and methods')
            write(w.run/'manifest.json',manifest)
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),threads=2,model=tc['model_id'],hook=tc['hook_module_path'])
        max_hidden=0.;phases={}
        def forward(records,delta=None,gradient=False,phase='clean'):
            nonlocal max_hidden
            batch=cfg['gradient_batch_pairs'] if gradient else cfg['batch_pairs'];actual=len(records)
            padded=records+[records[0]]*(batch-actual)
            ids=torch.full((2*batch,cfg['max_length']),tokenizer.eos_token_id,device=w.device,dtype=torch.long);length=[];positions=[]
            for i,r in enumerate(padded):
                for j,key in enumerate(['good','bad']):
                    ids[2*i+j,:len(r[key])]=torch.tensor(r[key],device=w.device);length.append(len(r[key]));positions.append(r['position'])
            idx=torch.arange(2*batch,device=w.device);pos=torch.tensor(positions,device=w.device);cache={}
            if delta is not None:
                delta=torch.cat([delta,torch.zeros((batch-actual,dim),device=w.device)]).repeat_interleave(2,0)
            def hook(m,i,out):
                h=out[0] if isinstance(out,tuple) else out;cache['h']=h[idx,pos].detach()
                if delta is None and not gradient:return out
                hh=h.clone()
                if delta is not None:hh[idx,pos]-=delta
                if gradient:hh=hh.detach().requires_grad_(True);cache['leaf']=hh
                return (hh,)+out[1:] if isinstance(out,tuple) else hh
            handle=module.register_forward_hook(hook)
            try:
                with torch.set_grad_enabled(gradient):
                    logits=model(ids,use_cache=False).logits
                    per=logits[:,:-1].log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
                    mask=torch.arange(cfg['max_length']-1,device=w.device)[None,:]<(torch.tensor(length,device=w.device)-1)[:,None]
                    total=(per*mask).double().sum(1).reshape(-1,2);margin=total[:,0]-total[:,1]
                    lp=logits[idx,pos].detach().double().log_softmax(-1).reshape(-1,2,logits.shape[-1])[:,0]
                    grad=None
                    if gradient:grad=torch.autograd.grad(margin.sum(),cache['leaf'])[0][idx,pos].reshape(batch,2,dim).sum(1).detach()
            finally:handle.remove()
            h=cache['h'].reshape(batch,2,dim);err=float((h[:,0]-h[:,1]).abs().max());max_hidden=max(max_hidden,err);assert err<cfg['hidden_atol'],err
            w.sequence_forwards+=len(ids);w.token_forwards+=ids.numel();phases[phase]=phases.get(phase,0)+len(ids)
            return margin[:actual].detach(),h[:actual,0],lp[:actual],None if grad is None else grad[:actual]
        def capture(split,gradient=False):
            records=[r for r in panel if r['split']==split];hs=[];gs=[];margins=[];batch=cfg['gradient_batch_pairs'] if gradient else cfg['batch_pairs']
            for off in range(0,len(records),batch):
                m,h,_,g=forward(records[off:off+batch],gradient=gradient,phase='capture_'+split);hs.append(h);margins.append(m)
                if g is not None:gs.append(g)
            h=torch.cat(hs);codes={}
            with torch.no_grad():
                for key,ae in saes.items():codes[key]=torch.cat([ae.encode(h[i:i+256]) for i in range(0,len(h),256)])
            return dict(rows=records,hidden=h,codes=codes,clean=torch.cat(margins),gradient=torch.cat(gs) if gs else None)
        if cfg.get('independent_consensus'):
            from independent_functional_consensus import run_consumer
            run_consumer(cfg,w,D,saes,capture,forward,checked,write,log,budget)
            w.checks['prefix_hidden_equality_and_replay']=max_hidden<cfg['hidden_atol']
            w.environment.update(maximum_prefix_hidden_error=max_hidden,forwards_by_phase=phases)
            return w.finish()
        if cfg.get('reuse_consumer_parent'):
            from functional_reuse_consumer import run_consumer
            run_consumer(cfg,w,D,saes,capture,forward,checked,write,log,budget)
            w.checks['prefix_hidden_equality_and_replay']=max_hidden<cfg['hidden_atol']
            w.environment.update(maximum_prefix_hidden_error=max_hidden,forwards_by_phase=phases)
            return w.finish()
        if cfg.get('raw_control_parent'):
            from component_raw_controls import run_controls
            run_controls(cfg,w,D,capture,forward,checked,write,log,budget)
            w.checks['prefix_hidden_equality_and_replay']=max_hidden<cfg['hidden_atol']
            w.environment.update(maximum_prefix_hidden_error=max_hidden,forwards_by_phase=phases)
            return w.finish()
        select=capture('source_selection',True);sources={};source_meta=[]
        # Check the shared-prefix derivative against actual symmetric edits
        # using the identical two-pair forward shape used for attribution.
        check_rows=select['rows'][:cfg['gradient_batch_pairs']]
        direction=select['gradient'][:len(check_rows)]
        direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-12)
        epsilon=.01
        plus,_,_,_=forward(check_rows,-epsilon*direction,gradient=True,phase='gradient_check')
        minus,_,_,_=forward(check_rows,epsilon*direction,gradient=True,phase='gradient_check')
        finite=(plus-minus)/(2*epsilon);analytic=(select['gradient'][:len(check_rows)]*direction).sum(1)
        check_error=float((finite-analytic).abs().max())
        write(w.run/'gradient_check.json',dict(epsilon=epsilon,analytic=analytic.cpu().tolist(),finite_difference=finite.cpu().tolist(),maximum_absolute_error=check_error,scope='Two source-selection pairs, identical two-pair batches, one normalized shared-prefix direction each.'))
        assert check_error<.02,check_error
        for obj in cfg['objectives']:
            for seed in cfg['seeds']:
                credit=select['codes'][obj,seed]*(select['gradient']@D[obj,seed].T)
                means=torch.stack([credit[[i for i,r in enumerate(select['rows']) if r['task']==task]].mean(0) for task in cfg['source_tasks']],1)
                score=means-(means.clamp_min(0).sum(1,keepdim=True)-means.clamp_min(0))/2
                values,winner=score.max(1);gate=torch.zeros_like(means);selected=[]
                for k in range(3):
                    eligible=torch.where((winner==k)&(values>0))[0];order=torch.argsort(score[eligible,k],descending=True,stable=True);ix=eligible[order[:cfg['source_members_per_component']]];gate[ix,k]=1
                    selected.append(dict(task=cfg['source_tasks'][k],members=ix.cpu().tolist(),selection_score=score[ix,k].cpu().tolist(),own_contribution=means[ix,k].cpu().tolist(),eligible=len(eligible)))
                sources[obj,seed]=gate;source_meta.append(dict(objective=obj,seed=seed,components=selected))
                np.savez_compressed(w.run/f'{obj}_s{seed}_source.npz',gate=gate.cpu().numpy(),mean_credit=means.cpu().numpy(),selectivity_score=score.cpu().numpy())
        np.savez_compressed(w.run/'source_gradients.npz',hidden=select['hidden'].cpu().numpy(),gradient=select['gradient'].cpu().numpy(),clean_margin=select['clean'].cpu().numpy())
        write(w.run/'source_selection.json',dict(sources=source_meta,rule=cfg['scope']))
        write(w.run/'SOURCE_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha256(w.run/'source_selection.json'),target_fit_or_intervention_outcomes_consumed=0,held_model_outcomes_consumed=0))
        log('SOURCE_COMPONENTS_FROZEN',components=len(cfg['objectives'])*len(cfg['seeds'])*len(cfg['source_tasks']))
        if cfg.get('source_path_export_only'):
            from export_functional_source_paths import export_paths
            export_paths(cfg,w,D,sources,select,forward,write,log,budget)
            w.checks['prefix_hidden_equality_and_replay']=max_hidden<cfg['hidden_atol']
            w.environment.update(maximum_prefix_hidden_error=max_hidden,forwards_by_phase=phases)
            return w.finish()
        del select
        fitdata=capture('fit');maps={};fit_meta=[]
        # A split/merge allocation verifies the actual projected solver.
        kt=torch.eye(3,device=w.device,dtype=torch.float64);bt=torch.tensor([[.8,.2],[.1,.7],[.3,.2]],device=w.device,dtype=torch.float64)
        xt,inf=fit(kt,bt,steps=300,ridge_fraction=0)
        assert float((xt-bt).abs().max())<1e-8 and float(xt.sum(1).max())<=1+1e-10
        write(w.run/'solver_check.json',dict(max_error=float((xt-bt).abs().max()),info=inf))
        for obj in cfg['objectives']:
            for s in cfg['seeds']:
                t=cfg['seeds'][(cfg['seeds'].index(s)+1)%len(cfg['seeds'])];sg=sources[obj,s];sp=torch.where(sg.sum(1)>0)[0]
                zs=fitdata['codes'][obj,s].double();zt=fitdata['codes'][obj,t].double();ds=D[obj,s].double();dt=D[obj,t].double();n=len(zs)
                se=zs[:,sp].square().mean(0)*ds[sp].square().sum(1);te=zt.square().mean(0)*dt.square().sum(1)
                cross=(zs[:,sp].T@zt/n)*(ds[sp]@dt.T);aff=cross/se.sqrt().clamp_min(1e-9)[:,None]/te.sqrt().clamp_min(1e-9)[None,:]
                tp=torch.argsort(aff.abs().max(0).values,descending=True,stable=True)[:cfg['target_pool']]
                x=zt[:,tp];d=dt[tp];K=(x.T@x/n)*(d@d.T);B=cross[:,tp].T@sg[sp].double()
                native={};details={}
                for method,cap in [('partition64',True),('independent64_clipped',False)]:
                    full,info=fit(K,B,steps=cfg['fit_steps'],ridge_fraction=cfg['ridge_fraction'],capacity=cap)
                    allowed=torch.zeros_like(full,dtype=torch.bool)
                    for k in range(3):allowed[torch.argsort(full[:,k].square()*K.diag(),descending=True,stable=True)[:cfg['target_members_per_component']],k]=True
                    native[method],details[method]=fit(K,B,steps=cfg['fit_steps'],ridge_fraction=cfg['ridge_fraction'],allowed=allowed,capacity=cap)
                    details[method]['full_pool_fit']=info
                si,ti=linear_sum_assignment(-aff[:,tp].cpu().numpy());allowed=torch.zeros_like(B,dtype=torch.bool)
                for a,b in zip(si,ti):allowed[b,int(sg[sp[a]].argmax())]=True
                native['assignment64'],details['assignment64']=fit(K,B,steps=cfg['fit_steps'],ridge_fraction=cfg['ridge_fraction'],allowed=allowed,capacity=True)
                y=torch.stack([(zs*sg[:,k].double())@ds for k in range(3)],1)
                A=x.T@x/n;A+=cfg['ridge_fraction']*A.diag().mean().clamp_min(1e-12)*torch.eye(len(tp),device=w.device,dtype=torch.float64)
                raw=torch.linalg.solve(A,x.T@y.reshape(n,-1)/n).reshape(len(tp),3,dim);rootA=torch.linalg.cholesky(A).T;low=[]
                for k in range(3):
                    _,_,vh=torch.linalg.svd(rootA@raw[:,k],full_matrices=False);low.append(raw[:,k]@vh[:2].T@vh[:2])
                rank2=torch.stack(low,1)
                key=f'{obj}_s{s}_t{t}';payload=dict(source_members=sp.cpu().numpy(),source_gate=sg[sp].cpu().numpy(),target_members=tp.cpu().numpy(),raw_full=raw.float().cpu().numpy(),raw_rank2=rank2.float().cpu().numpy(),**{m:g.float().cpu().numpy() for m,g in native.items()})
                np.savez_compressed(w.run/(key+'_map.npz'),**payload)
                maps[obj,s]=dict(target=t,tp=tp,native={m:g.float() for m,g in native.items()},raw=raw.float(),rank2=rank2.float())
                meta=dict(objective=obj,source=s,target=t,details=details,native_members={m:(g>1e-8).sum(0).cpu().tolist() for m,g in native.items()},source_members=sg.sum(0).cpu().tolist(),fit_rows=n,independent_union_rows_over_one=int((native['independent64_clipped'].sum(1)>1+1e-8).sum()))
                fit_meta.append(meta);log('COMPONENT_MAP_FITTED',**meta);budget()
        write(w.run/'fit_summary.json',dict(fits=fit_meta,source_only_selection=True,target_output_labels_consumed=False))
        frozen_files=list(w.run.glob('*_map.npz'))+list(w.run.glob('*_source.npz'))
        write(w.run/'MAP_FREEZE.json',dict(written_at_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha256(w.run/'config.resolved.json'),files=[dict(path=p.name,sha256=sha256(p)) for p in frozen_files],fit_summary_sha256=sha256(w.run/'fit_summary.json'),held_model_outcomes_consumed=0,combination_training_examples=0))
        log('ALL_MAPS_FROZEN',pairs=len(maps));del fitdata
        # Fixed singletons and joint component deletion; no outcome selects an operation.
        for split in ['development','new']:
            data=capture(split)
            for obj in cfg['objectives']:
                for s in cfg['seeds']:
                    mp=maps[obj,s];t=mp['target'];tp=mp['tp'];zs=data['codes'][obj,s];zt=data['codes'][obj,t][:,tp];ds=D[obj,s];dt=D[obj,t][tp];sg=sources[obj,s]
                    for op,weights in cfg['operations'].items():
                        c=torch.tensor(weights,device=w.device,dtype=zs.dtype);src=(zs*(sg@c))@ds
                        ds_by_method=dict(source=src,**{method:(zt*(g@c).clamp(0,1))@dt for method,g in mp['native'].items()},raw_full=zt@torch.einsum('pkd,k->pd',mp['raw'],c),raw_rank2=zt@torch.einsum('pkd,k->pd',mp['rank2'],c))
                        # Cyclic wrong component; for all-components the same allocation is intentionally identical.
                        wrong=c.roll(1);ds_by_method['wrong_partition']=(zt*(mp['native']['partition64']@wrong))@dt
                        for off in range(0,len(data['rows']),cfg['batch_pairs']):
                            rr=data['rows'][off:off+cfg['batch_pairs']];clean,h,clp,_=forward(rr,phase='eval_clean');replay=float((h-data['hidden'][off:off+len(rr)]).abs().max());assert replay<cfg['hidden_atol']
                            outputs={method:forward(rr,delta[off:off+len(rr)],phase='eval_'+method) for method,delta in ds_by_method.items()}
                            source_margin,_,slp,_=outputs['source'];effect=(slp.exp()*(slp-clp)).sum(1)
                            for method,(margin,_,lp,_) in outputs.items():
                                divergence=(slp.exp()*(slp-lp)).sum(1)
                                for j,r in enumerate(rr):
                                    w.record(kind='component_transfer',task=r['task'],row_id=r['row_id'],component=r['task']+':'+str(r['row_id']),mode=obj+'|'+op,operation=op,method=method,seed=s,target_seed=t,objective=obj,split=split,margin=float(margin[j]),clean_margin=float(clean[j]),source_margin=float(source_margin[j]),margin_error=abs(float(margin[j]-source_margin[j])),source_kl=float(divergence[j]),source_effect_kl=float(effect[j]),edit_norm=float(ds_by_method[method][off+j].norm()),source_changed=bool((source_margin[j]>0)!=(clean[j]>0)),source_decision_agreement=bool((margin[j]>0)==(source_margin[j]>0)))
                            budget()
                        log('OPERATION_EVALUATED',split=split,objective=obj,source=s,operation=op,rows=len(data['rows']))
            del data
        w.checks['prefix_hidden_equality_and_replay']=max_hidden<cfg['hidden_atol'];w.checks['source_before_target_and_all_maps_before_held']=True
        w.environment.update(maximum_prefix_hidden_error=max_hidden,forwards_by_phase=phases)
    except BaseException:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
