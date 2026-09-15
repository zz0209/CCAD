"""Binding mean interventions on the context's layer-stacked residual states.

Independent implementation of the causal-mediation protocol in Feng2024,
Section2/4.1. Clean context states are restored before every transformer
layer; the query stream remains live. No external implementation is imported.
"""
from __future__ import annotations
import argparse,json,os,platform,sys,time,traceback
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from run_binding_role_rules import make_panel
from run_causalgym_multisite import MultisiteWork,write,ROOT
from ccad.artifacts import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    c=json.loads(args.config.read_text());w=MultisiteWork(c,args.config,[
        'scripts/run_binding_context_roles.py','scripts/run_binding_role_rules.py',
        'scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']+
        (['scripts/binding_member_selection.py'] if c.get('member_selection') else []))
    error=None
    try:
        import torch,numpy as np,transformers
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
        w.torch=torch;w.device=torch.device(c['device']);torch.cuda.set_device(w.device);torch.cuda.reset_peak_memory_stats(w.device)
        root=Path(c['model_local_dir'])
        for n in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json']:w.checked(root/n)
        tok=transformers.AutoTokenizer.from_pretrained(root,local_files_only=True)
        model=transformers.AutoModelForCausalLM.from_pretrained(root,local_files_only=True,dtype=torch.float32,attn_implementation='eager').to(w.device).eval()
        model.requires_grad_(False);model.config.use_cache=False
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(w.device),matmul_precision='highest')
        rows,worlds=make_panel(c,tok);write(w.run/'panel.json',dict(rows=rows,worlds=worlds))
        for r in rows:
            boundary=r['prompt'].index('\nQuestion:')
            offsets=tok(r['prompt'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            r['context_end']=max(i for i,(_,b) in enumerate(offsets) if b<=boundary)+1
            assert max(r['positions'])<r['context_end']<len(r['tokens'])
        write(w.run/'panel.json',dict(rows=rows,worlds=worlds))
        depth=model.config.num_hidden_layers;dim=model.config.hidden_size;bs=c['batch_size'];length=max(len(r['tokens']) for r in rows)
        means=torch.zeros(depth,3,dim,device=w.device);fit_count=0;cache={};ii=[];positions=None;contextmask=None
        codecs={};codec_cost={'encode_batches':0,'nonzero_updates':{},'calls':{}}
        if c.get('sae_runtime'):
            for pre,run in c['sae_runtime']['training_runs'].items():
                pre=int(pre);tr=ROOT/run;tc=json.loads(w.checked(tr/'config.resolved.json').read_text())
                assert tc['hook_module_path']==f'model.layers.{pre-1}'
                snaps=json.loads(w.checked(tr/'checkpoints.json').read_text())['checkpoints']
                sys.path.extend([tc['dictionary_source_dir'],tc['dictionary_overlay_dir']])
                from dictionary_learning.trainers.top_k import AutoEncoderTopK
                w.checked(Path(tc['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py')
                w.checked(Path(tc['dictionary_source_dir'])/'LICENSE')
                ss=[]
                for seed in [c['source_seed'],c['target_seed']]:
                    snap=next(x for x in snaps if x['seed']==seed and x['step']==c['checkpoint_step'] and x['objective']=='topk')
                    cp=w.checked(snap['path']);assert sha256(cp)==snap['sha256']
                    sae=AutoEncoderTopK(dim,tc['dict_size'],tc['k']).to(w.device)
                    sae.load_state_dict(torch.load(cp,map_location=w.device,weights_only=True));sae.eval();sae.requires_grad_(False);ss.append(sae)
                ds,dt=[s.decoder.weight.T.detach() for s in ss]
                cosine=(ds/ds.norm(dim=1,keepdim=True))@(dt/dt.norm(dim=1,keepdim=True)).T
                match=cosine.argmax(1);ratio=ds.norm(dim=1)/dt[match].norm(dim=1)
                codecs[pre]=dict(s=ss[0],t=ss[1],ds=ds,dt=dt,match=match,ratio=ratio)
                np.savez_compressed(w.run/f'nearest_relation_pre{pre}.npz',match=match.cpu().numpy(),scale=ratio.cpu().numpy())
        roles=torch.tensor([0,1,2,0,1,2],device=w.device);sgn=torch.tensor([1,1,1,-1,-1,-1],device=w.device)
        ops={'entity':[1,1,0,1,1,0],'attribute':[0,0,1,0,0,1],'both':[1]*6}
        masks={k:torch.tensor(v,device=w.device) for k,v in ops.items()};active=None;capture=False;capture_mean=False
        hooks=[]
        def hook_for(l):
            def fn(module,args,kwargs):
                h=args[0] if args else kwargs['hidden_states']
                ix=torch.arange(len(ii),device=w.device)
                if capture:
                    cache[l]=h.detach().clone()
                    if capture_mean:
                        q=h[ix[:,None],positions];means[l].add_((q[:,3:]-q[:,:3]).sum(0))
                    return None
                if active is None:return None
                method,op=active
                hh=torch.where(contextmask[:,:,None],cache[l],h)
                use=l in method['layers']
                if use:
                    delta=means[l,roles]*sgn[:,None]*masks[op][:,None]
                    delta=delta[None].expand(len(ii),-1,-1)
                    if l in method.get('entity_fields',{}):
                        delta=delta.clone()
                        delta[:,[0,1,3,4]]=method['entity_fields'][l][:,[0,1,3,4]]
                    codec=method.get('codec')
                    if codec and l in codecs:
                        assets=codecs[l];s,t=assets['s'],assets['t'];ds,dt=assets['ds'],assets['dt']
                        clean=cache[l][ix[:,None],positions];zs=s.encode(clean.flatten(0,1)).reshape(len(ii),6,-1)
                        dz=s.encode((clean+delta).flatten(0,1)).reshape_as(zs)-zs
                        def sparse(v,d):
                            top=(v.abs()*d.norm(dim=1)).topk(c['sae_runtime']['members'],dim=-1).indices
                            return torch.zeros_like(v).scatter(-1,top,v.gather(-1,top))
                        if codec!='source_full':dz=sparse(dz,ds)
                        source_delta=dz@ds
                        if codec.startswith('source'):delta=source_delta;changes=dz
                        else:
                            zt=t.encode(clean.flatten(0,1)).reshape(len(ii),6,-1)
                            if codec=='nearest64':
                                changes=torch.zeros_like(zt).scatter_add(-1,assets['match'][None,None].expand_as(dz),dz*assets['ratio'])
                                changes=torch.maximum(changes,-zt)
                            else:
                                requested=delta if codec=='direct64' else source_delta
                                changes=t.encode((clean+requested).flatten(0,1)).reshape_as(zt)-zt
                                changes=sparse(changes,dt)
                            delta=changes@dt
                        key=f'{method["name"]}/pre{l}'
                        codec_cost['encode_batches']+=2 if codec.startswith('source') else 3 if codec=='nearest64' else 4
                        codec_cost['nonzero_updates'][key]=max(codec_cost['nonzero_updates'].get(key,0),int((changes!=0).sum(-1).max()))
                        codec_cost['calls'][key]=codec_cost['calls'].get(key,0)+1
                    hh[ix[:,None],positions]+=delta
                if args:return (hh,)+args[1:],kwargs
                return args,dict(kwargs,hidden_states=hh)
            return fn
        for l,module in enumerate(model.model.layers):hooks.append(module.register_forward_pre_hook(hook_for(l),with_kwargs=True))
        def batch(indices):
            nonlocal ii,positions,contextmask
            ii=indices;ids=torch.full((len(ii),length),tok.eos_token_id,device=w.device,dtype=torch.long);attn=torch.zeros_like(ids);last=[]
            for j,i in enumerate(ii):
                n=len(rows[i]['tokens']);ids[j,:n]=torch.tensor(rows[i]['tokens'],device=w.device);attn[j,:n]=1;last.append(n-1)
            positions=torch.tensor([rows[i]['positions'] for i in ii],device=w.device)
            contextmask=torch.arange(length,device=w.device)[None]<torch.tensor([rows[i]['context_end'] for i in ii],device=w.device)[:,None]
            return ids,attn,torch.tensor(last,device=w.device)
        def forward(b,gradient=False):
            with torch.set_grad_enabled(gradient):out=model(input_ids=b[0],attention_mask=b[1]).logits[torch.arange(len(ii),device=w.device),b[2]].float().log_softmax(-1)
            w.sequence_forwards+=len(ii);w.token_forwards+=len(ii)*length
            if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Context-role bounded adaptation budget')
            return out
        fits=[r['row_id'] for r in rows if r['split']=='fit' and r['query']==0]
        if c.get('frozen_means'):
            mp=w.checked(ROOT/c['frozen_means']);assert sha256(mp)==c['frozen_means_sha256']
            means.copy_(torch.tensor(np.load(mp)['means'],device=w.device));fit_count=len(fits)
        else:
            capture=True;capture_mean=True
            for j in range(0,len(fits),bs):
                ids=fits[j:j+bs];b=batch(ids);forward(b);fit_count+=len(ids)
            means.div_(fit_count)
        capture_mean=False
        write(w.run/'mean_protocol.json',dict(fit_rows=fit_count,total_layers=depth,sites=['entity','entity_next','attribute'],query_labels_used=False,
            context_protocol='Restore clean context residual states before every block, add mean contrasts at selected sites and layers, recompute query states. Pre14 and pre24 equal existing SAE post13 and post23.',
            layers=c['layer_sets']))
        np.savez_compressed(w.run/'context_role_means.npz',means=means.cpu().numpy())
        if c.get('member_selection'):
            from binding_member_selection import run_selection
            def control(method=None,op=None,capture_states=False):
                nonlocal capture,capture_mean,active
                capture=capture_states;capture_mean=False;active=(method,op) if method is not None else None
            run_selection(c,w,torch,model,rows,codecs,means,cache,batch,forward,control)
            for h in hooks:h.remove()
            return w.finish()
        ev=[r['row_id'] for r in rows if r['split']=='development'];noop_error=0.
        for j in range(0,len(ev),bs):
            capture=True;active=None;b=batch(ev[j:j+bs]);clean=forward(b);capture=False
            active=({'name':'noop','layers':[]},'both');noop=forward(b);noop_error=max(noop_error,float((noop-clean).abs().max()))
            for k,i in enumerate(ii):
                r=rows[i];w.record(kind='baseline',row_id=i,component=r['component'],split=r['split'],method='baseline',operation='none',seed=0,
                    answer_id=int(clean[k].argmax()),expected_id=r['answer_id'],correct=bool(clean[k].argmax()==r['answer_id']))
            for method in c['layer_sets']:
                for op in masks:
                    active=(method,op);lp=forward(b);pred=lp.argmax(-1)
                    norm=float((means[method['layers']][:,roles]*sgn[None,:,None]*masks[op][None,:,None]).norm())
                    for k,i in enumerate(ii):
                        r=rows[i];expected=r['answer_id'] if op=='both' else r['swap_answer_id']
                        w.record(kind='intervention',row_id=i,component=r['component'],split=r['split'],template=r['template'],order=r['order'],query=r['query'],
                            method=method['name'],operation=op,seed=c['source_seed'] if method.get('codec') else 0,target_seed=c['target_seed'] if method.get('codec') else 0,answer_id=int(pred[k]),answer=tok.decode([int(pred[k])]),
                            expected_id=expected,expected=tok.decode([expected]),correct=bool(pred[k]==expected),
                            expected_log_probability=float(lp[k,expected]),original_log_probability=float(lp[k,r['answer_id']]),
                            swap_log_probability=float(lp[k,r['swap_answer_id']]),edit_norm=norm,edit_norm_scope='Raw requested layer-stack norm, before SAE realization',clean_correct=bool(clean[k].argmax()==r['answer_id']))
            w.progress('CONTEXT_ROLE_EVALUATION',evaluated_rows=j+len(ii),total_rows=len(ev))
        for h in hooks:h.remove()
        write(w.run/'context_checks.json',dict(noop_max_logprob_error=noop_error))
        write(w.run/'codec_cost.json',codec_cost)
        w.checks.update(frozen_context_noop=noop_error<1e-5,mean_fit_has_no_development=all(rows[i]['split']=='fit' for i in fits),joint_not_fitted=True)
    except Exception:error=traceback.format_exc();print(error,file=sys.stderr)
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
