"""Measure localized entity--attribute replacements through existing SAE hooks.

This is an independent single-hook adaptation of binding factorization tests.
All remaining layers recompute; it does not implement the original paper's
multi-layer interventions with downstream context activations frozen.
"""
from __future__ import annotations
import argparse,json,os,platform,sys,time,traceback
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from run_causalgym_multisite import MultisiteWork,ROOT,write
from ccad.artifacts import sha256


def make_panel(cfg,tok):
    import numpy as np
    names=[x for x in cfg['names'] if len(tok.encode(' '+x,add_special_tokens=False))==1]
    places=[x for x in cfg['country_capitals'] if all(len(tok.encode(' '+v,add_special_tokens=False))==1 for v in x)]
    assert len(names)>=8 and len(places)>=8
    rng=np.random.default_rng(cfg['panel_seed']);rows=[];contexts=[];seen=set();rejected=[]
    excluded={r['prompt'] for name in cfg.get('exclude_evaluation_runs',[])
              for r in json.loads((ROOT/name/'panel.json').read_text())['rows']}
    for split,n in [('fit',cfg['fit_contexts']),('development',cfg['evaluation_contexts'])]:
        skip=cfg.get('evaluation_skip',0) if split=='development' else 0
        accepted=0;sample=0
        while accepted<n:
            while True:
                es=rng.choice(names,2,replace=False).tolist();ix=rng.choice(len(places),4,replace=False)
                pair=[places[int(i)] for i in ix];key=tuple(es+[x[0] for x in pair])
                if key not in seen:seen.add(key);break
            sample+=1
            if sample<=skip:continue
            candidate=[]
            for spec in cfg['templates']:
                for ps in [pair[:2],pair[2:]]:
                    context=''.join(spec['statement'].format(name=e,country=p[0]) for e,p in zip(es,ps))
                    candidate.extend(context+spec['question'].format(name=e) for e in es)
            if split=='development' and excluded.intersection(candidate):
                rejected.append(dict(names=es,places=pair,reason='exact prompt previously exposed'));continue
            accepted+=1
            cid=len(contexts);contexts.append(dict(context=cid,split=split,names=es,places=pair))
            for form in range(len(cfg['templates'])):
                spec=cfg['templates'][form]
                for donor in [False,True]:
                    ps=pair[2:] if donor else pair[:2]
                    text='';spans=[]
                    for e,(country,capital) in zip(es,ps):
                        sentence=spec['statement'].format(name=e,country=country)
                        start=len(text)+sentence.index(country);spans.append((start,start+len(country)))
                        text+=sentence
                    for query in [0,1]:
                        prompt=text+spec['question'].format(name=es[query])
                        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
                        positions=[]
                        for lo,hi in spans:
                            pp=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>lo and a<hi]
                            assert len(pp)==1,(country,pp)
                            positions.append(pp[0])
                        rows.append(dict(row_id=len(rows),component=cid,split=split,template=form,
                            donor=donor,query=query,names=es,countries=[p[0] for p in ps],
                            capitals=[p[1] for p in ps],all_capitals=[p[1] for p in pair],
                            prompt=prompt,tokens=enc['input_ids'],positions=positions,
                            answer=ps[query][1],answer_id=tok.encode(' '+ps[query][1],add_special_tokens=False)[0]))
    lookup={(r['component'],r['template'],r['donor'],r['query']):r['row_id'] for r in rows}
    for r in rows:r['paired_row']=lookup[r['component'],r['template'],not r['donor'],r['query']]
    return rows,contexts,dict(names=names,country_capitals=places,rejected_contexts=rejected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    cfg=json.loads(args.config.read_text());sources=[
        'scripts/run_binding_components.py','scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']
    if cfg.get('binding_transfer'):sources+=['scripts/binding_correspondence.py','scripts/arithmetic_relation_transfer.py','scripts/fit_component_correspondence.py']
    if cfg.get('binding_transfer',{}).get('adaptive_execution'):sources+=['scripts/adaptive_native_execution.py']
    w=MultisiteWork(cfg,args.config,sources)
    error=None
    try:
        import torch,numpy as np,transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
        w.torch=torch;w.device=torch.device(cfg['device']);torch.cuda.set_device(w.device)
        torch.cuda.reset_peak_memory_stats(w.device)
        modeldir=Path(cfg['model_local_dir'])
        for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json']:w.checked(modeldir/name)
        tok=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,
            trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False);model.config.use_cache=False
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
            transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(w.device))
        rows,contexts,vocab=make_panel(cfg,tok);write(w.run/'panel.json',dict(rows=rows,contexts=contexts,vocabulary=vocab))
        for prior in cfg.get('exclude_evaluation_runs',[]):
            old=json.loads(w.checked(ROOT/prior/'panel.json').read_text())
            assert not {r['prompt'] for r in rows if r['split']=='development'} & {r['prompt'] for r in old['rows']}
        bs=cfg['batch_size'];length=max(len(r['tokens']) for r in rows)
        modules={layer:model.get_submodule(f'model.layers.{layer}') for layer in cfg['layers']}
        hidden={layer:torch.zeros(len(rows),2,model.config.hidden_size,device=w.device) for layer in modules}
        base=torch.zeros(len(rows),dtype=torch.long,device=w.device)
        def batches(ids):
            for offset in range(0,len(ids),bs):yield ids[offset:offset+bs]
        def forward(indices,layer=None,delta=None,capture=False,grad=False):
            ids=torch.full((len(indices),length),tok.eos_token_id,device=w.device,dtype=torch.long)
            mask=torch.zeros_like(ids);last=[]
            for j,i in enumerate(indices):
                x=rows[i]['tokens'];ids[j,:len(x)]=torch.tensor(x,device=w.device);mask[j,:len(x)]=1;last.append(len(x)-1)
            ix=torch.arange(len(indices),device=w.device);positions=torch.tensor([rows[i]['positions'] for i in indices],device=w.device)
            def hook(ll):
                def apply(m,a,out):
                    h=out[0] if isinstance(out,tuple) else out
                    if capture:hidden[ll][indices]=h[ix[:,None],positions].detach()
                    if ll==layer and delta is not None:
                        hh=h.clone();hh[ix[:,None],positions]+=delta
                        return (hh,)+out[1:] if isinstance(out,tuple) else hh
                    return out
                return apply
            handles=[m.register_forward_hook(hook(ll)) for ll,m in modules.items() if capture or ll==layer]
            try:
                with torch.set_grad_enabled(grad):
                    logits=model(input_ids=ids,attention_mask=mask).logits[ix,torch.tensor(last,device=w.device)].float()
                    probs=logits.log_softmax(-1)
                w.sequence_forwards+=len(indices);w.token_forwards+=len(indices)*length
                if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Binding development budget')
                return probs
            finally:
                for h in handles:h.remove()
        for ii in batches(list(range(len(rows)))):
            logits=forward(ii,capture=True);base[ii]=logits.argmax(-1)
            for j,i in enumerate(ii):
                r=rows[i];w.record(kind='base',task=f"template{r['template']}",row_id=i,component=r['component'],
                    mode='clean',method='model',seed=0,operation='original',split=r['split'],
                    query=r['query'],donor=r['donor'],answer_id=int(base[i]),answer=tok.decode([int(base[i])]),
                    expected=r['answer'],correct=bool(base[i]==r['answer_id']))
        np.savez_compressed(w.run/'states.npz',**{f'hidden_{ll}':h.cpu().numpy() for ll,h in hidden.items()},base=base.cpu().numpy())
        w.progress('STATES',rows=len(rows),length=length)
        fits=[r['row_id'] for r in rows if r['split']=='fit' and not r['donor']]
        evals=[r['row_id'] for r in rows if r['split']=='development' and not r['donor']]
        for layer in cfg['layers']:
            tr=ROOT/cfg['training_runs'][str(layer)]
            tc=json.loads(w.checked(tr/'config.resolved.json').read_text());assert tc['hook_module_path']==f'model.layers.{layer}'
            snaps=json.loads(w.checked(tr/'checkpoints.json').read_text())['checkpoints']
            sys.path.extend([tc['dictionary_source_dir'],tc['dictionary_overlay_dir']])
            from dictionary_learning.trainers.top_k import AutoEncoderTopK
            w.checked(Path(tc['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py')
            w.checked(Path(tc['dictionary_source_dir'])/'LICENSE')
            for seed in cfg['seeds']:
                snap=next(x for x in snaps if x['seed']==seed and x['step']==cfg['checkpoint_step'] and x['objective']=='topk')
                cp=w.checked(snap['path']);assert sha256(cp)==snap['sha256']
                sae=AutoEncoderTopK(model.config.hidden_size,tc['dict_size'],tc['k']).to(w.device)
                sae.load_state_dict(torch.load(cp,map_location=w.device,weights_only=True));sae.eval();sae.requires_grad_(False)
                with torch.no_grad():z=sae.encode(hidden[layer].flatten(0,1)).reshape(len(rows),2,-1)
                dzfit=z[[rows[i]['paired_row'] for i in fits]]-z[fits]
                energy=dzfit.square().mean((0,1))*sae.decoder.weight.square().sum(0)
                order=torch.argsort(energy,descending=True,stable=True)
                np.savez_compressed(w.run/f'source_members_l{layer}_s{seed}.npz',energy=energy.cpu().numpy(),order=order.cpu().numpy())
                learned={}
                conditional=None
                if cfg.get('semantic_queries'):
                    qc=cfg['semantic_queries'];countries=[p[0] for p in vocab['country_capitals']]
                    means=torch.zeros(len(countries),tc['dict_size'],device=w.device);counts=[]
                    for country in countries:
                        sites=[(r['row_id'],j) for r in rows if r['split']=='fit'
                               for j in range(2) if r['countries'][j]==country]
                        assert sites,country
                        means[countries.index(country)]=torch.stack([z[i,j] for i,j in sites]).mean(0)
                        counts.append(len(sites))
                    conditional=torch.zeros_like(z)
                    allowed=order[:qc['parent_members']]
                    norm=sae.decoder.weight.norm(dim=0)
                    for r in rows:
                        dr=rows[r['paired_row']]
                        for j in range(2):
                            a,b=[countries.index(x['countries'][j]) for x in [r,dr]]
                            relevance=(means[b]-means[a]).abs()*norm
                            pick=allowed[torch.argsort(relevance[allowed],descending=True,stable=True)[:qc['members']]]
                            conditional[r['row_id'],j,pick]=1
                    np.savez_compressed(w.run/f'source_semantic_queries_l{layer}_s{seed}.npz',
                        countries=np.array(countries),means=means.cpu().numpy(),counts=np.array(counts),
                        parent_indices=allowed.cpu().numpy(),masks=conditional.cpu().numpy())
                if cfg.get('source_fit'):
                    fc=cfg['source_fit'];rng=np.random.default_rng(fc['seed']+seed)
                    g=torch.zeros(tc['dict_size'],device=w.device,requires_grad=True)
                    score=torch.zeros_like(g)
                    def loss_batch():
                        ii=rng.choice(fits,size=fc['batch_size'],replace=False).tolist()
                        donor=[rows[i]['paired_row'] for i in ii]
                        sites=torch.tensor(rng.integers(0,2,len(ii)),device=w.device)
                        dz=(z[donor]-z[ii])*g
                        active=torch.arange(2,device=w.device)[None,:]==sites[:,None]
                        delta=(dz*active[:,:,None])@sae.decoder.weight.T
                        labels=[rows[d]['answer_id'] if rows[i]['query']==int(site) else rows[i]['answer_id']
                                for i,d,site in zip(ii,donor,sites)]
                        probs=forward(ii,layer,delta,grad=True)
                        return -probs[torch.arange(len(ii),device=w.device),torch.tensor(labels,device=w.device)].mean()
                    for _ in range(fc['score_steps']):
                        loss=loss_batch();loss.backward();score-=g.grad.detach()/fc['score_steps'];g.grad=None
                    selected=torch.argsort(score,descending=True,stable=True)[:fc['members']]
                    keep=torch.zeros_like(g);keep[selected]=1
                    with torch.no_grad():g[selected]=.5
                    optim=torch.optim.Adam([g],lr=fc['lr']);history=[]
                    for step in range(fc['steps']):
                        optim.zero_grad();loss=loss_batch();loss.backward();optim.step()
                        with torch.no_grad():g.clamp_(0,1);g.mul_(keep)
                        history.append(float(loss))
                        w.record(kind='fit',task='binding',row_id=step,component=seed,mode=f'layer{layer}',
                                 method='learned',seed=seed,operation='source_fit',split='fit',loss=float(loss))
                        if step%32==0:w.progress('SOURCE_FIT',layer=layer,seed=seed,step=step,loss=float(loss))
                    learned[f"learned{fc['members']}"]=g.detach()
                    np.savez_compressed(w.run/f'source_learned_l{layer}_s{seed}.npz',gate=g.detach().cpu().numpy(),
                                        selected=selected.cpu().numpy(),score=score.cpu().numpy(),loss=np.array(history))
                methods=[('raw',None)] if seed==cfg['seeds'][0] else []
                methods += [(f'sae{k}',k) for k in cfg['members']]
                methods += [(name,-1) for name in learned]
                if conditional is not None:methods.append((f"country{cfg['semantic_queries']['members']}",-2))
                transfers={};target_seed=None
                if cfg.get('binding_transfer'):
                    from binding_correspondence import build_transfers
                    target_seed=cfg['seed_pairs'][str(seed)]
                    ts=next(x for x in snaps if x['seed']==target_seed and x['step']==cfg['checkpoint_step'] and x['objective']=='topk')
                    cp=w.checked(ts['path']);assert sha256(cp)==ts['sha256']
                    target=AutoEncoderTopK(model.config.hidden_size,tc['dict_size'],tc['k']).to(w.device)
                    target.load_state_dict(torch.load(cp,map_location=w.device,weights_only=True));target.eval();target.requires_grad_(False)
                    with torch.no_grad():zt=target.encode(hidden[layer].flatten(0,1)).reshape(len(rows),2,-1)
                    transfers=build_transfers(w,cfg,rows,hidden[layer],z,sae,order,conditional,target,zt,seed,target_seed)
                    methods += [(name,-3) for name in transfers]
                    del target,zt
                if cfg.get('evaluate_methods'):methods=[(name,cap) for name,cap in methods if name in cfg['evaluate_methods']]
                for method,cap in methods:
                    for operation,sites in [('first',[0]),('second',[1]),('both',[0,1])]:
                        for ii in batches(evals):
                            donor=[rows[i]['paired_row'] for i in ii]
                            with torch.no_grad():
                                if cap==-3:delta=transfers[method][ii].clone()
                                elif cap is None:delta=hidden[layer][donor]-hidden[layer][ii]
                                else:
                                    dz=z[donor]-z[ii]
                                    if cap==-2:
                                        delta=(dz*conditional[ii])@sae.decoder.weight.T
                                    elif method in learned:
                                        delta=(dz*learned[method])@sae.decoder.weight.T
                                    elif cap<tc['dict_size']:
                                        keep=order[:cap];delta=dz[:,:,keep]@sae.decoder.weight[:,keep].T
                                    else:delta=dz@sae.decoder.weight.T
                                for j in range(2):
                                    if j not in sites:delta[:,j]=0
                            scores=forward(ii,layer,delta);pred=scores.argmax(-1)
                            for j,i in enumerate(ii):
                                r=rows[i];dr=rows[r['paired_row']];change=r['query'] in sites
                                expected=dr['answer_id'] if change else r['answer_id']
                                options=[tok.encode(' '+v,add_special_tokens=False)[0] for v in r['all_capitals']]
                                choice=options[int(scores[j,options].argmax())]
                                w.record(kind='intervention',task=f"template{r['template']}",row_id=i,component=r['component'],
                                    mode=f'layer{layer}',method=method,seed=0 if cap is None else seed,
                                    target_seed=target_seed if cap==-3 else None,operation=operation,
                                    split=r['split'],query=r['query'],requested=change,answer_id=int(pred[j]),answer=tok.decode([int(pred[j])]),
                                    expected_id=expected,expected=tok.decode([expected]),correct=bool(pred[j]==expected),
                                    four_choice_correct=bool(choice==expected),expected_log_probability=float(scores[j,expected]),
                                    clean_correct=bool(base[i]==r['answer_id']),edit_norm=float(delta[j].norm()))
                        w.progress('SOURCE_INTERVENTIONS',layer=layer,seed=seed,method=method,operation=operation)
                del sae,z
        w.checks['contexts_disjoint']=len({(tuple(c['names']),tuple(tuple(p) for p in c['places'])) for c in contexts})==len(contexts)
        w.checks['source_member_selection_fit_only']=all(rows[i]['split']=='fit' for i in fits)
    except Exception:error=traceback.format_exc();print(error,file=sys.stderr)
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
