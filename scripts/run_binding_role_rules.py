"""Test source/target binding-role operations and unfitted joint restoration.

Mean slot contrasts follow Feng and Steinhardt's binding-ID hypothesis.
This is a single-hook, recomputed-tail adaptation. Fixed signed code columns
map source roles to target members; every runtime edit retains the SAE residual.
"""
from __future__ import annotations
import argparse, json, os, platform, sys, time, traceback
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
from run_causalgym_multisite import MultisiteWork, ROOT, write
from ccad.artifacts import sha256


def make_panel(cfg, tok):
    import numpy as np
    names=[x for x in cfg['names'] if len(tok.encode(' '+x,add_special_tokens=False))==1]
    places=[x for x in cfg['country_capitals'] if all(len(tok.encode(' '+v,add_special_tokens=False))==1 for v in x)]
    rng=np.random.default_rng(cfg['panel_seed']); rows=[]; worlds=[]; seen=set()
    for split,count in [('fit',cfg['fit_contexts']),('development',cfg['evaluation_contexts'])]:
        skip=cfg.get('evaluation_skip',0) if split=='development' else 0
        for sample in range(count+skip):
            while True:
                es=rng.choice(names,2,replace=False).tolist(); ps=[places[int(j)] for j in rng.choice(len(places),2,replace=False)]
                identity=tuple(sorted(zip(es,[p[0] for p in ps])))
                if identity not in seen:seen.add(identity);break
            if sample<skip:continue
            cid=len(worlds);worlds.append(dict(component=cid,split=split,names=es,places=ps))
            for order in [0,1]:
                perm=[0,1] if order==0 else [1,0]
                for form,spec in enumerate(cfg.get('evaluation_templates',cfg['templates']) if split=='development' else cfg['templates']):
                    text='According to the record, ';spans=[]
                    for j in perm:
                        sentence=spec['statement'].format(name=es[j],country=ps[j][0])
                        e0=len(text)+sentence.index(es[j]);a0=len(text)+sentence.index(ps[j][0])
                        spans.append((e0,e0+len(es[j]),a0,a0+len(ps[j][0])));text+=sentence
                    for query in [0,1]:
                        prompt=text+spec['question'].format(name=es[query])
                        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);positions=[]
                        for elo,ehi,alo,ahi in spans:
                            ei=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>elo and a<ehi]
                            ai=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>alo and a<ahi]
                            assert len(ei)==len(ai)==1,(ei,ai)
                            positions.extend([ei[0],ei[0]+1,ai[0]])
                        assert len(set(positions))==6 and max(positions)<len(enc['input_ids'])-1
                        ans=[tok.encode(' '+p[1],add_special_tokens=False)[0] for p in ps]
                        rows.append(dict(row_id=len(rows),component=cid,split=split,order=order,template=form,
                            query=query,names=es,countries=[p[0] for p in ps],capitals=[p[1] for p in ps],
                            positions=positions,tokens=enc['input_ids'],prompt=prompt,answer_ids=ans,
                            answer_id=ans[query],swap_answer_id=ans[1-query]))
    return rows,worlds


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    cfg=json.loads(args.config.read_text());w=MultisiteWork(cfg,args.config,[
        'scripts/run_binding_role_rules.py','scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    error=None
    try:
        import torch,numpy as np,transformers
        from scipy.optimize import linear_sum_assignment
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        w.torch=torch;w.device=torch.device(cfg['device']);torch.cuda.set_device(w.device);torch.cuda.reset_peak_memory_stats(w.device)
        modeldir=Path(cfg['model_local_dir'])
        for name in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json']:w.checked(modeldir/name)
        tok=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False);model.config.use_cache=False
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
            transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(w.device),
            matmul_precision=torch.get_float32_matmul_precision())
        rows,worlds=make_panel(cfg,tok);write(w.run/'panel.json',dict(rows=rows,worlds=worlds,
            rule={'entity':'swap answers','attribute':'swap answers','both':'retain original answers'},
            sites=['entity0','entity0_next','attribute0','entity1','entity1_next','attribute1']))
        bs=cfg['batch_size'];length=max(len(r['tokens']) for r in rows);dim=model.config.hidden_size
        layer=cfg['layer'];module=model.get_submodule(f'model.layers.{layer}')
        hidden=torch.zeros(len(rows),6,dim,device=w.device);base=torch.zeros(len(rows),dtype=torch.long,device=w.device)
        calls=dict(baseline_forward=0,teacher_forward=0,training_forward=0,training_backward=0,evaluation_forward=0)
        def batches(ii):
            for j in range(0,len(ii),bs):yield ii[j:j+bs]
        def forward(ii,delta=None,capture=False,grad=False,category='evaluation_forward'):
            ids=torch.full((len(ii),length),tok.eos_token_id,device=w.device,dtype=torch.long);mask=torch.zeros_like(ids);last=[]
            for j,i in enumerate(ii):
                x=rows[i]['tokens'];ids[j,:len(x)]=torch.tensor(x,device=w.device);mask[j,:len(x)]=1;last.append(len(x)-1)
            ix=torch.arange(len(ii),device=w.device);pos=torch.tensor([rows[i]['positions'] for i in ii],device=w.device)
            def hook(m,a,out):
                h=out[0] if isinstance(out,tuple) else out
                if capture:hidden[ii]=h[ix[:,None],pos].detach()
                if delta is None:return out
                hh=h.clone();hh[ix[:,None],pos]+=delta
                return (hh,)+out[1:] if isinstance(out,tuple) else hh
            handle=module.register_forward_hook(hook)
            try:
                with torch.set_grad_enabled(grad):
                    logits=model(input_ids=ids,attention_mask=mask).logits[ix,torch.tensor(last,device=w.device)].float()
                    lp=logits.log_softmax(-1)
                w.sequence_forwards+=len(ii);w.token_forwards+=len(ii)*length;calls[category]+=1
                if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('R56 joint pilot budget')
                return lp
            finally:handle.remove()
        for ii in batches(list(range(len(rows)))):
            lp=forward(ii,capture=True,category='baseline_forward');base[ii]=lp.argmax(-1)
            for j,i in enumerate(ii):w.record(kind='baseline',row_id=i,component=rows[i]['component'],split=rows[i]['split'],
                method='baseline',operation='none',seed=0,answer_id=int(base[i]),expected_id=rows[i]['answer_id'],correct=bool(base[i]==rows[i]['answer_id']))
        # Query appears after every capture site; use one query copy for means and field fits.
        fits=[r['row_id'] for r in rows if r['split']=='fit' and r['query']==0]
        fitall=[r['row_id'] for r in rows if r['split']=='fit'];ev=[r['row_id'] for r in rows if r['split']=='development']
        means=(hidden[fits,3:]-hidden[fits,:3]).mean(0)
        tr=ROOT/cfg['training_run'];tc=json.loads(w.checked(tr/'config.resolved.json').read_text())
        assert tc['hook_module_path']==f'model.layers.{layer}'
        snaps=json.loads(w.checked(tr/'checkpoints.json').read_text())['checkpoints']
        sys.path.extend([tc['dictionary_source_dir'],tc['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        for name in ['dictionary_learning/trainers/top_k.py','LICENSE']:w.checked(Path(tc['dictionary_source_dir'])/name)
        def load(seed):
            snap=next(x for x in snaps if x['seed']==seed and x['step']==cfg['checkpoint_step'] and x['objective']=='topk')
            cp=w.checked(snap['path']);assert sha256(cp)==snap['sha256']
            sae=AutoEncoderTopK(dim,tc['dict_size'],tc['k']).to(w.device)
            sae.load_state_dict(torch.load(cp,map_location=w.device,weights_only=True));sae.eval();sae.requires_grad_(False)
            with torch.no_grad():z=sae.encode(hidden.flatten(0,1)).reshape(len(rows),6,-1)
            return sae,z
        source,zs=load(cfg['source_seed']);target,zt=load(cfg['target_seed'])
        ds=source.decoder.weight.T.detach();dt=target.decoder.weight.T.detach();m=len(ds);k=cfg['members']
        sm=(zs[fits,3:]-zs[fits,:3]).mean(0);tm=(zt[fits,3:]-zt[fits,:3]).mean(0)
        # One fixed source set per causal token role, chosen without target outcomes.
        si=torch.argsort(sm.abs()*ds.norm(dim=1),dim=-1,descending=True,stable=True)[:,:k]
        sc=torch.zeros_like(sm).scatter(1,si,sm.gather(1,si))
        ti_direct=torch.argsort(tm.abs()*dt.norm(dim=1),dim=-1,descending=True,stable=True)[:,:k]
        direct=torch.zeros_like(tm).scatter(1,ti_direct,tm.gather(1,ti_direct))
        roles=torch.tensor([0,1,2,0,1,2],device=w.device);sign=torch.tensor([1,1,1,-1,-1,-1],device=w.device).float()
        opmask={'entity':torch.tensor([1,1,0,1,1,0],device=w.device).float(),
                'attribute':torch.tensor([0,0,1,0,0,1],device=w.device).float(),
                'both':torch.ones(6,device=w.device)}
        def native(code,decoder,c,ii,op):
            proposed=c[roles]*sign[:,None]
            delta=torch.maximum(proposed[None],-code[ii])*opmask[op][None,:,None]
            return delta@decoder
        def raw(c,ii,op):return (c[roles]*sign[:,None]*opmask[op][:,None])[None].expand(len(ii),-1,-1)
        source_fields={op:native(zs,ds,sc,list(range(len(rows))),op).detach() for op in opmask}
        # Candidate selection for field transfer uses the source contribution direction.
        sv=sc@ds;cosine=(ds[si]/ds[si].norm(dim=-1,keepdim=True))@(dt/dt.norm(dim=1,keepdim=True)).T
        assignment=torch.zeros_like(tm)
        for r in range(3):
            aa,bb=linear_sum_assignment(-cosine[r].cpu().numpy())
            assert np.array_equal(aa,np.arange(k))
            bb=torch.tensor(bb,device=w.device)
            assignment[r,bb]=sc[r,si[r]]*ds[si[r]].norm(dim=1)/dt[bb].norm(dim=1)
        score=(sv@dt.T).abs()/dt.norm(dim=1)
        pool=torch.argsort(score,dim=-1,descending=True,stable=True)[:,:cfg['candidate_pool']]
        field_coeff=torch.zeros_like(tm)
        fitlog=[]
        # Fit the clipped realization on both signs and real fit codes.
        for r in range(3):
            pp=pool[r];dd=dt[pp];cc=torch.nn.Parameter(torch.zeros(len(pp),device=w.device))
            opt=torch.optim.Adam([cc],lr=cfg['field_lr']);ii=torch.tensor(fits,device=w.device)
            zz=zt[ii][:,[r,r+3]][:,:,pp];truth=source_fields['both'][ii][:,[r,r+3]]
            normal=truth.square().mean().clamp_min(1e-9)
            for step in range(cfg['field_steps']):
                opt.zero_grad();coef=cc[None,None,:]*torch.tensor([1.,-1.],device=w.device)[None,:,None]
                pred=torch.maximum(coef,-zz)@dd;loss=(pred-truth).square().mean()/normal
                loss.backward();opt.step()
            kk=torch.argsort(cc.detach().abs()*dd.norm(dim=1),descending=True,stable=True)[:k]
            field_coeff[r,pp[kk]]=cc.detach()[kk]
            fitlog.append(dict(role=r,field_pool_loss=float(loss.detach()),selected=k))
        support=field_coeff!=0
        # The same single-role teacher rows drive response and gain fitting. Both is never fitted.
        trainops=['entity','attribute'];teach={}
        for op in trainops:
            teach[op]=torch.empty(len(rows),model.config.vocab_size,device='cpu',dtype=torch.float32)
            for ii in batches(fitall):teach[op][ii]=forward(ii,source_fields[op][ii],category='teacher_forward').detach().cpu()
        params={'response':torch.nn.Parameter(field_coeff.clone()),'gain':torch.nn.Parameter(torch.ones(3,device=w.device)),
                'direct':torch.nn.Parameter(direct.clone()),'raw_response':torch.nn.Parameter(sv.clone())}
        supports={'response':support,'direct':direct!=0}
        opts={name:torch.optim.Adam([v],lr=cfg['response_lr']) for name,v in params.items()}
        rng=np.random.default_rng(cfg['fit_seed']);schedule=[];history=[]
        for step in range(cfg['response_steps']):
            ii=rng.choice(fitall,bs,replace=False).tolist();op=trainops[step%2];schedule.append(dict(step=step,rows=ii,operation=op))
            teacher=teach[op][ii].to(w.device)
            for name,param in params.items():
                opts[name].zero_grad()
                c=field_coeff*param[:,None] if name=='gain' else param
                delta=raw(c,ii,op) if name=='raw_response' else native(zt,dt,c,ii,op)
                lp=forward(ii,delta,grad=True,category='training_forward')
                if name=='direct':
                    labels=torch.tensor([rows[i]['swap_answer_id'] for i in ii],device=w.device)
                    loss=-lp[torch.arange(len(ii),device=w.device),labels].mean()
                else:loss=(teacher.exp()*(teacher-lp)).sum(-1).mean()
                loss.backward();calls['training_backward']+=1;opts[name].step()
                with torch.no_grad():
                    if name in supports:param.mul_(supports[name])
                    if name=='gain':param.clamp_(0,4)
                if step%16==0 or step==cfg['response_steps']-1:history.append(dict(step=step,method=name,loss=float(loss.detach())))
            if step%16==0:w.progress('ROLE_RESPONSE_FIT',step=step,calls=calls)
        write(w.run/'fit_schedule.json',schedule);write(w.run/'fit_history.json',dict(field=fitlog,response=history))
        methods={'raw_mean':('raw',means),'source_mean64':('source',sc),'source_mean_full':('source',sm),
                 'target_mean64':('target',direct),'assignment':('target',assignment),'field':('target',field_coeff),
                 'response':('target',params['response'].detach()),'gain':('target',field_coeff*params['gain'].detach()[:,None]),
                 'direct_fit':('target',params['direct'].detach()),'raw_response':('raw',params['raw_response'].detach())}
        diagnostics={}
        for name,(kind,c) in methods.items():
            diagnostics[name]=dict(coefficient_norm=float(c.norm()),stored_members=int((c!=0).any(0).sum()) if kind!='raw' else None,
                max_members_per_role=int((c!=0).sum(1).max()) if kind!='raw' else None)
            for op in opmask:
                for ii in batches(ev):
                    delta=raw(c,ii,op) if kind=='raw' else native(zs if kind=='source' else zt,ds if kind=='source' else dt,c,ii,op)
                    lp=forward(ii,delta);pred=lp.argmax(-1)
                    for j,i in enumerate(ii):
                        r=rows[i];expected=r['answer_id'] if op=='both' else r['swap_answer_id']
                        w.record(kind='intervention',row_id=i,component=r['component'],split=r['split'],template=r['template'],
                            order=r['order'],query=r['query'],method=name,operation=op,seed=cfg['source_seed'],target_seed=cfg['target_seed'],
                            answer_id=int(pred[j]),answer=tok.decode([int(pred[j])]),expected_id=expected,expected=tok.decode([expected]),
                            correct=bool(pred[j]==expected),expected_log_probability=float(lp[j,expected]),
                            original_log_probability=float(lp[j,r['answer_id']]),swap_log_probability=float(lp[j,r['swap_answer_id']]),
                            edit_norm=float(delta[j].norm()),clean_correct=bool(base[i]==r['answer_id']))
                w.progress('RULE_EVALUATION',method=name,operation=op,calls=calls)
        np.savez_compressed(w.run/'role_relations.npz',source_indices=si.cpu().numpy(),
            target_pool=pool.cpu().numpy(),**{name:c.detach().cpu().numpy() for name,(_,c) in methods.items()})
        np.savez_compressed(w.run/'role_states.npz',hidden=hidden.cpu().numpy(),source_codes=zs.cpu().numpy(),target_codes=zt.cpu().numpy())
        write(w.run/'role_diagnostics.json',dict(methods=diagnostics,calls=calls,fit_contexts=cfg['fit_contexts'],
            evaluation_contexts=cfg['evaluation_contexts'],single_fit_only=True,full_vocabulary_argmax=True,
            information='Direct fit receives symbolic swapped answer labels; response/gain/raw-response use source full-output teacher. All labels are available to all methods but objectives differ.',
            source_rule='Source-only counterbalanced slot code means, selected by decoder-contribution magnitude, no output fitting.',
            source_quality='Evaluate source mean64/full and raw mean before interpreting target fidelity.'))
        w.checks.update(context_identity_disjoint=len(worlds)==len({tuple(sorted(zip(r['names'],[p[0] for p in r['places']]))) for r in worlds}),
            joint_not_fitted=all(x['operation']!='both' for x in schedule),nonoverlapping_entity_attribute_sites=all(len(set(r['positions']))==6 for r in rows))
    except Exception:error=traceback.format_exc();print(error,file=sys.stderr)
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
