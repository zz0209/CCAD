"""Fit source-member counterparts on natural states, then reuse the human decision.

The correspondence fits individual decoder contributions and supports new unions.
The published SHIFT source procedure and annotations are MIT, copyright2024
saprmarks; original license/commit accompany the configured source manifest.
"""
from pathlib import Path
import argparse,json,sys,time,platform,traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork,write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module


def simplex_rows(x):
    x=np.maximum(x,0);need=x.sum(1)>1
    if np.any(need):
        a=x[need];s=np.sort(a,axis=1)[:,::-1];v=(np.cumsum(s,axis=1)-1)/np.arange(1,s.shape[1]+1);k=(s>v).sum(1)-1;x[need]=np.maximum(a-v[np.arange(len(a)),k,None],0)
    return x


def fit_members(k,b,iterations):
    lipschitz=max(np.linalg.eigvalsh(k)[-1],1e-15);a=np.zeros_like(b)
    for _ in range(iterations):a=simplex_rows(a+(b-k@a)/lipschitz)
    return a


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);a=p.parse_args();c=json.loads(a.config.read_text());w=MultisiteWork(c,a.config,['scripts/run_shift_transfer.py','scripts/run_shift_explanation.py','scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']);hooks=[];error=None
    try:
        import torch,transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(c['device']);torch.cuda.reset_peak_memory_stats();w.torch=torch;w.device=torch.device(c['device'])
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']]);from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.environment=dict(python=sys.executable,torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,matmul_precision='highest',cpu_threads=2)
        source=json.loads(w.checked(c['source_manifest'],'Published manual feature decisions','MIT').read_text());groups,annotations=source_groups(c['notebook'],source['members']);sites=list(source['members']);sb=np.load(w.checked(c['source_parameters'],'Published feature parameters','MIT'))
        sp={site:{k:torch.tensor(sb[site+'__'+k],device=w.device) for k in ['encoder','encoder_bias','decoder','center']} for site in sites};targets={}
        for site in sites:
            state=torch.load(w.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt','Independent natural-data target dictionary','MIT trainer'),map_location=w.device,weights_only=True)
            sae=AutoEncoderTopK(512,state['encoder.weight'].shape[0],int(state['k'])).to(w.device);sae.load_state_dict(state);sae.requires_grad_(False);targets[site]=sae
        for f in ['config.json','tokenizer.json','model.safetensors']:w.checked(Path(c['model_local_dir'])/f,'Pinned Pythia70M','Apache-2.0')
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device);model.requires_grad_(False);model.config.use_cache=False
        mode='capture';alpha=0.;query='full';observed={};responses={};relations={};mask=None;pooled=None
        functional_weight=c.get('functional_weight',0.)
        source_run=Path(c['frozen_source_run']);probe=np.load(w.checked(source_run/'probe.npz','Fixed original source classifier'));pw=torch.tensor(probe['weight'],device=w.device);pb=torch.tensor(probe['bias'],device=w.device)
        def hook(site):
            def f(module,inputs,out):
                nonlocal pooled
                x=out[0] if isinstance(out,tuple) else out
                if mode=='capture' and functional_weight and site=='embed':x=x.detach().requires_grad_(True)
                s=sp[site]
                if mode in ('capture','source'):
                    z=torch.relu((x-s['center'])@s['encoder'].T+s['encoder_bias'])
                q=torch.tensor([query=='full' or i in groups.get(query,{}).get(site,[]) for i in source['members'][site]],device=w.device,dtype=x.dtype)
                if mode=='capture':observed[site]=x.detach();x=x-alpha*(z@s['decoder']);responses[site]=x
                elif mode=='source':x=x-(z*q)@s['decoder']
                elif mode!='none':
                    t=targets[site];r=relations[site];zt=t.encode(x);coeff=r[mode]@q
                    if mode=='raw':x=x-(zt[:, :, r['candidates']]@r['raw']*q)@s['decoder']
                    else:x=x-(zt*coeff)@t.decoder.weight.T
                if site=='resid_4' and mode!='capture':pooled=(x*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
                return (x,*out[1:]) if isinstance(out,tuple) else x
            return f
        for site in sites:hooks.append(site_module(model,site).register_forward_hook(hook(site)))
        # The source deletion path supplies input states only; no task labels/logits.
        pth=w.checked(c['paired_tokens'],'Existing discovery natural split','ODC-By-1.0');nat=np.memmap(pth,dtype='<u2',mode='r').reshape(-1,128)[:c['fit_sequences']]
        cache={site:[] for site in sites};gradient_cache={site:[] for site in sites}
        for alpha in [0.,1.]:
            for off in range(0,len(nat),c['fit_batch_sequences']):
                ids=torch.tensor(np.array(nat[off:off+c['fit_batch_sequences']],dtype='int64'),device=w.device)
                with torch.set_grad_enabled(bool(functional_weight)):
                    model.gpt_neox(ids,use_cache=False)
                    if functional_weight:
                        score=(responses['resid_4'].mean(1)@pw.T+pb).sum()*ids.shape[1]
                        gradients=torch.autograd.grad(score,[responses[site] for site in sites])
                        for site,g in zip(sites,gradients):gradient_cache[site].append(g.detach().flatten(0,1).cpu().numpy().copy())
                for site in sites:cache[site].append(observed[site].flatten(0,1).cpu().numpy().copy())
                w.sequence_forwards+=len(ids);w.token_forwards+=ids.numel()
        fit_summary={};export={}
        for site in sites:
            x=torch.tensor(np.concatenate(cache.pop(site)),device=w.device);s=sp[site];t=targets[site];ns=len(source['members'][site]);dt=t.decoder.weight.T.detach();ds=s['decoder'];cos=(dt/dt.norm(dim=1)[:,None])@(ds/ds.norm(dim=1)[:,None]).T
            baseidx=torch.topk(cos,c['neighbors_per_source'],dim=0).indices.flatten().unique();nt=len(dt)
            zz=[];ss=[]
            with torch.no_grad():
                for off in range(0,len(x),512):
                    xx=x[off:off+512];zz.append(t.encode(xx)[:,baseidx].cpu().numpy());ss.append(torch.relu((xx-s['center'])@s['encoder'].T+s['encoder_bias']).cpu().numpy())
            zt=np.concatenate(zz).astype('float64');zs=np.concatenate(ss).astype('float64');di=dt[baseidx].cpu().numpy().astype('float64');dsrc=ds.cpu().numpy().astype('float64');n=len(zt)
            cz=zt.T@zt/n;cross=zt.T@zs/n;ke=cz*(di@di.T);be=cross*(di@dsrc.T);k,b=ke,be;weighted_source=None
            if functional_weight:
                gs=np.concatenate(gradient_cache.pop(site)).astype('float64');at=zt*(gs@di.T);ass=zs*(gs@dsrc.T)
                if c.get('context_response'):
                    at=at.reshape(-1,128,at.shape[1]).sum(1);ass=ass.reshape(-1,128,ass.shape[1]).sum(1)
                kf=at.T@at/len(at);bf=at.T@ass/len(at)
                se=max(np.trace(ke)/len(ke),1e-15);sf=max(np.trace(kf)/len(kf),1e-15)
                k=(1-functional_weight)*ke/se+functional_weight*kf/sf;b=(1-functional_weight)*be/se+functional_weight*bf/sf
                weighted_source=gs@dsrc.T
            candidate_fit=fit_members(k,b,c['fit_iterations'])
            allowance=min(len(baseidx),c['members_per_source']*ns);score=candidate_fit.sum(1)*np.sqrt(np.maximum(np.diag(k),0));chosen=np.argsort(-score,kind='stable')[:allowance];refit=fit_members(k[np.ix_(chosen,chosen)],b[chosen],c['fit_iterations'])
            native=np.zeros((nt,ns));native[baseidx.cpu().numpy()[chosen]]=refit
            # Same member allowance, dictionary-only matching, with source-fit gains.
            geom=np.zeros((nt,ns));used=set()
            for j in range(ns):
                for idx in baseidx[torch.argsort(cos[baseidx,j],descending=True)].cpu().tolist():
                    if idx not in used:geom[idx,j]=1.;used.add(idx)
                    if int(geom[:,j].sum())==c['members_per_source']:break
            geom_gain=geom.copy()
            # Geometric selections are in the retained candidate pool; no labels.
            lookup={int(v):i for i,v in enumerate(baseidx.cpu().tolist())}
            for j in range(ns):
                selected=np.flatnonzero(geom[:,j]);assert all(int(v) in lookup for v in selected)
                ix=[lookup[int(v)] for v in selected];den=k[np.ix_(ix,ix)].sum();gain=np.clip(b[ix,j].sum()/max(den,1e-15),0,1);geom_gain[selected,j]*=gain
            # A target-code readout retains every source member column and uses the whole candidate pool.
            ridge=c['ridge']*max(np.trace(cz)/len(cz),1e-12);raw=np.linalg.solve(cz+ridge*np.eye(len(cz)),cross)
            if functional_weight:
                for j in range(ns):
                    weights=weighted_source[:,j]**2;weights=weights/max(weights.mean(),1e-15);weights=(1-functional_weight)+functional_weight*weights
                    weighted_cov=zt.T@(zt*weights[:,None])/n;weighted_cross=zt.T@(zs[:,j]*weights)/n
                    if c.get('context_response'):
                        ax=(zt*weighted_source[:,j,None]).reshape(-1,128,zt.shape[1]).sum(1);cx=(zs[:,j]*weighted_source[:,j]).reshape(-1,128).sum(1)
                        functional_cov=ax.T@ax/len(ax);functional_cross=ax.T@cx/len(ax);sc=max(np.trace(cz)/len(cz),1e-15);sf=max(np.trace(functional_cov)/len(cz),1e-15)
                        weighted_cov=(1-functional_weight)*cz/sc+functional_weight*functional_cov/sf;weighted_cross=(1-functional_weight)*cross[:,j]/sc+functional_weight*functional_cross/sf
                    raw[:,j]=np.linalg.solve(weighted_cov+c['ridge']*max(np.trace(weighted_cov)/len(cz),1e-12)*np.eye(len(cz)),weighted_cross)
            relations[site]={'native':torch.tensor(native,device=w.device,dtype=torch.float32),'geometry':torch.tensor(geom,device=w.device,dtype=torch.float32),'geometry_gain':torch.tensor(geom_gain,device=w.device,dtype=torch.float32),'raw':torch.tensor(raw,device=w.device,dtype=torch.float32),'candidates':baseidx}
            target_mean=zt.mean(0);source_mean=zs.mean(0)
            for name,value in [('native',native),('geometry',geom),('geometry_gain',geom_gain),('raw',raw),('candidates',baseidx.cpu().numpy()),('target_mean',target_mean),('source_mean',source_mean)]:export[site+'__'+name]=value
            full=np.zeros_like(candidate_fit);full[chosen]=refit;source_energy=np.sum(zs**2,axis=0)/n*np.sum(dsrc**2,axis=1);loss=source_energy-2*(full*be).sum(0)+(full*(ke@full)).sum(0)
            fit_summary[site]=dict(source_members=ns,candidates=len(baseidx),selected=allowance,n_states=n,functional_weight=functional_weight,context_response=c.get('context_response',False),field_relative_error=(loss/np.maximum(source_energy,1e-15)).tolist(),row_capacity=float(native.sum(1).max()),natural_source_active=(zs>0).sum(0).tolist(),solver_iterations=c['fit_iterations'],target_mean_norm=float(np.linalg.norm(target_mean)),source_mean_norm=float(np.linalg.norm(source_mean)))
            if functional_weight:
                energy=(ass**2).mean(0);floss=energy-2*(full*bf).sum(0)+(full*(kf@full)).sum(0);fit_summary[site]['response_relative_error']=(floss/np.maximum(energy,1e-15)).tolist()
            w.progress('RELATION_FIT',site=site,result=fit_summary[site]);del x,zt,zs
        np.savez_compressed(w.run/'relation.npz',**export);write(w.run/'RELATION_FIT.json',fit_summary)
        panel=json.loads(w.checked(source_run/'panel.json','Frozen source-consumer development panel').read_text());dev=[r for r in panel['rows'] if r['split']=='dev']
        results={};pad=0
        for mode in c['methods']:
            for query in c['queries']:
                logits=np.empty(len(dev),np.float32);order=sorted(range(len(dev)),key=lambda i:len(dev[i]['tokens']))
                for start in range(0,len(dev),c['eval_batch_size']):
                    ix=order[start:start+c['eval_batch_size']];length=max(len(dev[i]['tokens']) for i in ix);ids=torch.full((len(ix),length),pad,device=w.device,dtype=torch.long);mask=torch.zeros_like(ids)
                    for j,i in enumerate(ix):v=dev[i]['tokens'];ids[j,:len(v)]=torch.tensor(v,device=w.device);mask[j,:len(v)]=1
                    with torch.no_grad():model.gpt_neox(ids,attention_mask=mask,use_cache=False);values=(pooled@pw.T+pb).squeeze(-1)
                    logits[ix]=values.cpu().numpy();w.sequence_forwards+=len(ix);w.token_forwards+=ids.numel()
                for r,value in zip(dev,logits):w.record(kind='classification',task='profession',row_id=r['row_id'],component=r['document_sha256'],method=mode,operation=query,seed=1,target_seed=c['target_seed'],split='dev',label=r['label'],gender=r['gender'],prediction=int(value>0),logit=float(value))
                acc={f'{y}/{g}':float(np.mean([(v>0)==r['label'] for r,v in zip(dev,logits) if r['label']==y and r['gender']==g])) for y in [0,1] for g in [0,1]};res=dict(profession=float(np.mean([(v>0)==r['label'] for r,v in zip(dev,logits)])),gender=float(np.mean([(v>0)==r['gender'] for r,v in zip(dev,logits)])),worst_group=min(acc.values()),groups=acc)
                results[f'{mode}/{query}']=res;w.progress('RESULT',method=mode,query=query,result=res)
                if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Bounded correspondence consumer')
        write(w.run/'TRANSFER_RESULTS.json',dict(results=results,scope=c['scope'],fit=fit_summary))
        w.checks.update(all_query_methods=len(results)==len(c['methods'])*len(c['queries']),native_capacity=all(v['row_capacity']<=1.000001 for v in fit_summary.values()))
    except Exception:error=traceback.format_exc()
    finally:
        for h in hooks:h.remove()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
