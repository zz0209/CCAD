"""Execute the published SHIFT member decision on the original Pythia/BiB assets.

The source procedure follows saprmarks/feature-circuits, MIT,
commit cf080789e16f50238097db1fb9a3947a6cfcf9f6, experiments/bib_shift.ipynb.
Copyright (c) 2024 saprmarks. The original license accompanies the local assets.
This implementation uses ordinary Torch hooks and the existing CCAD run writer.
"""
from pathlib import Path
import argparse, json, random, re, sys, time, traceback, platform
import numpy as np
from run_causalgym_multisite import MultisiteWork, write


def source_groups(notebook, members):
    text=''.join(json.loads(Path(notebook).read_text())['cells'][11]['source'])
    site_order=['embed']+[f'{kind}_{i}' for i in range(5) for kind in ['attn','mlp','resid']]
    groups={name:{} for name in ['pronouns','names','associated_words']};annotations={}
    site=None
    for line in text.splitlines():
        m=re.search(r'submodules\[(\d+)\]',line)
        if m:site=site_order[int(m.group(1))]
        m=re.match(r'\s*(\d+),\s*#\s*(.*)',line)
        if not m:continue
        idx=int(m.group(1));note=m.group(2)
        assert idx in members.get(site,[])
        group='names' if 'names' in note else ('pronouns' if ('pronoun' in note or re.search(r"['\"](?:he|she|his|her)['\"]",note,re.I)) else 'associated_words')
        groups[group].setdefault(site,[]).append(idx);annotations[f'{site}/{idx}']=note
    assert sum(len(v) for g in groups.values() for v in g.values())==55
    return groups,annotations


def select_rows(path, split, ambiguous, seed):
    import pyarrow.parquet as pq
    table=pq.read_table(path,filters=[('profession','in',[13,21])],columns=['hard_text','profession','gender'])
    buckets={(p,g):[] for p in [0,1] for g in [0,1]}
    import hashlib
    for row in table.to_pylist():
        y=int(row['profession']==13);g=row['gender'];text=row['hard_text']
        buckets[y,g].append(dict(text=text,label=y,gender=g,split=split,document_sha256=hashlib.sha256(text.encode()).hexdigest()))
    keys=[(0,0),(1,1)] if ambiguous else [(0,0),(0,1),(1,0),(1,1)]
    n=min(len(buckets[k]) for k in keys);rows=[x for k in keys for x in buckets[k][:n]]
    random.Random(seed).shuffle(rows)
    return rows,{str(k):len(v) for k,v in buckets.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text())
    w=MultisiteWork(c,a.config,['scripts/run_shift_explanation.py','scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    error=None;hooks=[]
    try:
        import torch,transformers,pyarrow
        torch.set_num_threads(c['cpu_threads']);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        w.torch=torch;w.device=torch.device(c['device']);torch.cuda.set_device(w.device);torch.cuda.reset_peak_memory_stats(w.device)
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,pyarrow=pyarrow.__version__,device=str(w.device),matmul_precision='highest',cpu_threads=c['cpu_threads'])
        source=json.loads(w.checked(c['source_manifest'],'Published human feature IDs, fixed original notebook','MIT').read_text());groups,annotations=source_groups(w.checked(c['notebook'],'Published source method and annotations','MIT'),source['members'])
        bank=np.load(w.checked(c['source_parameters'],'Extracted published encoder/decoder rows','MIT'))
        params={site:{key:torch.as_tensor(bank[f'{site}__{key}'],device=w.device) for key in ['encoder','encoder_bias','decoder','center']} for site in source['members']}
        train,tc=select_rows(w.checked(c['train_data'],'Pinned BiB train','MIT'),'train',True,c['probe_seed'])
        dev,dc=select_rows(w.checked(c['dev_data'],'Pinned BiB dev; official test untouched','MIT'),'dev',False,c['probe_seed'])
        train_hashes={r['document_sha256'] for r in train};duplicates=[r['document_sha256'] for r in dev if r['document_sha256'] in train_hashes]
        if duplicates:dev=[r for r in dev if r['document_sha256'] not in train_hashes]
        write(w.run/'data_selection.json',dict(train_counts=tc,dev_counts=dc,train_selected=len(train),dev_selected=len(dev),excluded_cross_split_duplicates=duplicates,test_opened=False,groups=groups,annotations=annotations))
        for i,r in enumerate(train+dev):r['row_id']=i
        modeldir=Path(c['model_local_dir'])
        for f in ['config.json','model.safetensors','tokenizer.json']:w.checked(modeldir/f,'Original Pythia-70m pinned model','Apache-2.0')
        tokenizer=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
        truncated={}
        for split,rows in [('train',train),('dev',dev)]:
            tokens=tokenizer([r['text'] for r in rows],add_special_tokens=True,truncation=False)['input_ids'];truncated[split]=sum(len(t)>c['max_length'] for t in tokens)
            for r,t in zip(rows,tokens):r['tokens']=t[:c['max_length']]
        write(w.run/'panel.json',dict(rows=train+dev,max_length=c['max_length'],truncated=truncated,selection='Original first-minimum balancing and Random(42) order; dev replaces original test for development.'))
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.config.use_cache=False
        for q in model.parameters():q.requires_grad_(False)
        method='none';capture=None;mask=None
        def hook(site):
            def apply(module,inputs,output):
                nonlocal capture
                h=output[0] if isinstance(output,tuple) else output
                ids=source['members'][site] if method=='full' else groups.get(method,{}).get(site,[])
                if ids:
                    ix=[source['members'][site].index(i) for i in ids];v=params[site];z=torch.relu((h-v['center'])@v['encoder'][ix].T+v['encoder_bias'][ix]);h=h-z@v['decoder'][ix]
                if site=='resid_4':capture=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
                return (h,*output[1:]) if isinstance(output,tuple) else h
            return apply
        for site in source['members']:
            module=model.gpt_neox.embed_in if site=='embed' else getattr(model.gpt_neox.layers[int(site.split('_')[1])],{'attn':'attention','mlp':'mlp'}.get(site.split('_')[0],'__unused__'),None)
            if site.startswith('resid'):module=model.gpt_neox.layers[int(site.split('_')[1])]
            hooks.append(module.register_forward_hook(hook(site)))
        def collect(rows,operation):
            nonlocal method,mask,capture
            method=operation;out=np.empty((len(rows),512),np.float32)
            # Ordering model inference by length avoids padding work; the probe sees original order.
            order=sorted(range(len(rows)),key=lambda i:len(rows[i]['tokens']));bs=c['inference_batch_size']
            for start in range(0,len(order),bs):
                ix=order[start:start+bs];n=max(len(rows[i]['tokens']) for i in ix)
                ids=torch.full((len(ix),n),tokenizer.pad_token_id,device=w.device,dtype=torch.long);mask=torch.zeros_like(ids)
                for k,i in enumerate(ix):t=rows[i]['tokens'];ids[k,:len(t)]=torch.tensor(t,device=w.device);mask[k,:len(t)]=1
                with torch.no_grad():model.gpt_neox(input_ids=ids,attention_mask=mask,use_cache=False)
                assert capture is not None;out[ix]=capture.cpu().numpy();w.sequence_forwards+=len(ix);w.token_forwards+=len(ix)*n
                if start%(bs*32)==0:w.progress('COLLECT',split=rows[0]['split'],method=operation,done=start+len(ix),total=len(rows))
                if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Source consumer initial budget')
            return out
        x=collect(train,'none');torch.manual_seed(c['probe_seed']);probe=torch.nn.Linear(512,1,device=w.device);opt=torch.optim.AdamW(probe.parameters(),lr=c['probe_lr']);losses=[]
        for start in range(0,len(train),c['probe_batch_size']):
            b=train[start:start+c['probe_batch_size']];xx=torch.as_tensor(x[start:start+len(b)],device=w.device);yy=torch.tensor([r['label'] for r in b],device=w.device,dtype=torch.float32)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(probe(xx).squeeze(-1),yy);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
        np.savez_compressed(w.run/'probe.npz',weight=probe.weight.detach().cpu().numpy(),bias=probe.bias.detach().cpu().numpy(),losses=losses)
        np.savez_compressed(w.run/'train_pooled.npz',hidden=x);results={}
        for method_name in c['methods']:
            z=collect(dev,method_name)
            with torch.no_grad():logits=probe(torch.as_tensor(z,device=w.device)).squeeze(-1).cpu().numpy()
            np.savez_compressed(w.run/f'dev_{method_name}.npz',hidden=z,logits=logits)
            for r,logit in zip(dev,logits):w.record(kind='classification',task='profession',row_id=r['row_id'],component=r['document_sha256'],method=method_name,seed=c['probe_seed'],operation='delete',split='dev',label=r['label'],gender=r['gender'],prediction=int(logit>0),logit=float(logit))
            acc={f'{y}/{g}':float(np.mean([(q>0)==r['label'] for q,r in zip(logits,dev) if r['label']==y and r['gender']==g])) for y in [0,1] for g in [0,1]}
            results[method_name]=dict(profession=float(np.mean([(q>0)==r['label'] for q,r in zip(logits,dev)])),gender=float(np.mean([(q>0)==r['gender'] for q,r in zip(logits,dev)])),worst_group=min(acc.values()),groups=acc)
            w.progress('RESULT',method=method_name,result=results[method_name])
        write(w.run/'SOURCE_RESULTS.json',dict(results=results,scope='Published human source operation on balanced original dev; source reproduction/adaptation, no target correspondence result.',probe=dict(lr=c['probe_lr'],batch=c['probe_batch_size'],epochs=1,seed=c['probe_seed']),truncated=truncated))
        w.checks.update(source_members_55=sum(map(len,source['members'].values()))==55,disjoint_documents=not any(r['document_sha256'] in train_hashes for r in dev),all_methods_recorded=len(results)==len(c['methods']))
    except Exception:error=traceback.format_exc()
    finally:
        for h in hooks:h.remove()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
