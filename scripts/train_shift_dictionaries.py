"""Train replacement dictionaries at the published explanation's actual sites.

Uses the unchanged MIT dictionary_learning TopKTrainer at the configured commit.
Each site sees the same natural tokens. No biography/task data enter training.
"""
from pathlib import Path
import argparse,json,sys,time,platform,traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork,write


def site_module(model,site):
    if site=='embed':return model.gpt_neox.embed_in
    kind,n=site.split('_');layer=model.gpt_neox.layers[int(n)]
    return layer if kind=='resid' else getattr(layer,{'attn':'attention','mlp':'mlp'}[kind])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text())
    w=MultisiteWork(c,a.config,['scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']);error=None;hooks=[]
    try:
        import torch,transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.cuda.set_device(c['device']);torch.cuda.reset_peak_memory_stats()
        w.torch=torch;w.device=torch.device(c['device']);sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import TopKTrainer
        for rel in ['dictionary_learning/trainers/top_k.py','dictionary_learning/trainers/trainer.py','dictionary_learning/dictionary.py','LICENSE']:w.checked(Path(c['dictionary_source_dir'])/rel,'Pinned unchanged TopK training implementation','MIT')
        tok=json.loads(w.checked(c['token_manifest']).read_text());data={}
        for split,info in tok['outputs'].items():
            path=w.checked(info['path'],'Pinned natural tokens','ODC-By-1.0');assert w.inputs[-1]['sha256']==info['sha256'];data[split]=np.memmap(path,dtype=info['dtype'],mode='r').reshape(-1,128)
        bulk=Path(c['bulk_output_dir']);bulk.mkdir(parents=True,exist_ok=False)
        for f in ['config.json','tokenizer.json','model.safetensors']:w.checked(Path(c['model_local_dir'])/f,'Pythia70M pinned base','Apache-2.0')
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device);model.requires_grad_(False);model.config.use_cache=False
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,matmul_precision='highest',sites=c['sites'],device=str(w.device),dictionary_commit=c['dictionary_commit'])
        observed={};replacement={}
        def hook(site):
            def h(m,i,out):
                x=out[0] if isinstance(out,tuple) else out;observed[site]=x.detach()
                if site in replacement:
                    value=replacement[site]
                    x=torch.zeros_like(x) if value is None else value.reshape_as(x)
                    return (x,*out[1:]) if isinstance(out,tuple) else x
            return h
        for site in c['sites']:hooks.append(site_module(model,site).register_forward_hook(hook(site)))
        trainers={}
        for seed in c['seeds']:
            for site in c['sites']:
                trainers[seed,site]=TopKTrainer(steps=c['steps'],activation_dim=512,dict_size=c['dict_size'],k=c['k'],layer=-1 if site=='embed' else int(site.split('_')[1]),lm_name='EleutherAI/pythia-70m-deduped',lr=c['learning_rate'],warmup_steps=c['warmup_steps'],decay_start=c['decay_start'],threshold_start_step=c['threshold_start_step'],seed=seed,device=c['device'])
        write(w.run/'trainer_configs.json',dict(rows=[dict(seed=s,site=site,config=tr.config) for (s,site),tr in trainers.items()]))
        snapshots=[]
        def save(step):
            dest=bulk/f'step_{step}';dest.mkdir()
            for (seed,site),tr in trainers.items():
                path=dest/f'{site}_seed{seed}.pt';torch.save(tr.ae.state_dict(),path);snapshots.append(dict(step=step,seed=seed,site=site,path=str(path),tokens=step*c['batch_sequences']*128))
            write(w.run/'checkpoints.json',dict(checkpoints=snapshots))
        @torch.no_grad()
        def quality(step):
            ids=torch.tensor(np.array(data['validation'][:c['validation_sequences']],dtype='int64'),device=w.device);model.gpt_neox(ids,use_cache=False);xs={site:x.clone().flatten(0,1) for site,x in observed.items()}
            def ce():
                loss=model(ids,use_cache=False).logits
                return float(torch.nn.functional.cross_entropy(loss[:,:-1].reshape(-1,loss.shape[-1]),ids[:,1:].reshape(-1)))
            clean=ce();w.sequence_forwards+=2*len(ids);w.token_forwards+=2*ids.numel()
            for (seed,site),tr in trainers.items():
                x=xs[site];z=tr.ae.encode(x);recon=tr.ae.decode(z);counts=(z>0).sum(0);replacement[site]=recon;rc=ce();replacement[site]=None;zero=ce();replacement.clear();w.sequence_forwards+=2*len(ids);w.token_forwards+=2*ids.numel()
                w.record(kind='quality',task=site,row_id=step,component=site,seed=seed,method='topk',step=step,tokens=step*c['batch_sequences']*128,fve=float(1-(x-recon).square().sum()/(x-x.mean(0)).square().sum()),ce_clean=clean,ce_recon=rc,ce_zero=zero,ce_recovery=1-(rc-clean)/(zero-clean) if abs(zero-clean)>1e-9 else 0.,l0=float((z>0).sum(-1).float().mean()),alive=int((counts>0).sum()),dead=int((counts==0).sum()),decoder_norm_error=float((tr.ae.decoder.weight.norm(dim=0)-1).abs().max()))
            w.progress('QUALITY',step=step,rows=len(trainers))
        started=time.perf_counter();bs=c['batch_sequences'];losses={}
        for step in range(c['steps']):
            ids=torch.tensor(np.array(data['train'][step*bs:(step+1)*bs],dtype='int64'),device=w.device);assert len(ids)==bs
            with torch.no_grad():model.gpt_neox(ids,use_cache=False)
            xs={site:x.flatten(0,1) for site,x in observed.items()};w.sequence_forwards+=len(ids);w.token_forwards+=ids.numel()
            for (seed,site),tr in trainers.items():
                loss=tr.update(step,xs[site]);assert np.isfinite(loss);losses[f'{seed}/{site}']=loss
            done=step+1
            if done%32==0:w.progress('TRAINING',completed_steps=done,total_steps=c['steps'],training_elapsed=time.perf_counter()-started,projected_seconds=(time.perf_counter()-started)/done*c['steps'],losses=losses)
            if done in c['checkpoint_steps']:save(done);quality(done)
            if time.perf_counter()-started>c['budget_seconds']:
                if done not in c['checkpoint_steps']:save(done);quality(done)
                raise TimeoutError('Bounded target dictionary training; completed weights retained')
        w.checks['all_sites_and_checkpoints']=len(snapshots)==len(trainers)*len(c['checkpoint_steps'])
    except Exception:error=traceback.format_exc()
    finally:
        for h in hooks:h.remove()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
