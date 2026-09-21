"""Compare updated dictionaries on the same held-out natural-text states."""
from pathlib import Path
import argparse, json, sys, time, traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork
from train_shift_dictionaries import site_module


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text())
    w=MultisiteWork(c,a.config,['scripts/evaluate_shift_program_material.py','scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    handles=[];error=None
    try:
        import torch, transformers
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest');torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats();w.torch=torch;w.device=torch.device(c['device'])
        sys.path.extend([c['dictionary_source_dir'],c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__)
        for name in ['config.json','tokenizer.json','model.safetensors']:
            w.checked(Path(c['model_local_dir'])/name,'Pinned Pythia70M','Apache-2.0')
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py','Unchanged TopK implementation','MIT')
        tm=json.loads(w.checked(c['token_manifest'],'Natural train/validation split').read_text())['outputs']['validation']
        path=w.checked(tm['path'],'Held-out natural validation tokens','ODC-By-1.0')
        assert w.inputs[-1]['sha256']==tm['sha256']
        tokens=np.array(np.memmap(path,dtype='<u2',mode='r').reshape(-1,128)[c['sequence_indices']],dtype='int64')
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.requires_grad_(False);model.config.use_cache=False
        observed={};replacement={};offset=0
        def hook(site):
            def f(module,inputs,out):
                x=out[0] if isinstance(out,tuple) else out
                observed[site]=x.detach()
                if site in replacement:
                    value=replacement[site]
                    x=torch.zeros_like(x) if value is None else value[offset:offset+len(x)]
                return (x,*out[1:]) if isinstance(out,tuple) else x
            return f
        for site in c['sites']:handles.append(site_module(model,site).register_forward_hook(hook(site)))
        @torch.no_grad()
        def evaluate(capture=False):
            nonlocal offset
            losses=[];states={s:[] for s in c['sites']}
            for offset in range(0,len(tokens),c['eval_batch_size']):
                ids=torch.tensor(tokens[offset:offset+c['eval_batch_size']],device=w.device)
                logits=model(ids,use_cache=False).logits
                loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='sum')
                losses.append(float(loss));w.sequence_forwards+=len(ids);w.token_forwards+=ids.numel()
                if capture:
                    for site in c['sites']:states[site].append(observed[site].clone())
            return sum(losses)/(len(tokens)*127),{s:torch.cat(v) for s,v in states.items() if v}
        clean,xs=evaluate(True);zeros={}
        for site in c['sites']:
            replacement[site]=None;zeros[site]=evaluate()[0];replacement.clear()
        for seed in c['seeds']:
            for method in c['variants']:
                if c.get('dictionary_directories'):
                    directory=Path(c['dictionary_directories'][method].format(seed=seed))
                else:
                    directory=Path(c['original_directory']) if method=='original' else Path(c['adapted_root'])/f'IR04_shift_program_seed{seed}_v1_20260916'/method
                for site in c['sites']:
                    state=torch.load(w.checked(directory/f'{site}_seed{seed}.pt','Target dictionary checkpoint'),map_location='cpu',weights_only=True)
                    sae=AutoEncoderTopK(512,state['encoder.weight'].shape[0],int(state['k'])).to(w.device)
                    sae.load_state_dict(state);sae.eval();sae.requires_grad_(False)
                    x=xs[site];flat=x.flatten(0,1)
                    with torch.no_grad():
                        z=sae.encode(x);recon=sae.decode(z);counts=(z>0).sum((0,1))
                        replacement[site]=recon;ce=evaluate()[0];replacement.clear()
                        w.record(kind='quality',task='heldout_natural',component=site,row_id=f'{seed}/{method}/{site}',seed=seed,method=method,site=site,
                            fve=float(1-(x-recon).square().sum()/(flat-flat.mean(0)).square().sum()),ce_clean=clean,ce_recon=ce,ce_zero=zeros[site],
                            ce_recovery=1-(ce-clean)/(zeros[site]-clean),l0=float((z>0).sum(-1).float().mean()),alive=int((counts>0).sum()),dead=int((counts==0).sum()),
                            decoder_norm_error=float((sae.decoder.weight.norm(dim=0)-1).abs().max()),tokens=int(z.shape[0]*z.shape[1]))
                    del sae,z,recon,state;torch.cuda.empty_cache()
                    w.progress('QUALITY',seed=seed,method=method,site=site)
                    if time.perf_counter()-w.wall_start>c['budget_seconds']:raise TimeoutError('Material evaluation budget exhausted')
        w.checks['all_quality_cells']=True
    except Exception:error=traceback.format_exc()
    finally:
        for h in handles:h.remove()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
