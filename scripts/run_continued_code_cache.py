"""Encode existing paired activations with continued target checkpoints.

The source teacher and panel remain unchanged. No new model activations or
labels select the continued checkpoint: the configured fixed budget does.
"""
import argparse, json, traceback
from pathlib import Path
from composition_runtime import CompositionRun, ROOT, write, np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_continued_code_cache.py',[]);error=None
    try:
        work.load();cfg=work.cfg;torch=work.torch
        from sparsify import SparseCoder
        panels={}
        for name,rel in cfg['cache_panels'].items():
            parent=ROOT/rel
            panel=json.loads(work.checked(parent/'panel.json').read_text())
            cache=np.load(work.checked(parent/'raw_cache.npz'))
            h=cache[f"layer{cfg['sae_layer']}"][:,:2]
            assert len(h)==len(panel['rows']) and np.array_equal(h[:,0],h[:,1])
            panels[name]=h;(work.run/name).mkdir()
        for spec in cfg['sae_checkpoints']:
            path=Path(spec['path']);work.checked(path/'sae.safetensors');work.checked(path/'cfg.json')
            sae=SparseCoder.load_from_disk(path,device='cuda').eval()
            for name,h in panels.items():
                with torch.no_grad():
                    acts,indices,_=sae.encode(torch.as_tensor(h.reshape(-1,h.shape[-1]),device='cuda'))
                    z=torch.zeros((len(acts),sae.num_latents),device='cuda').scatter_(1,indices,acts).cpu().numpy().reshape(len(h),2,-1)
                assert np.array_equal(z[:,0],z[:,1]) and np.isfinite(z).all()
                np.savez_compressed(work.run/name/f"seed{spec['seed']}_codes.npz",time_z=z[:,0],number_z=z[:,1])
                work.record(dict(kind='code_cache',seed=spec['seed'],panel=name,rows=len(h),latents=int(sae.num_latents),mean_l0=float(np.mean(np.count_nonzero(z[:,0],axis=1)))))
            del sae
            work.progress('ENCODE_COMPLETE',seed=spec['seed'])
        work.checks['all_seed_panel_codes']=len(work.metrics)==len(cfg['sae_checkpoints'])*len(panels)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
