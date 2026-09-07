"""Measure fixed-budget compact source function along the existing training curve.

The raw-path gradient field and natural-data checkpoints already exist. This
consumer neither trains an SAE nor changes the frozen confirmation experiment.
"""
import argparse
import json
import traceback
from pathlib import Path
from composition_runtime import CompositionRun, ROOT, write, np
from run_composition_functional_source import source_coordinates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_composition_checkpoint_function.py',['scripts/run_composition_functional_source.py'])
    error=None
    try:
        work.load();cfg=work.cfg;torch=work.torch
        original=ROOT/cfg['source_run'];gradients=np.load(work.checked(original/'discovery_gradients.npz'))
        ids=gradients['row_ids'];assert np.array_equal(ids,np.flatnonzero(work.discovery))
        checkpoints=json.loads(work.checked(ROOT/cfg['training_run']/'checkpoints.json').read_text())['checkpoints']
        selected_steps=cfg.get('checkpoint_steps',[256,1024,4096])
        checkpoints=[r for r in checkpoints if r['step'] in selected_steps]
        assert {(r['step'],r['seed']) for r in checkpoints}=={(step,seed) for step in selected_steps for seed in range(1,6)}
        from sparsify import SparseCoder
        components=[];final_replay=[]
        for spec in checkpoints:
            path=Path(spec['path']);work.checked(path/'sae.safetensors');work.checked(path/'cfg.json')
            sae=SparseCoder.load_from_disk(path,device='cuda').eval();decoder=sae.W_dec.detach().cpu().numpy().astype(np.float64)
            with torch.no_grad():
                acts,indices,_=sae.encode(torch.as_tensor(work.h[:,:2].reshape(-1,work.h.shape[-1]),device='cuda'))
                z=torch.zeros((len(acts),sae.num_latents),device='cuda').scatter_(1,indices,acts).cpu().numpy().reshape(work.n,2,-1)
            del sae
            q=np.zeros_like(work.h,dtype=np.float64)
            for factor,slot in [('number',1),('time',0)]:
                dz=z[work.donors[factor],slot].astype(np.float64)-z[:,slot].astype(np.float64)
                scores=np.mean(dz[ids]*(gradients[factor]@decoder.T),axis=0)
                budget=cfg['source_budgets'][factor];members=np.argsort(-scores,kind='stable')[:budget]
                source=source_coordinates(dz,decoder,members);q[:,slot]=source['coordinates']@source['basis'].T
                if spec['step']==4096:
                    old=np.load(work.checked(original/f'seed{spec["seed"]}_source_coordinates.npz'))
                    prefix=f'{factor}_{budget}'
                    final_replay.append(bool(np.array_equal(members,old[prefix+'_support']) and np.allclose(q[:,slot],old[prefix+'_coordinates']@old[prefix+'_basis'].T,rtol=1e-9,atol=1e-9)))
                components.append(dict(step=spec['step'],seed=spec['seed'],factor=factor,budget=budget,members=members.tolist(),scores=scores[members].tolist(),span_error=source['span_error'],checkpoint_sha256=spec['sha256']))
            for factor,slot in [('number',1),('time',0),('joint',None)]:
                delta=q.copy()
                if slot is not None:delta[:,1-slot]=0
                work.measure('source_checkpoint',factor,delta,seed=spec['seed'],step=spec['step'])
            work.progress('CHECKPOINT_FUNCTION_COMPLETE',step=spec['step'],seed=spec['seed'])
        write(work.run/'source_components.json',dict(rows=components,ranking='Same frozen raw-path gradient ranking rule, re-evaluated on each independent checkpoint; final checkpoint exactly replays original source support and operation',discovery_row_ids=ids.tolist()))
        work.checks['final_source_replay']=len(final_replay)==(10 if 4096 in selected_steps else 0) and all(final_replay)
        work.checks['all_rows']=len(work.metrics)==len(checkpoints)*3*work.n
        work.checks['unique']=len(work.metrics)==len({(r['step'],r['seed'],r['factor'],r['row_id']) for r in work.metrics})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
