"""Discovery-only functional member ranking, verified by actual native edits."""
import argparse, json, traceback
from pathlib import Path
from composition_runtime import CompositionRun, ROOT, write, np


def source_coordinates(dz, decoder, support):
    _,singular,vt=np.linalg.svd(decoder[support].astype(np.float64),full_matrices=False)
    rank=int(np.sum(singular>singular[0]*1e-10)); basis=vt[:rank].T
    coefficients=decoder[support]@basis; coordinates=dz[:,support]@coefficients
    error=float(np.linalg.norm(coordinates@basis.T-dz[:,support]@decoder[support])/max(np.linalg.norm(coordinates@basis.T),1e-12))
    return dict(support=support,basis=basis,coefficients=coefficients,coordinates=coordinates,span_error=error,rank=rank)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True,type=Path); args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_composition_functional_source.py',[]); error=None
    try:
        work.load(); cfg=work.cfg; torch=work.torch; ids=np.flatnonzero(work.discovery); gradients={}
        for factor,slot in [('number',1),('time',0)]:
            raw=work.h[work.donors[factor]]-work.h; grad=np.zeros((len(ids),work.h.shape[-1]),np.float64)
            for alpha,weight in zip(cfg['gradient_path_points'],cfg['gradient_path_weights']):
                for off in range(0,len(ids),cfg['gradient_batch_size']):
                    ix=ids[off:off+cfg['gradient_batch_size']]; delta=np.zeros_like(work.h[ix]); delta[:,slot]=alpha*raw[ix,slot]
                    wanted=work.labels[[work.rows[i]['expected_label_index'] for i in work.donors[factor][ix]]]; original=work.labels[[work.rows[i]['expected_label_index'] for i in ix]]
                    grad[off:off+len(ix)]+=weight*work.forward(ix,delta,slot,(wanted,original))
            gradients[factor]=grad; work.progress('GRADIENT_COMPLETE',factor=factor)
        np.savez_compressed(work.run/'discovery_gradients.npz',row_ids=ids,**gradients)
        from sparsify import SparseCoder
        components=[]
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed']; path=Path(spec['path']); work.checked(path/'sae.safetensors'); work.checked(path/'cfg.json')
            sae=SparseCoder.load_from_disk(path,device='cpu').eval(); decoder=sae.W_dec.detach().numpy().astype(np.float64); del sae
            codes=np.load(work.checked(ROOT/cfg['material_run']/f'seed{seed}_codes.npz'))
            variants={budget:np.zeros_like(work.h,dtype=np.float64) for budget in cfg['member_budgets']}; arrays={}
            for factor,slot in [('number',1),('time',0)]:
                z=codes[factor+'_z'].astype(np.float64); dz=z[work.donors[factor]]-z
                score=np.mean(dz[ids]*(gradients[factor]@decoder.T),axis=0)
                for budget in cfg['member_budgets']:
                    support=np.argsort(-score,kind='stable')[:budget]; source=source_coordinates(dz,decoder,support)
                    variants[budget][:,slot]=source['coordinates']@source['basis'].T
                    work.checks[f'span_seed{seed}_{factor}_{budget}']=source['span_error']<1e-9
                    components.append(dict(seed=seed,factor=factor,budget=budget,members=support.tolist(),scores=score[support].tolist(),rank=source['rank'],span_error=source['span_error'],positive_scores=int(np.sum(score[support]>0))))
                    for key in ['support','basis','coefficients','coordinates']:arrays[f'{factor}_{budget}_{key}']=source[key]
                arrays[factor+'_all_scores']=score
            np.savez_compressed(work.run/f'seed{seed}_source_coordinates.npz',**arrays)
            for budget,all_delta in variants.items():
                for factor,slot in [('number',1),('time',0),('joint',None)]:
                    delta=all_delta.copy()
                    if slot is not None:delta[:,1-slot]=0
                    work.measure(f'source_gain{budget}',factor,delta,seed=seed)
            work.progress('SOURCE_COMPLETE',seed=seed)
        write(work.run/'source_components.json',dict(rows=components,ranking='Mean signed desired-minus-base logit integrated-gradient gain along raw donor path, discovery only',discovery_row_ids=ids.tolist()))
        work.checks['all_rows']=len(work.metrics)==work.n*len(cfg['sae_checkpoints'])*len(cfg['member_budgets'])*3
        work.checks['unique']=len(work.metrics)==len({(r['seed'],r['factor'],r['method'],r['row_id']) for r in work.metrics})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'; (work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
