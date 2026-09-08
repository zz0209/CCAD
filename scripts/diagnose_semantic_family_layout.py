"""Compare cached old-source arithmetic without any new-city intervention."""
from __future__ import annotations
import argparse,json,platform,time,traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);path=ap.parse_args().config;cfg=json.loads(path.read_text())
    w=MultisiteWork(cfg,path,['scripts/diagnose_semantic_family_layout.py','scripts/run_causalgym_multisite.py',
        'scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/semantic_family.py',
        'src/ccad/semantic_readout.py','src/ccad/ravel_controls.py','src/ccad/semantic_participation.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']);error=None
    try:
        import torch
        from safetensors import safe_open
        from ccad.semantic_family import make_family,get_bases,apply_torch
        torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        w.torch=torch;w.device=torch.device(cfg['device']);torch.cuda.set_device(w.device);torch.cuda.reset_peak_memory_stats(w.device)
        w.environment=dict(python=__import__('sys').executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device=str(w.device))
        freeze=json.loads(w.checked(ROOT/cfg['freeze_path']).read_text());failed=ROOT/cfg['failed_run']
        panel=json.loads(w.checked(ROOT/freeze['old_panel']).read_text())['rows'];donors=np.asarray([r['donor_id'] for r in panel]);held=np.asarray([i for i,r in enumerate(panel) if r['split']=='held_component_development'])
        results=[]
        for spec in freeze['sources']:
            if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Source diagnosis budget exhausted')
            seed=spec['source_seed'];sr=ROOT/spec['run'];pc=json.loads(w.checked(sr/'config.resolved.json').read_text())
            old=np.load(w.checked(sr/f'seed{seed}_codes.npz'));new=np.load(w.checked(failed/f'seed{seed}_codes.npz'))
            zold=old['codes'];znew=new['codes'][:len(panel)]
            code_error=float(np.max(np.abs(zold-znew)));same_positions=np.array_equal(old['positions'],new['positions'][:len(panel)])
            with safe_open(w.checked(Path(cfg['sae_root'])/f'seed_{seed}/sae.safetensors'),framework='pt',device=str(w.device)) as f:d=f.get_tensor('W_dec')
            with torch.no_grad():
                xold=torch.as_tensor(zold[donors]-zold,device=w.device)@d
                xnew=torch.as_tensor(znew[donors]-znew,device=w.device)@d
            torch.manual_seed(pc['optimizer_seed']);op=make_family(d.shape[1],3,pc['das_rank'],False).to(w.device)
            op.load_state_dict(torch.load(w.checked(sr/f"{spec['family']}_state.pt"),map_location=w.device,weights_only=True));op.eval()
            basis=torch.as_tensor(np.load(w.checked(ROOT/spec['basis_path']))['bases'],device=w.device)
            column_layout=basis.transpose(-1,-2).contiguous().transpose(-1,-2)
            with torch.no_grad(),torch.nn.utils.parametrize.cached():
                weights=get_bases(op);weight_error=float((weights-basis).abs().max())
                strides=list(op.rotations[0].weight.stride())
                for control in cfg['binary_controls']:
                    c=torch.as_tensor(control,device=w.device);expected=op(xold[held],None,c)
                    current=op(xnew[held],None,c)
                    for method,b in [('contiguous_basis',basis),('original_column_layout',column_layout)]:
                        actual=apply_torch(xnew[held],b,c);maximum=float((actual-expected).abs().max())
                        row=dict(source_seed=seed,control=control,method=method,max_code_error=code_error,positions_equal=bool(same_positions),
                            saved_basis_value_error=weight_error,decoder_difference_error=float((xnew-xold).abs().max()),
                            original_operator_error=float((current-expected).abs().max()),max_delta_error=maximum,
                            original_weight_stride=strides,applied_weight_stride=list(b[0].stride()))
                        results.append(row);w.record(kind='old_source_arithmetic',task='semantic_family',row_id=seed,component=f'source{seed}',seed=seed,
                            method=method,operation=''.join(map(str,control)),split='old_held_replay',max_delta_error=maximum)
            del op,basis,column_layout,weights,xold,xnew,d,zold,znew,old,new
            write(w.run/'layout_diagnosis.json',dict(rows=results,scope=cfg['scope']));w.progress('OLD_SOURCE_LAYOUT_DIAGNOSIS',source_seed=seed)
        w.checks['cached_old_codes_exact']=all(r['max_code_error']==0 and r['positions_equal'] for r in results)
        w.checks['basis_values_exact']=all(r['saved_basis_value_error']==0 for r in results)
        w.checks['original_operator_exact']=all(r['original_operator_error']==0 for r in results)
        w.checks['column_layout_exact']=all(r['max_delta_error']==0 for r in results if r['method']=='original_column_layout')
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
