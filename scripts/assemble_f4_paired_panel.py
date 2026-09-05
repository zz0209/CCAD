"""Assemble fixed decoded/code FCC comparisons, without refitting or endpoints."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ccad.artifacts import sha256
from run_r011s1_raw_hook_asset import write_json as write,entry


def match_norm(raw,reference):
    """Preserve raw direction; a zero raw cannot match a nonzero reference."""
    a=np.linalg.norm(raw,axis=1);b=np.linalg.norm(reference,axis=1)
    if np.any((a==0)&(b!=0)):raise ValueError('Zero raw direction with nonzero reference norm')
    ratio=np.divide(b,a,out=np.zeros_like(b),where=a!=0)
    matched=raw*ratio[:,None]
    return matched,ratio


def compile_frozen_task_panel(tasks,cache,decoders,decoded,code,source_native,source_seed=2):
    """Apply saved weights to new inputs; never fit or read behavioral outcomes."""
    from run_f4_agreement_source import swap_indices
    for key in decoded.files:
        if not key.startswith('fcc_w_'):np.testing.assert_array_equal(decoded[key],code[key])
    basis=decoded['basis'];bank={};methods=[]
    for target in (1,3,4,5):
        d=decoders[target]
        labels=[('fcc','fcc'),('fcc_codes','fcc_codes'),('native','native'),('single','single'),('random','random')]
        for label,operation in labels:
            key=f'{label}_target{target}';methods.append(dict(key=key,target_seed=target,operation=operation))
            for axis in ('subject','attractor'):
                donor=swap_indices(tasks,axis);x=cache[f'codes_{target}']-cache[f'codes_{target}'][donor]
                if label=='fcc':delta=(x@d@decoded[f'fcc_w_{target}'])@basis.T
                elif label=='fcc_codes':delta=(x@code[f'fcc_w_{target}'])@basis.T
                else:
                    ids=decoded[f'{label}_ids_{target}'];g=decoded[f'{label}_g_{target}'];delta=(x[:,ids]*g)@d[ids]
                bank[f'{key}_{axis}']=delta
    methods.append(dict(key='raw_fitted',target_seed=None,operation='raw_basis_transport'))
    sid=source_native['ids'];sg=source_native['g'];sd=decoders[source_seed][sid]
    for axis in ('subject','attractor'):
        donor=swap_indices(tasks,axis)
        bank[f'raw_fitted_{axis}']=((cache['hidden']-cache['hidden'][donor])@decoded['raw_w'])@basis.T
        bank[f'source_native_reference_{axis}']=((cache[f'codes_{source_seed}'][:,sid]-cache[f'codes_{source_seed}'][donor][:,sid])*sg)@sd
    for target in (1,3,4,5):
        key=f'raw_matched_target{target}';methods.append(dict(key=key,target_seed=target,operation='raw_norm_matched_to_code_fcc'))
        for axis in ('subject','attractor'):
            bank[f'{key}_{axis}'],_=match_norm(bank[f'raw_fitted_{axis}'],bank[f'fcc_codes_target{target}_{axis}'])
    if not all(np.isfinite(x).all() for x in bank.values()):raise ValueError('Nonfinite compiled operation')
    return bank,methods


def norm_matched_panel():
    parent=ROOT/'artifacts/f4_task_paired_panel_v1'
    expected={'compiled_natural_deltas.npz':'416b1885cdb3c90bdc08feb892d49db01fe9dd9a571288e3573bbb784a975576',
              'compiled_methods.json':'bf2fc64118fb2321bf042c4ad602f777e6fc10eced6eb968a4c09b4a6f3fd686'}
    for name,digest in expected.items():
        if sha256(parent/name)!=digest:raise ValueError('Parent identity changed')
    out=ROOT/'artifacts/f4_task_norm_matched_panel_v1';out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((parent/'compiled_methods.json').read_text());bank=np.load(parent/'compiled_natural_deltas.npz',allow_pickle=False)
    methods=[r for r in meta['methods'] if r['operation']=='fcc_codes' or r['key']=='raw_fitted']
    result={f'{m["key"]}_{axis}':bank[f'{m["key"]}_{axis}'] for m in methods for axis in ('subject','attractor')}
    checks=[]
    for target in (1,3,4,5):
        key=f'raw_matched_target{target}';methods.append(dict(key=key,target_seed=target,operation='raw_norm_matched_to_code_fcc'))
        for axis in ('subject','attractor'):
            raw=bank[f'raw_fitted_{axis}'];ref=bank[f'fcc_codes_target{target}_{axis}']
            matched,ratio=match_norm(raw,ref);result[f'{key}_{axis}']=matched
            err=float(np.max(np.abs(np.linalg.norm(matched,axis=1)-np.linalg.norm(ref,axis=1))))
            if err>1e-10:raise ValueError('Norm matching failed')
            checks.append(dict(target_seed=target,axis=axis,maximum_norm_difference=err,ratios=ratio.tolist(),zero_raw_rows=int(np.sum(np.linalg.norm(raw,axis=1)==0))))
    for axis in ('subject','attractor'):result[f'source_native_reference_{axis}']=bank[f'source_native_reference_{axis}']
    np.savez_compressed(out/'compiled_natural_deltas.npz',**result)
    write(out/'compiled_methods.json',dict(meta,methods=methods))
    write(out/'assembly.json',dict(inputs=[entry(parent/name,'CCAD frozen compiled data','parent') for name in expected],
        checks=checks,method_count=len(methods),script_sha256=sha256(Path(__file__)),endpoint_data_read=False,
        rule='Per-input and axis raw*norm(codeFCC)/norm(raw); existing common source-native scale applied later. No fit or behavioral selection. Zero raw with nonzero reference errors.'))
    (out/'assembly_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({name:sha256(out/name) for name in expected}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--norm-matched',action='store_true');args=p.parse_args()
    if args.norm_matched:return norm_matched_panel()
    parents=[ROOT/'runs'/f'F4_task_paired_{mode}_fit_v1_20260905' for mode in ('decoded','codes')]
    out=ROOT/'artifacts/f4_task_paired_panel_v1';out.mkdir(parents=True,exist_ok=False)
    for p in parents:
        if json.loads((p/'status.json').read_text())['status']!='PASS':raise ValueError('Fit failed')
    meta=[json.loads((p/'compiled_methods.json').read_text()) for p in parents]
    for key in ('sample_ids','basis','source_native_sha256','output_rank'):
        if meta[0][key]!=meta[1][key]:raise ValueError(f'Identity mismatch: {key}')
    banks=[np.load(p/'compiled_natural_deltas.npz',allow_pickle=False) for p in parents]
    result={k:banks[0][k] for k in banks[0].files}
    checks={k:float(np.max(np.abs(banks[0][k]-banks[1][k]))) for k in banks[0].files if not k.startswith('fcc_')}
    if any(checks.values()):raise ValueError('Non-FCC controls changed')
    methods=meta[0]['methods']
    for method in meta[1]['methods']:
        if method['operation']!='fcc':continue
        old=method['key'];new=old.replace('fcc_','fcc_codes_',1)
        methods.append(dict(method,key=new,operation='fcc_codes'))
        for axis in ('subject','attractor'):result[f'{new}_{axis}']=banks[1][f'{old}_{axis}']
    np.savez_compressed(out/'compiled_natural_deltas.npz',**result)
    write(out/'compiled_methods.json',dict(meta[0],methods=methods))
    inputs=[entry(p/name,'CCAD frozen fit','parent') for p in parents for name in ('compiled_methods.json','compiled_natural_deltas.npz','config.resolved.json')]
    write(out/'assembly.json',dict(inputs=inputs,control_max_abs_errors=checks,method_count=len(methods),
        script_sha256=sha256(Path(__file__)),endpoint_data_read=False,scope='Assembly only, no fitting or selection by outcome'))
    print(json.dumps({name:sha256(out/name) for name in ('compiled_natural_deltas.npz','compiled_methods.json')}))


if __name__=='__main__':main()
