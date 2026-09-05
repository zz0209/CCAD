"""Assemble fixed decoded/code FCC comparisons, without refitting or endpoints."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ccad.artifacts import sha256
from run_r011s1_raw_hook_asset import write_json as write,entry


def main():
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
