"""Measure actual union interactions in saved source controls, without LM calls."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import time
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from safetensors import safe_open


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-run',type=Path,required=True)
    ap.add_argument('--coverage-run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();start=time.perf_counter();cpu=time.process_time();inputs=[]
    def checked(path):
        path=Path(path);h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        inputs.append(dict(path=str(path.resolve()),sha256=h.hexdigest(),bytes=path.stat().st_size))
        return path
    def read(path):return json.loads(checked(path).read_text())
    sc=read(args.source_run/'config.resolved.json');cc=read(args.coverage_run/'config.resolved.json')
    if any(sc[k]!=cc[k] for k in ['sae_root','model_revision','layer']):raise ValueError('Source and coverage material differ')
    if read(args.coverage_run/'status.json')['status']!='PASS':raise ValueError('Coverage must be complete')
    source_status=read(args.source_run/'status.json')['status']
    sp=read(args.source_run/'panel.json')['rows'];cp=read(args.coverage_run/'panel.json')['rows']
    key=lambda r:(r['pair_key'],r['entity'],r['task'])
    source_rows={key(r):r for r in sp if r['split']=='held_component_development'}
    ids=np.array([r['row_id'] for r in cp if r['split']=='held_component_development'])
    for i in ids:
        r=cp[i];other=source_rows[key(r)]
        if any(r[k]!=other[k] for k in ['text','tokens','donor_entity','expected_ids','donor_expected_ids']):
            raise ValueError('Held contexts differ')
    donors=np.array([cp[i]['donor_id'] for i in ids]);seed=sc['source_seed']
    codes=np.load(checked(args.coverage_run/f'seed{seed}_codes.npz'))['codes']
    if codes.ndim!=2:raise ValueError('Requires compact entity-end codes')
    path=checked(Path(sc['sae_root'])/f'seed_{seed}/sae.safetensors')
    rows=[];arrays={};methods=[]
    for kind in sc['methods']:
        if kind not in ['native_shared','native_exclusive','sae_mdbm'] or not (args.source_run/f'{kind}_fit.json').exists():continue
        fit=read(args.source_run/f'{kind}_fit.json')
        gates=np.load(checked(args.source_run/f'{kind}_gates.npz'))['gates']
        selected=np.flatnonzero(gates.any(1));g=gates[selected].astype(float)
        with safe_open(str(path),framework='np') as f:
            decoder=np.concatenate([f.get_slice('W_dec')[int(j):int(j)+1] for j in selected]).astype(float)
        dz=codes[donors][:,selected].astype(float)-codes[ids][:,selected]
        singles=[(dz*g[:,a])@decoder for a in range(3)]
        for count in [2,3]:
            for active in itertools.combinations(range(3),count):
                control=np.zeros(3);control[list(active)]=1
                union=(dz*(1-np.prod(1-g*control,axis=1)))@decoder
                additive=sum(singles[a] for a in active);error=union-additive
                name=''.join(map(str,control.astype(int)));prefix=kind+'_'+name
                for j,i in enumerate(ids):
                    rows.append(dict(method=kind,control=name,row_id=int(i),component=cp[i]['component'],task=cp[i]['task'],
                        union_energy=float(union[j]@union[j]),additive_energy=float(additive[j]@additive[j]),
                        interaction_error_energy=float(error[j]@error[j])))
                arrays[prefix+'_union']=union.astype(np.float32);arrays[prefix+'_additive']=additive.astype(np.float32)
        methods.append(dict(method=kind,selected_step=fit['selected_step'],active_members=len(selected)))
    summary=[]
    for kind in methods:
        for control in ['110','101','011','111']:
            rr=[r for r in rows if r['method']==kind['method'] and r['control']==control]
            denominator=sum(r['union_energy'] for r in rr);numerator=sum(r['interaction_error_energy'] for r in rr)
            summary.append(dict(method=kind['method'],control=control,relative_rms_interaction=(numerator/denominator)**.5 if denominator else None,
                union_energy=denominator,interaction_error_energy=numerator))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'),row_ids=ids,**arrays)
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),source_parent_status=source_status,methods=methods,rows=rows,summary=summary,
        inputs=inputs,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        scope='Saved source-operation hook algebra on held development rows; no new model forwards or target correspondence. Parent status and completed method states are explicit. Ratio sums squared errors and union energies before taking the root; not a mean of row ratios. No downstream or semantic conclusion follows from nonadditivity alone.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(source_parent_status=source_status,summary=summary,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
