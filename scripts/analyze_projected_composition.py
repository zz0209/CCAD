"""Measure order and additive ambiguities for an actual frozen source family."""
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors import safe_open
from ccad.projected_correspondence import ordered_operator


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family-run',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();start=time.perf_counter();cpu=time.process_time();inputs=[]
    if args.output.exists():raise FileExistsError(args.output)
    def checked(path):
        path=Path(path);h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        inputs.append(dict(path=str(path.resolve()),sha256=h.hexdigest(),bytes=path.stat().st_size));return path
    cfg=json.loads(checked(args.family_run/'config.resolved.json').read_text())
    if json.loads(checked(args.family_run/'status.json').read_text())['status']!='PASS':raise ValueError('Correspondence family must pass')
    source=Path(cfg['source_run']);panel=json.loads(checked(source/'panel.json').read_text())['rows']
    held=np.array([r['row_id'] for r in panel if r['split']=='held_component_development'])
    donors=np.array([panel[i]['donor_id'] for i in held]);codes=np.load(checked(source/f'seed{cfg["source_seed"]}_codes.npz'))['codes']
    with safe_open(checked(Path(cfg['sae_root'])/f'seed_{cfg["source_seed"]}/sae.safetensors'),framework='np') as f:decoder=f.get_tensor('W_dec')
    x=(codes[donors].astype(float)-codes[held].astype(float))@decoder.astype(float)
    a=np.load(checked(args.family_run/'source_family.npz'));p=a['projectors'];binary=[tuple(c) for c in a['controls']]
    matrices={c:b for c,b in zip(binary,a['operators'])};matrices[(0,0,0)]=np.zeros_like(p[0])
    reverse={c:ordered_operator(p[::-1],np.array(c)[::-1]) for c in matrices}
    rows=[];summary=[]
    for c in binary:
        q=x@matrices[c].T;qr=x@reverse[c].T;add=x@sum(v*pp for v,pp in zip(c,p)).T
        for j,i in enumerate(held):
            rows.append(dict(row_id=int(i),component=panel[i]['component'],task=panel[i]['task'],control=list(map(int,c)),
                ordered_energy=float(q[j]@q[j]),reverse_error=float(np.sum((q[j]-qr[j])**2)),
                additive_error=float(np.sum((q[j]-add[j])**2))))
        summary.append(dict(control=list(map(int,c)),reverse_relative_rms=float(np.linalg.norm(q-qr)/np.linalg.norm(q)),
                            additive_relative_rms=float(np.linalg.norm(q-add)/np.linalg.norm(q))))
    interpolation=[]
    for c in [[.5,0,0],[1,.5,0],[.25,.75,.5]]:
        direct=ordered_operator(p,c)-ordered_operator(p[::-1],c[::-1]);mixture=np.zeros_like(p[0])
        for b in itertools.product([0,1],repeat=3):
            weight=np.prod([c[j] if value else 1-c[j] for j,value in enumerate(b)])
            mixture+=weight*(matrices[b]-reverse[b])
        interpolation.append(dict(control=c,max_matrix_error=float(np.max(np.abs(direct-mixture))),
            actual_source_vector_error=float(np.max(np.abs(x@(direct-mixture).T)))))
    result=dict(analysis_id='FINAL5_R17_ORDERED_COMPOSITION_v1',written_at_utc=datetime.now(timezone.utc).isoformat(),
        family_run=str(args.family_run),source_run=str(source),rows=rows,summary=summary,interpolation=interpolation,inputs=inputs,
        environment=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__),
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        scope='Actual frozen source projectors and held-development donor-code differences, evaluated in float64 algebra. No LM calls or semantic claim. Order reversal agrees on all singletons; joint deviations and vertex interpolation are measured, not assumed. Parent source may still be finishing its separate raw-DAS phase; fitted family/source files are fixed.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,interpolation=interpolation,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
