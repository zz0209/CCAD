from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if json.loads((args.run/'status.json').read_text())['status']!='PASS':
        raise ValueError('Analysis requires a completed run')
    if args.output.exists(): raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    arrays=np.load(args.run/'pooled.npz')
    membership=json.loads((args.run/'membership.json').read_text())
    rows=membership['human_rows']
    tasks=[('composer_surgeon_orientation0',[5,25]),('composer_surgeon_orientation1',[5,25]),
           ('model_software_engineer_orientation0',[12,24]),('model_software_engineer_orientation1',[12,24])]
    result=[]; sources=[]
    for name,professions in tasks:
        path=root/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{name}__probe42.npz'
        probe=np.load(path); weight=probe['weight'].astype(float).ravel(); bias=float(probe['bias'].item())
        keep=np.array([r['profession'] in professions for r in rows])
        source=arrays['source'][:,keep].astype(float)@weight+bias
        clean=arrays['none'][:,keep].astype(float)@weight+bias
        energy=np.mean((source-clean)**2,axis=1)
        if np.any(energy<=1e-12): raise ValueError('A later-head request has insufficient source effect')
        for key in arrays.files:
            if key in ['source','none']: continue
            target=arrays[key][:,keep].astype(float)@weight+bias
            error=np.mean((target-source)**2,axis=1)
            result.append(dict(head=name,method=key,nrmse=float(np.sqrt(error/energy).mean()),
                               per_query=np.sqrt(error/energy).tolist(),source_energy=energy.tolist(),
                               agreement=float(((target>0)==(source>0)).mean()),documents=int(keep.sum())))
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    methods=sorted({r['method'] for r in result})
    means={m:float(np.mean([r['nrmse'] for r in result if r['method']==m])) for m in methods}
    args.output.write_text(json.dumps(dict(run=str(args.run),means=means,rows=result,heads=sources,
        scope='Four previously frozen heads on their respective cohorts. Heads excluded from source-profile estimation. Fixed source and target seed; developmental unless the run has a separate confirmation freeze.'),indent=2)+'\n')
    print(json.dumps(means,indent=2))


if __name__=='__main__': main()
