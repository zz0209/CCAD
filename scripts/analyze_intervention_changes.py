from pathlib import Path
import argparse
import json
import numpy as np


def summarize(folder):
    arrays=np.load(folder/'responses.npz')
    config=json.loads((folder/'config.resolved.json').read_text())
    output=[]
    for dataset in ['human','grammar']:
        source=arrays[dataset+'__source'].astype(np.float64)
        clean=arrays[dataset+'__none'].astype(np.float64)
        energy=np.mean((source-clean)**2,axis=1)
        if np.any(energy<=1e-12): raise ValueError('A source request has negligible energy')
        for key in arrays.files:
            if not key.startswith(dataset+'__') or key.endswith(('__source','__none')): continue
            error=np.mean((arrays[key].astype(np.float64)-source)**2,axis=1)
            row=dict(run=folder.name,dataset=dataset,method=key.split('__')[1],
                mean_query_nrmse=float(np.sqrt(error/energy).mean()),
                global_nrmse=float(np.sqrt(error.mean()/energy.mean())),
                query_errors=np.sqrt(error/energy).tolist(),source_energy=energy.tolist(),
                documents=source.shape[1],queries=source.shape[0])
            output.append(row)
    return dict(run=folder.name,rows=output,config_available=config is not None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    c=p.parse_args()
    results=[summarize(folder) for folder in c.runs]
    c.output.parent.mkdir(parents=True,exist_ok=True)
    c.output.write_text(json.dumps(dict(results=results,scope='Development results. Fixed source functions and target seed; shared document and query dependence.'),indent=2)+'\n')
    for result in results:
        for r in result['rows']:
            print(f"{r['run']} {r['dataset']} {r['method']} {r['mean_query_nrmse']:.6f} {r['global_nrmse']:.6f}")


if __name__=='__main__': main()
