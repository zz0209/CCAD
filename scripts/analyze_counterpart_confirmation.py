from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np


def score(values, method, queries, indices):
    scores = []
    for query in queries:
        source = values[query][indices]
        clean = values['none'][indices]
        energy = np.mean((source-clean)**2)
        if energy <= 1e-12:
            raise ValueError('Source effect has zero energy')
        scores.append(np.sqrt(np.mean((values[method+'__'+query][indices]-source)**2)/energy))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text())
    for path,digest in freeze['identities'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError(path)
    methods = freeze['methods']
    families = {'parts':['singular','plural'],'union':['full'],'all':['singular','plural','full']}
    arrays, studies, panels = [], [], []
    for run in args.runs:
        if json.loads((run/'status.json').read_text())['status']!='PASS':
            raise ValueError(run)
        config = json.loads((run/'config.resolved.json').read_text())
        design = json.loads((run/'source_design.json').read_text())
        values = dict(np.load(run/'responses.npz'))
        if panels and design!=panels[0]:
            raise ValueError('Source program or panel changed')
        if arrays and any(not np.array_equal(values[q],arrays[0][q]) for q in ['none','singular','plural','full']):
            raise ValueError('Source responses changed')
        if not np.array_equal(values['counterpart_action_span_64__full'],values['counterpart_swapped_action_span_64__full']):
            raise ValueError('Swapped union changed')
        metrics = {family:{method:score(values,method,queries,np.arange(len(design['rows']))) for method in methods}
                   for family,queries in families.items()}
        studies.append(dict(run=str(run),target_seed=config['target_seed'],metrics=metrics))
        arrays.append(values)
        panels.append(design)
    if [s['target_seed'] for s in studies]!=freeze['target_seeds']:
        raise ValueError('Target set changed')
    rows = panels[0]['rows']
    cells = []
    for structure in sorted({r['structure'] for r in rows}):
        keys = sorted({r['pair_sha256'] for r in rows if r['structure']==structure})
        cells.append([np.array([i for i,r in enumerate(rows) if r['structure']==structure and r['pair_sha256']==key]) for key in keys])
    winner = 'counterpart_action_span_64'
    comparisons = {family:{method:[] for method in methods if method!=winner} for family in families}
    rng = np.random.default_rng(2026092114)
    for _ in range(2000):
        indices = np.concatenate([np.concatenate([cell[i] for i in rng.integers(len(cell),size=len(cell))]) for cell in cells])
        for family,queries in families.items():
            result = {method:float(np.mean([score(v,method,queries,indices) for v in arrays])) for method in methods}
            for method in comparisons[family]:
                comparisons[family][method].append(result[method]-result[winner])
    output = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),studies=studies,
        mean={family:{m:float(np.mean([s['metrics'][family][m] for s in studies])) for m in methods} for family in families},
        comparisons={family:{m:dict(reduction=float(np.mean([s['metrics'][family][m]-s['metrics'][family][winner] for s in studies])),
            ci95=np.quantile(samples,[.025,.975]).tolist()) for m,samples in contrasts.items()} for family,contrasts in comparisons.items()},
        documents=len(rows),clusters=sum(map(len,cells)),statistics=freeze['statistics'],
        checks=dict(freeze_identity=True,same_source=True,same_panel=True,swapped_union_exact=True))
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(mean=output['mean'],comparisons=output['comparisons']),indent=2))


if __name__=='__main__':
    main()
