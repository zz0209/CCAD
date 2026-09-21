from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np


def score(arrays, method, documents):
    values = []
    for name in ['singular','plural','full']:
        source = arrays[name][documents]
        clean = arrays['none'][documents]
        prediction = arrays[method+'__'+name][documents]
        energy = np.mean((source-clean)**2)
        if energy <= 1e-12:
            raise ValueError('No measurable source effect')
        values.append(np.sqrt(np.mean((prediction-source)**2)/energy))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['identities'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(path)
    data = []
    designs = []
    methods = ['local_action','recorded_action','state_feedback','raw_readout']
    seeds = [json.loads((run/'config.resolved.json').read_text())['target_seed'] for run in args.runs]
    if seeds != freeze['target_seeds']:
        raise ValueError('Unexpected target set or order')
    source_run = Path(freeze['source_reference_run'])
    source_identity = json.loads((source_run/'source_identity.json').read_text())
    source_selection = json.loads((source_run/'source_design.json').read_text())['selection']
    studies = []
    for run in args.runs:
        if json.loads((run/'status.json').read_text())['status'] != 'PASS':
            raise ValueError(run)
        arrays = dict(np.load(run/'responses.npz'))
        design = json.loads((run/'source_design.json').read_text())
        if design['selection'] != source_selection or json.loads((run/'source_identity.json').read_text()) != source_identity:
            raise ValueError('Source program identity mismatch')
        if designs and design['rows'] != designs[0]['rows']:
            raise ValueError('Panel changed')
        if data and any(not np.array_equal(arrays[key], data[0][key]) for key in ['none','singular','plural','full']):
            raise ValueError('Source changed')
        data.append(arrays)
        designs.append(design)
        studies.append(dict(run=str(run), mean_nrmse={method:score(arrays,method,np.arange(len(design['rows']))) for method in methods},
                            requests=json.loads((run/'target_analysis.json').read_text())['metrics']))
    rows = designs[0]['rows']
    clusters = []
    for structure in sorted({row['structure'] for row in rows}):
        keys = sorted({row['pair_sha256'] for row in rows if row['structure'] == structure})
        clusters.append([np.array([i for i,row in enumerate(rows) if row['structure']==structure and row['pair_sha256']==key]) for key in keys])
    rng = np.random.default_rng(2026092107)
    controls = [method for method in methods if method != 'state_feedback']
    samples = {method:[] for method in controls}
    for _ in range(2000):
        documents = np.concatenate([np.concatenate([cell[i] for i in rng.integers(len(cell),size=len(cell))]) for cell in clusters])
        scores = {method:float(np.mean([score(arrays,method,documents) for arrays in data])) for method in methods}
        for method in controls:
            samples[method].append(scores[method]-scores['state_feedback'])
    comparisons = {method:dict(reduction=float(np.mean([study['mean_nrmse'][method]-study['mean_nrmse']['state_feedback'] for study in studies])),
                              ci95=np.quantile(samples[method],[.025,.975]).tolist()) for method in controls}
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),studies=studies,comparisons=comparisons,
                  checks=dict(freeze_identity=True,same_source=True,same_panel=True),
                  documents=len(rows), clusters=sum(map(len,clusters)),
                  scope=freeze['statistics'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
