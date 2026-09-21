import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-run', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert json.loads((args.evaluation_run/'status.json').read_text())['status'] == 'PASS'
    assert json.loads((args.training_run/'status.json').read_text())['status'] == 'PASS'
    config = json.loads((args.evaluation_run/'config.resolved.json').read_text())
    rows = [json.loads(line) for line in (args.evaluation_run/'metrics.raw.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['kind'] == 'classification']
    queries = config['queries']
    documents = sorted({r['component'] for r in rows})
    index = {(r['method'], r['operation'], r['component']):r for r in rows}
    assert len(index) == len(rows)
    clean = np.array([index['none', 'full', d]['logit'] for d in documents])
    source = np.array([[index['source', q, d]['logit'] for q in queries] for d in documents])
    methods = [m for m in config['methods'] if m not in ['none', 'source']]
    values = {m: np.array([[index[m, q, d]['logit'] for q in queries] for d in documents]) for m in methods}
    rng = np.random.default_rng(606)
    strata = {}
    for i,d in enumerate(documents):
        row = index['none', 'full', d]
        strata.setdefault((row['label'], row['gender']), []).append(i)
    resamples = np.concatenate([rng.choice(ix, size=(4000, len(ix)), replace=True)
                               for ix in strata.values()], axis=1)
    summary = []
    for family, select in [('singletons', lambda q:'+' not in q and q != 'full'),
                           ('pairs', lambda q:'+' in q), ('full', lambda q:q == 'full')]:
        ids = [i for i,q in enumerate(queries) if select(q)]
        denominator = ((source[:, ids]-clean[:, None])**2).sum(1)
        boot = {}
        estimates = {}
        for method in methods:
            error = ((values[method][:, ids]-source[:, ids])**2).sum(1)
            estimates[method] = float(np.sqrt(error.sum()/denominator.sum()))
            boot[method] = np.sqrt(error[resamples].sum(1)/denominator[resamples].sum(1))
            summary.append(dict(method=method, family=family, nrmse=estimates[method],
                                documents=len(documents), requests=len(ids)))
        for alternative in ['local_whole', 'native', 'raw']:
            improvement = boot[alternative]-boot['local_parts']
            summary.append(dict(method=alternative+' minus local_parts', family=family,
                                nrmse=estimates[alternative]-estimates['local_parts'],
                                documents=len(documents), requests=len(ids),
                                interval=np.quantile(improvement, [.025,.975]).tolist()))
    local = json.loads((args.training_run/'local_quality.json').read_text())
    quality = []
    for method in sorted({r['method'] for r in local}):
        selected = [r for r in local if r['method'] == method]
        quality.append(dict(method=method, local_nrmse=float(np.sqrt(
            sum(r['part_error'] for r in selected)/sum(r['source_energy'] for r in selected))),
            clean_fve=1-sum(r['reconstruction_error'] for r in selected)/sum(r['clean_variance'] for r in selected)))
    result = dict(evidence='Exposed development,32 biographies,one target;paired document intervals conditional on this target',
                  functional=summary, representation=quality)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'ANALYSIS.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with (args.output/'COMPARISON.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method','family','nrmse','documents','requests','interval'])
        writer.writeheader()
        writer.writerows(summary)
    np.savez_compressed(args.output/'function_arrays.npz', documents=documents, queries=queries,
                        clean=clean, source=source, resamples=resamples, **values)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
