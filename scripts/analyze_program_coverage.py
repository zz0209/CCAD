import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--evaluation-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    quality = json.loads((args.training_run/'representation_quality.json').read_text())
    representation = []
    for method in sorted({r['method'] for r in quality}):
        for family, choose in [('singletons', lambda q:'+' not in q and q not in ['full', 'clean']),
                               ('pairs', lambda q:'+' in q), ('full', lambda q:q == 'full')]:
            rows = [r for r in quality if r['method'] == method and choose(r['query'])]
            representation.append(dict(condition=method, family=family,
                change_nrmse=float(np.sqrt(sum(r['change_error'] for r in rows)/sum(r['change_energy'] for r in rows))),
                normalized_reconstruction=float(sum(r['reconstruction_error'] for r in rows)/sum(r['clean_variance'] for r in rows))))
    reference = None
    function = []
    arrays = {}
    for folder in args.evaluation_runs:
        status = json.loads((folder/'status.json').read_text())
        assert status['status'] == 'PASS', folder
        config = json.loads((folder/'config.resolved.json').read_text())
        condition = config['dictionary_condition']
        metrics = [json.loads(line) for line in (folder/'metrics.raw.jsonl').read_text().splitlines()]
        rows = [r for r in metrics if r['kind'] == 'classification']
        docs = sorted({r['component'] for r in rows})
        queries = config['queries']
        index = {(r['method'], r['operation'], r['component']):r['logit'] for r in rows}
        assert len(index) == len(rows)
        clean = np.array([index['none', 'full', d] for d in docs])
        source = np.array([[index['source', q, d] for q in queries] for d in docs])
        identity = (docs, queries, clean, source)
        if reference is None:
            reference = identity
        else:
            assert docs == reference[0] and queries == reference[1]
            np.testing.assert_array_equal(clean, reference[2])
            np.testing.assert_array_equal(source, reference[3])
        for method in config['methods']:
            if method in ['source', 'none']:
                continue
            values = np.array([[index[method, q, d] for q in queries] for d in docs])
            arrays[condition+'__'+method] = values
            for family, choose in [('singletons', lambda q:'+' not in q and q != 'full'),
                                   ('pairs', lambda q:'+' in q), ('full', lambda q:q == 'full')]:
                ids = [i for i,q in enumerate(queries) if choose(q)]
                err = values[:, ids]-source[:, ids]
                effect = source[:, ids]-clean[:, None]
                function.append(dict(condition=condition, method=method, family=family,
                    nrmse=float(np.sqrt(np.square(err).sum()/np.square(effect).sum())),
                    source_effect_rms=float(np.sqrt(np.square(effect).mean())), documents=len(docs), requests=len(ids)))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(evidence='exposed development,one target,matched frozen relation fit',
                   representation=representation, function=function, source_arrays_equal=True)
    with (args.output/'ANALYSIS.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (args.output/'FUNCTION_COMPARISON.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(function[0]))
        writer.writeheader()
        writer.writerows(function)
    np.savez_compressed(args.output/'function_arrays.npz', source=reference[3], clean=reference[2],
                        documents=reference[0], queries=reference[1], **arrays)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
