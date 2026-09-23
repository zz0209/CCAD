from datetime import datetime, timezone
from pathlib import Path
import json
import time

import numpy as np

from analyze_interaction_structure import BITS, HIGHER, OPERATIONS, decompose, digest, summary


def main():
    started = time.perf_counter()
    root = Path('artifacts/scientific_reform_20260923')
    parent_path = root/'INTERACTION_STRUCTURE_ADDITIVE_SUMMARY.json'
    output = root/'INTERACTION_MIXING_DIAGNOSTIC.json'
    assert not output.exists()
    parent = json.loads(parent_path.read_text())
    fit = parent['shared_split']['fit_indices']
    check = parent['shared_split']['check_indices']
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Fixed exposed fit8/check56 diagnostic; no new data, coefficient tuning or confirmation',
        parent=dict(path=str(parent_path), sha256=digest(parent_path)),
        shared_split=parent['shared_split'], targets={}, common={},
        matrix_definition='A[source_part,target_part], predicted target singleton_j = sum_i source singleton_i * A[i,j]. Ordinary unregularized least squares without intercept over fit documents and 512 coordinates.',
        high_order='The original per-target joint_scalar coefficient is reused unchanged for every high-order term.')
    combined = {}
    for item in parent['inputs']:
        path = Path(item['array'])
        assert digest(path) == item['sha256']
        seed = str(json.loads((Path(item['run'])/'config.resolved.json').read_text())['target_seed'])
        coefficients = parent['targets'][seed]['coefficients']
        probe_path = Path(item['probe'])
        assert digest(probe_path) == item['probe_sha256']
        with np.load(probe_path) as probe:
            weight = probe['weight'].reshape(-1).astype(np.float64)
        with np.load(path) as data:
            documents = data['document_sha256'].tolist()
            assert [documents[i] for i in fit] == parent['shared_split']['fit_documents']
            assert [documents[i] for i in check] == parent['shared_split']['check_documents']
            names, ops = data['method_names'].tolist(), data['operation_names'].tolist()
            order = [ops.index(name) for name in OPERATIONS]
            baseline = data['baseline_pooled512'].astype(np.float64)
            source, source_effect, _ = decompose(data['pooled512'][:,names.index('source'),order].astype(np.float64), baseline)
            target, target_effect, _ = decompose(data['pooled512'][:,names.index('native'),order].astype(np.float64), baseline)
        x = np.stack([source[name] for name in OPERATIONS[:3]], axis=-1)
        y = np.stack([target[name] for name in OPERATIONS[:3]], axis=-1)
        matrix, residuals, rank, singular_values = np.linalg.lstsq(x[fit].reshape(-1,3), y[fit].reshape(-1,3), rcond=None)
        diagonal = np.diag([coefficients['singleton'][name]['value'] for name in OPERATIONS[:3]])
        predicted = {'separate_scalar': x[check]@diagonal, 'mixing_matrix': x[check]@matrix}
        current = dict(input=item, A=matrix.tolist(), diagonal=diagonal.tolist(), fit_rank=int(rank),
            fit_singular_values=singular_values.tolist(), fit_residual_sum_squares=residuals.tolist(),
            unchanged_joint_scalar=coefficients['joint_scalar']['value'], singleton={}, complete_response={}, per_document=[])
        per_doc = {method: {'singleton':{},'complete_response':{}} for method in predicted}
        for method, values in predicted.items():
            for j, name in enumerate(OPERATIONS[:3]):
                pred, actual, src = values[:,:,j], y[check,:,j], x[check,:,j]
                current['singleton'].setdefault(name,{})[method] = summary(pred,actual,src,weight)
                combined.setdefault(('singleton',name,method),[]).append((pred,actual,src))
                per_doc[method]['singleton'][name] = (pred,actual)
            for name, bits in zip(HIGHER,BITS[3:]):
                pred = sum(values[:,:,j] for j,bit in enumerate(BITS[:3]) if bit & bits == bit)
                pred = pred+coefficients['joint_scalar']['value']*sum(source[part][check]
                    for part,bit in zip(HIGHER,BITS[3:]) if bit & bits == bit)
                actual,src = target_effect[check,OPERATIONS.index(name)],source_effect[check,OPERATIONS.index(name)]
                current['complete_response'].setdefault(name,{})[method] = summary(pred,actual,src,weight)
                combined.setdefault(('complete_response',name,method),[]).append((pred,actual,src))
                per_doc[method]['complete_response'][name] = (pred,actual)
                if method == 'separate_scalar':
                    np.testing.assert_allclose(current['complete_response'][name][method]['rmse'],
                        parent['targets'][seed]['complete_response'][name]['joint_scalar']['rmse'],rtol=1e-12)
        for position,index in enumerate(check):
            row = dict(document_sha256=documents[index],methods={})
            for method,sections in per_doc.items():
                row['methods'][method] = {}
                for section,terms in sections.items():
                    row['methods'][method][section] = {name:dict(vector_rmse=float(np.sqrt(np.mean((pred[position]-actual[position])**2))),
                        head_prediction=float(pred[position]@weight), head_actual=float(actual[position]@weight),
                        head_error=float((pred[position]-actual[position])@weight)) for name,(pred,actual) in terms.items()}
            current['per_document'].append(row)
        result['targets'][seed] = current
    for (section,name,method),values in combined.items():
        pred,actual,src = [np.concatenate([row[i] for row in values]) for i in range(3)]
        result['common'].setdefault(section,{}).setdefault(name,{})[method] = summary(pred,actual,src,weight)
    result['script_sha256'] = digest(Path(__file__))
    result['cpu_analysis_wall_seconds'] = time.perf_counter()-started
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,
        matrices={seed:data['A'] for seed,data in result['targets'].items()},
        common={section:{name:{method:{key:stats[key] for key in ('rmse','nrmse','head_rmse')}
            for method,stats in methods.items()} for name,methods in values.items()}
            for section,values in result['common'].items()})))


if __name__ == '__main__':
    main()
