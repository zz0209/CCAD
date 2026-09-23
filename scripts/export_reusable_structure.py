import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def export(unit, paper):
    paths = dict(result=unit/'confirmation/RESULTS.json',
                 supplementary=unit/'confirmation/SUPPLEMENTARY_CONTRASTS.json',
                 models=unit/'frozen_models/MODELS.json', cost=unit/'COST.json')
    source = {key: read(path) for key, path in paths.items()}
    stats = source['result']['statistics']
    outputs = [paper/'data/reusable_structure.json', paper/'tables/reusable_structure.tex',
               paper/'tables/reusable_structure_values.tex']
    # 论文派生文件可由原始统计重新生成，统计输入保持只读。
    macros, macro_sources, rows = [], {}, []

    def macro(name, value, origin, scale=1, digits=3):
        rendered = f'{value*scale:.{digits}f}'
        macros.append('\\newcommand{\\RS'+name+'}{'+rendered+'}')
        macro_sources['RS'+name] = dict(value=value, source=origin, scale=scale,
                                        digits=digits, rendered=rendered)
        return rendered

    def estimate(name, entry, origin, scale=1, digits=3):
        value = macro(name, entry['value'], origin+'/value', scale, digits)
        for suffix, index in [('Lower', 0), ('Upper', 1)]:
            macro(name+suffix, entry['ci95'][index], origin+'/ci95/'+str(index), scale, digits)
        return value

    methods = [('M_source_higher', r'$M$ + source interactions', 'Higher'),
               ('B', r'Coordinate $B$', 'Coordinate'),
               ('M_additive', r'Additive $M$', 'Additive'),
               ('diagonal_source_higher', r'Diagonal + interactions', 'Diagonal'),
               ('source_unit', r'Source / original $q$', 'Source')]
    for method, label, prefix in methods:
        cells = [label]
        for key, suffix, scale, digits in [
                (method+'/prediction/unseen23/nrmse', 'Prediction', 1, 3),
                (method+'_fixed/selection/pooled_nrmse', 'FixedError', 1, 3),
                (method+'_fixed/selection/accuracy', 'FixedAccuracy', 100, 2)]:
            cells.append(estimate(prefix+suffix, stats[key], 'result/statistics/'+key, scale, digits))
        rows.append(dict(method=method, cells=cells))
    rows.append(dict(method='oracle', cells=['Full-grid oracle', '--',
        estimate('OracleError', stats['oracle/selection/pooled_nrmse'],
                 'result/statistics/oracle/selection/pooled_nrmse'),
        estimate('OracleAccuracy', stats['oracle/selection/accuracy'],
                 'result/statistics/oracle/selection/accuracy', 100, 2)]))
    key = 'prediction/unseen23/nrmse'
    estimate('PredictionDifference', source['result']['paired_contrasts'][
        'M_source_higher_minus_M_additive'][key],
        'result/paired_contrasts/M_source_higher_minus_M_additive/'+key)
    for key, name, scale, digits in [
            ('all/pooled_nrmse', 'FixedDifference', 1, 3),
            ('all/accuracy', 'AccuracyDifference', 100, 2),
            ('P/accuracy', 'PAccuracyDifference', 100, 2)]:
        estimate(name, source['supplementary']['paired_contrasts'][key],
                 'supplementary/paired_contrasts/'+key, scale, digits)
    for method, prefix in [('M_additive_fixed', 'Additive'), ('original_request', 'Original')]:
        for part in ['P', 'W']:
            key = method+'/selection/'+part+'/pooled_nrmse'
            estimate(prefix+part+'Error', stats[key], 'result/statistics/'+key)
        key = method+'/selection/regret_l2'
        estimate(prefix+'Regret', stats[key], 'result/statistics/'+key)
    for term in ['interpolation', 'correspondence', 'cross', 'total']:
        key = 'mechanism/unseen23/'+term+'_normalized_square'
        estimate(term.title()+'Square', stats[key], 'result/statistics/'+key, digits=6)
    macro('DriverSeconds', source['cost']['totals']['driver_seconds'],
          'cost/totals/driver_seconds', digits=2)
    macro('SequenceForwards', source['cost']['totals']['sequence_forwards'],
          'cost/totals/sequence_forwards', digits=0)
    table = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Predictor & \shortstack{23-request\\nRMSE} & \shortstack{Fixed\\nRMSE} & \shortstack{Fixed\\acc. (\%)} \\',
             r'\midrule']
    table += [' & '.join(row['cells'])+r' \\' for row in rows]
    table += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    payload = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                   inputs={key: identity(path) for key, path in paths.items()},
                   exporter=identity(Path(__file__)), table_rows=rows, macros=macro_sources,
                   evidence=source,
                   information_cost=dict(additive_source_calibration_forwards=32,
                       higher_order_source_calibration_forwards=64,
                       target_singleton_calibration_forwards_per_seed=24,
                       baseline='Shared source q=0', fixed_runtime_source_forwards=0,
                       fixed_runtime_target_forwards_per_goal=1))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    outputs[1].write_text('\n'.join(table), encoding='utf-8')
    outputs[2].write_text('\n'.join(macros)+'\n', encoding='utf-8')
    print(json.dumps(dict(outputs=[str(path) for path in outputs], rows=len(rows))))


def register(unit, paper):
    path = paper/'EVIDENCE_INDEX.json'
    index = read(path)
    claim_id = 'shared_interaction_prediction_and_fixed_calls'
    assert claim_id not in index
    index[claim_id] = dict(
        evidence='Three singleton anchors on eight fitting biographies; frozen requests and coefficients; 23 unfitted requests and actual fixed-dose use on 128 new documents and targets4/5. Shared source interactions improve prediction; additive fixed calls improve response error while retaining P/N/W task tradeoffs.',
        analysis=str(unit/'confirmation/RESULTS.json'),
        supplementary=str(unit/'confirmation/SUPPLEMENTARY_CONTRASTS.json'),
        models=str(unit/'frozen_models/MODELS.json'),
        cost=str(unit/'COST.json'), exporter='scripts/export_reusable_structure.py',
        data=str(paper/'data/reusable_structure.json'),
        table=str(paper/'tables/reusable_structure.tex'),
        paper='app:reusable_structure',
        identities={name: identity(asset) for name, asset in dict(
            data=paper/'data/reusable_structure.json',
            result=unit/'confirmation/RESULTS.json',
            supplementary=unit/'confirmation/SUPPLEMENTARY_CONTRASTS.json').items()})
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(evidence_index=str(path), added_claim=claim_id)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', type=Path, required=True)
    parser.add_argument('--paper', type=Path, default=Path('paper'))
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    if args.register_only:
        register(args.unit, args.paper)
    else:
        export(args.unit, args.paper)
