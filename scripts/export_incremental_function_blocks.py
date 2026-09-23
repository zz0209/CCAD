import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


METHODS = [('global_frozen', 'Global frozen', 'Global'),
           ('group_frozen', 'Group frozen', 'Group'),
           ('global_replay', 'Global replay', 'Replay'),
           ('raw', 'Raw operators', 'Raw'), ('native', 'Original native', 'Native')]
ENDPOINTS = [('new_own', 'New'), ('old_own', 'Old'), ('new_combinations', 'Combination')]
METRICS = [('nrmse', 'NRMSE', 1, 3), ('rmse', 'RMSE', 1, 3),
           ('source_effect_rms', 'SourceRMS', 1, 3), ('actual_effect_rms', 'EffectRMS', 1, 3),
           ('accuracy', 'Accuracy', 100, 2), ('source_accuracy', 'SourceAccuracy', 100, 2),
           ('binary_agreement', 'Agreement', 100, 2), ('clean_correct_retention', 'CleanRetention', 100, 2),
           ('baseline_flip', 'Flip', 100, 2), ('stage_change_rmse', 'CallChange', 1, 3),
           ('stage_change_nrmse', 'CallChangeNRMSE', 1, 3), ('stage_binary_agreement', 'StageAgreement', 100, 2)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build(result_path, cost_path, quality_paths):
    result, cost = read(result_path), read(cost_path)
    stats = result['statistics']
    assert result['query_order'] and result['bootstrap']['draws'] > 0
    macro_lines, sources = [], {}

    def macro(name, value, source, multiplier=1, digits=3):
        assert value is not None and math.isfinite(value)
        rendered = f'{value*multiplier:.{digits}f}'
        macro_lines.append('\\newcommand{\\IFB'+name+'}{'+rendered+'}')
        sources['IFB'+name] = dict(source=source, value=value, multiplier=multiplier,
                                  decimal_places=digits, rendered=rendered)
        return rendered

    def estimate(name, value, source, multiplier, digits):
        if value['value'] is None:
            return r'\textemdash'
        text = macro(name, value['value'], source+'/value', multiplier, digits)
        if value['ci95'] is not None:
            for suffix, index in [('Lower', 0), ('Upper', 1)]:
                macro(name+suffix, value['ci95'][index], source+'/ci95/'+str(index), multiplier, digits)
        return text

    table_rows = []
    for method, label, prefix in METHODS:
        stage = 0 if method == 'native' else 2
        selected, displayed = {}, {}
        for endpoint, endpoint_name in ENDPOINTS:
            for metric, metric_name, multiplier, digits in METRICS:
                key = f'{method}/stage{stage}/{endpoint}/{metric}'
                if key not in stats:
                    continue
                selected[endpoint+'/'+metric] = dict(source_key=key, **stats[key])
                displayed[endpoint+'/'+metric] = estimate(prefix+endpoint_name+metric_name, stats[key],
                                                        'result/statistics/'+key, multiplier, digits)
        cells = [label]+[displayed[endpoint+'/nrmse'] for endpoint, _ in ENDPOINTS]
        cells += [displayed.get('old_own/stage_change_rmse', r'\textemdash')]
        cells += [displayed['new_combinations/'+key] for key in ('binary_agreement', 'accuracy', 'clean_correct_retention')]
        table_rows.append(dict(method=method, label=label, selected_statistics=selected, rendered=cells))
    for first, _, first_prefix in METHODS[:3]:
        for second, _, second_prefix in METHODS:
            if first == second:
                continue
            for endpoint, endpoint_name in ENDPOINTS:
                for metric, metric_name, multiplier, digits in METRICS:
                    key = first+'_minus_'+second+'/'+endpoint+'/'+metric
                    if key in result['paired_contrasts']:
                        estimate(first_prefix+'Vs'+second_prefix+endpoint_name+metric_name,
                                 result['paired_contrasts'][key], 'result/paired_contrasts/'+key, multiplier, digits)
    macro('Rows', result['rows'], 'result/rows', digits=0)
    macro('Targets', len(result['fixed_targets']), 'length(result/fixed_targets)', digits=0)
    macro('ArrivalOrders', len(result['fixed_arrivals']), 'length(result/fixed_arrivals)', digits=0)
    macro('BootstrapDraws', result['bootstrap']['draws'], 'result/bootstrap/draws', digits=0)
    for name, field, multiplier, digits in [('DriverSeconds', 'driver_seconds', 1, 2),
        ('AssetGiB', 'bytes', 1/1024**3, 3), ('PeakCudaGiB', 'maximum_run_peak_allocated_bytes', 1/1024**3, 3)]:
        value = cost['totals'][field]
        if value is not None:
            macro(name, value, 'cost/totals/'+field, multiplier, digits)
    quality = cost['quality']
    seeds = sorted(quality['seeds'])
    quality_means, quality_rows = {}, []
    for method, prefix in [('global_frozen', 'Global'), ('group_frozen', 'Group'),
                           ('global_replay', 'Replay'), ('original', 'Original')]:
        records = [row for row in quality['rows'] if row['method'] == method]
        assert sorted(set(row['seed'] for row in records)) == seeds
        assert len({(row['seed'], row['arrival']) for row in records}) == len(records)
        expected_arrivals = {None} if method == 'original' else set(result['fixed_arrivals'])
        for seed in seeds:
            assert {row['arrival'] for row in records if row['seed'] == seed} == expected_arrivals
        quality_means[method] = {}
        for key, suffix, digits in [('fve', 'FVE', 4), ('ce_recovered', 'CERecovery', 4), ('l0', 'LZero', 2)]:
            by_seed = {str(seed): sum(row[key] for row in records if row['seed'] == seed)/len(expected_arrivals) for seed in seeds}
            mean = sum(by_seed.values())/len(seeds)
            quality_means[method][key] = dict(mean=mean, by_seed=by_seed, seeds=seeds,
                                             arrival_count_per_seed=len(expected_arrivals))
            macro(prefix+suffix, mean, 'quality_descriptive_means/'+method+'/'+key+'/mean', digits=digits)
    labels = {method: label for method, label, _ in METHODS}
    labels['original'] = 'Original dictionary'
    groups = sorted(quality['grouped'], key=lambda item: ('' if item['arrival'] is None else item['arrival'], item['method']))
    for group in groups:
        assert sorted(group['seeds']) == seeds
        cells = [group['arrival'] or r'\textemdash', labels[group['method']]]
        for key, digits in [('fve', 4), ('ce_recovered', 4), ('l0', 2), ('alive', 1)]:
            cells.append(f"{group['statistics'][key]['mean']:.{digits}f}")
        quality_rows.append(dict(arrival=group['arrival'], method=group['method'], seeds=seeds,
                                 statistics=group['statistics'], rendered=cells))
    quality_table = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}',
                     r'\begin{tabular}{llrrrr}', r'\toprule',
                     r'Arrival & Dictionary & FVE & CE recovery & $L_0$ & Alive \\', r'\midrule']
    quality_table += [' & '.join(row['rendered'])+r' \\' for row in quality_rows]
    quality_table += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    table = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'Method & \multicolumn{3}{c}{Response nRMSE} & \shortstack{Old-call\\change} & \multicolumn{3}{c}{New combinations (\%)} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){6-8}',
             r'& New & Old & Combined & RMSE & Agreement & Correct & Retained \\', r'\midrule']
    table += [' & '.join(row['rendered'])+r' \\' for row in table_rows]
    table += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    payload = dict(export_written_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                   provenance=dict(result=identity(result_path), cost=identity(cost_path),
                                   quality=[identity(path) for path in quality_paths], exporter=identity(Path(__file__))),
                   table_rows=table_rows, macros=sources, quality_table_rows=quality_rows,
                   quality_descriptive_means=quality_means,
                   definitions=dict(response=result['normalization'], resampling=result['bootstrap'],
                       new='Arriving function on its corresponding grammar', old='The two earlier functions on their corresponding grammars',
                       combinations='Three untrained combined requests containing the arriving function, evaluated on all three grammars',
                       agreement='Target and source post-intervention full-sentence good-minus-bad margins have the same positive/nonpositive classification',
                       correct='Post-intervention good-minus-bad margin is positive',
                       retained='Fraction of clean-correct sentence pairs that remain correct after intervention, averaged with the recorded task/request weights',
                       old_call_change='Actual stage-two minus stage-one margins for the earlier functions on their own grammars; absent stage-one model execution is shown as unavailable',
                       percentage_display='Accuracy and agreement percentages; corresponding contrasts use percentage points',
                       inference='Targets and arrival orders are fixed dependent conditions; sentence pairs are resampled within grammar',
                       quality='Five independent initializations per arrival; overall adapted means average arrivals within each seed and then average seeds. Arrival orders are repeated conditions, not independent seeds. Original means use each seed once.',
                       cost=cost['interpretation']), result=result, cost=cost,
                   quality=[dict(identity=identity(path), result=read(path)) for path in quality_paths])
    return payload, '\n'.join(table), '\n'.join(macro_lines)+'\n', '\n'.join(quality_table)


def main():
    parser = argparse.ArgumentParser()
    for name in ('result', 'cost', 'data', 'table', 'values', 'quality-table'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--quality', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    for path in (args.data, args.table, args.values, args.quality_table):
        if path.exists():
            raise FileExistsError(path)
    data, table, values, quality_table = build(args.result, args.cost, args.quality)
    assert sorted(data['cost']['quality']['seeds']) == [1, 2, 3, 4, 5]
    assert len(data['cost']['quality']['rows']) == 50
    for path, content in [(args.data, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n'),
                          (args.table, table), (args.values, values), (args.quality_table, quality_table)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    print(json.dumps(dict(data=str(args.data), table=str(args.table), values=str(args.values), quality_table=str(args.quality_table))))


if __name__ == '__main__':
    main()
