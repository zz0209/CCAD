import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


METHODS = [
    ('global_supervised', 'Global TopK', 'Global'),
    ('group_supervised', 'Group TopK', 'Group'),
    ('original', 'Original masks', 'Original'),
    ('raw', 'Raw operator', 'Raw'),
]
METRICS = [
    ('primary_nrmse', 'PartialNRMSE', 1, 3),
    ('primary_four_word_correct', 'PartialAccuracy', 100, 2),
    ('joint_from_joint/four_word_correct', 'JointAccuracy', 100, 2),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def render(value, multiplier, digits):
    assert value is not None and math.isfinite(value)
    return f'{value*multiplier:.{digits}f}'


def export(result_path, quality_path, cost_path, data_path, table_path, values_path):
    outputs = [data_path, table_path, values_path]
    assert len({path.resolve() for path in outputs}) == len(outputs)
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
    result, quality, cost = read(result_path), read(quality_path), read(cost_path)
    stats = result['statistics']
    macros, macro_sources, table_rows = [], {}, []

    def macro(name, value, source, multiplier=1, digits=3):
        text = render(value, multiplier, digits)
        macros.append('\\newcommand{\\FB'+name+'}{'+text+'}')
        macro_sources['FB'+name] = dict(source=source, value=value, multiplier=multiplier,
                                        decimal_places=digits, rendered=text)
        return text

    def estimate(name, entry, source, multiplier, digits):
        value = macro(name, entry['value'], source+'/value', multiplier, digits)
        assert len(entry['ci95']) == 2
        for suffix, position in [('Lower', 0), ('Upper', 1)]:
            macro(name+suffix, entry['ci95'][position], source+'/ci95/'+str(position), multiplier, digits)
        return value

    for method, label, prefix in METHODS:
        cells, measures = [label], {}
        for key, suffix, multiplier, digits in METRICS:
            source = method+'/'+key
            entry = stats[source]
            cells.append(estimate(prefix+suffix, entry, 'result/statistics/'+source, multiplier, digits))
            measures[key] = dict(source_key=source, **entry)
        table_rows.append(dict(method=method, label=label, measures=measures, rendered=cells))

    contrasts = [
        ('global_supervised_minus_original', 'GlobalVsOriginal'),
        ('global_supervised_minus_raw', 'GlobalVsRaw'),
        ('group_supervised_minus_global_supervised', 'GroupVsGlobal'),
    ]
    for comparison, prefix in contrasts:
        for key, suffix, multiplier, digits in METRICS:
            source = comparison+'/'+key
            estimate(prefix+suffix, result['paired_contrasts'][source],
                     'result/paired_contrasts/'+source, multiplier, digits)

    macro('Rows', result['rows'], 'result/rows', digits=0)
    macro('Subjects', len(result['subjects']), 'length(result/subjects)', digits=0)
    macro('Targets', len(result['fixed_targets']), 'length(result/fixed_targets)', digits=0)
    macro('BootstrapDraws', result['bootstrap']['draws'], 'result/bootstrap/draws', digits=0)
    macro('KnownDriverSeconds', cost['known_driver_seconds'], 'cost/known_driver_seconds', digits=2)
    macro('AssetGiB', cost['run_asset_bytes'], 'cost/run_asset_bytes', multiplier=1/1024**3, digits=3)
    if cost['maximum_reported_cuda_bytes'] is not None:
        macro('PeakCudaGiB', cost['maximum_reported_cuda_bytes'],
              'cost/maximum_reported_cuda_bytes', multiplier=1/1024**3, digits=3)
    for method, _, prefix in METHODS[:2]:
        entry = quality['fixed_seed_means'][method]
        for key, suffix in [('ce_recovered', 'CERecovery'), ('fve', 'FVE'), ('l0', 'LZero')]:
            macro(prefix+suffix, entry[key], 'quality/fixed_seed_means/'+method+'/'+key)

    # 两种准确率分别对应部分请求和联合请求，表格统一使用百分数。
    table = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Method & \shortstack{Partial\\nRMSE} & \shortstack{Partial\\acc. (\%)} & \shortstack{Joint\\acc. (\%)} \\',
             r'\midrule']
    table += [' & '.join(row['rendered'])+r' \\' for row in table_rows]
    table += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    payload = dict(
        export_written_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        provenance=dict(result=identity(result_path), quality=identity(quality_path), cost=identity(cost_path),
                        exporter=identity(Path(__file__))),
        table_rows=table_rows, macros=macro_sources,
        definitions=dict(
            partial_response=result['primary'],
            partial_accuracy='Four-word argmax equals the requested-factor change with the other factor preserved, averaged equally over two partial requests and fixed targets',
            joint_accuracy='Four-word argmax for joint_from_joint, exchanging the union of the two target groups once from the common donor',
            confidence_intervals=result['bootstrap'],
            accuracy_display='Percent; accuracy contrasts in percentage points',
            quality='Saved descriptive means across the seeds explicitly listed in QUALITY.json, independent of functional confirmation targets',
            cost='All recorded attempts in this unit; reused acquisition costs are preserved separately; known durations are lower bounds when unavailable runs are listed'),
        result=result, quality=quality, cost=cost)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    table_path.write_text('\n'.join(table), encoding='utf-8')
    values_path.write_text('\n'.join(macros)+'\n', encoding='utf-8')
    print(json.dumps(dict(data=str(data_path), table=str(table_path), values=str(values_path),
                          rows=result['rows'], fixed_targets=result['fixed_targets']), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    for name in ('result', 'quality', 'cost', 'data', 'table', 'values'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    export(args.result, args.quality, args.cost, args.data, args.table, args.values)


if __name__ == '__main__':
    main()
