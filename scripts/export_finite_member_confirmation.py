import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def identity(path):
    path = Path(path)
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--selection-meta', nargs='+', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--cost-output', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    cohort = read(args.cohort)
    panel = read(args.panel)
    split_counts = dict(Counter(row['original_split'] for row in panel['rows']))
    assert len(panel['rows']) == cohort['evaluation_documents'] == 256
    assert cohort['target_count'] == 2
    stats = cohort['statistics']
    source_stats = stats['direct_calibration_n8']
    source = dict(nrmse=0., calibration_documents=0,
        clean_accuracy=source_stats['clean/source_profession_accuracy'],
        conditional_accuracy=source_stats['conditional/source_profession_accuracy'],
        clean_effect_rms=source_stats['clean/source_whole_W_rms'],
        conditional_effect_rms=source_stats['conditional/source_whole_W_rms'])
    budgets = []
    for path in args.selection_meta:
        meta = read(path)
        target = int(read(Path(meta['run'])/'config.resolved.json')['target_seed'])
        candidates = len(meta['candidate_ids'])
        assert meta['methods']['direct_calibration_n8']['target_calibration_member_condition_calls'] == candidates*2*8
        budgets.append(dict(target_seed=target, candidates=candidates, calibration_documents=8, conditions=2,
            target_member_document_condition_calls=candidates*2*8, selection_meta=identity(path)))
    data = dict(written_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        source=source, statistics=stats, paired_contrasts=cohort['paired_contrasts'], calibration_budgets=budgets,
        scope=dict(source_group='Associated words, 11 source members at six sites', target_group='22 complete target members, site quotas 2/6/6/4/2/2',
            conditions=['No prior deletion', 'Common source pronouns deletion'], fixed_target_seeds=[4, 5],
            source_model='Pythia-70m-deduped', head='Fixed original source profession classifier',
            evaluation_documents=256, dataset_split_counts=split_counts,
            benchmark_status='Previously unused intervention-cohort biographies combining dataset train/test; not an official benchmark score',
            estimand=cohort['estimand'], bootstrap=cohort['bootstrap']),
        provenance=dict(cohort=identity(args.cohort), panel=identity(args.panel), exporter=identity(Path(__file__)),
            target_reports=cohort['target_reports']))
    write(args.data, data)
    labels = [('native_support', 'Native support', 0), ('direct_calibration_n8', 'Direct finite', 8),
              ('activation_amplitude_n8', 'Activation gain', 8), ('path_calibration_n8', 'Path gain', 8)]
    lines = [r'\begingroup', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Selection & Cal. & nRMSE & Clean & After $P$ \\', r'\midrule',
        f"Source $W$ & 0 & 0.000 & {100*source['clean_accuracy']['value']:.2f} & {100*source['conditional_accuracy']['value']:.2f}"+r' \\']
    for method, label, calibration in labels:
        values = stats[method]
        cells = [label, str(calibration), f"{values['joint/nrmse_common_source_scale']['value']:.3f}",
            f"{100*values['clean/profession_accuracy']['value']:.2f}", f"{100*values['conditional/profession_accuracy']['value']:.2f}"]
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    if args.table.exists():
        raise FileExistsError(args.table)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text('\n'.join(lines), encoding='utf-8')
    runs, failures = [], []
    total_keys = ('wall_seconds', 'process_cpu_seconds', 'sequence_forwards', 'token_forwards', 'directory_bytes')
    for run in sorted(args.run_root.iterdir()):
        if not run.is_dir():
            continue
        config = read(run/'config.resolved.json')
        assert config['run_parent'] == 'FINITE_RESPONSE_SPACE_20260923'
        status = read(run/'status.json')
        if status['status'] == 'PASS':
            summary = read(run/'metrics.summary.json')
            entry = dict(run=run.as_posix(), status=status, **{key: summary[key] for key in total_keys if key != 'directory_bytes'},
                peak_allocated_bytes=summary['peak_allocated_bytes'],
                directory_bytes=sum(path.stat().st_size for path in run.rglob('*') if path.is_file()))
            runs.append(entry)
        else:
            assert status['status'] == 'FAIL'
            progress = read(run/'progress.json')
            failures.append(dict(run=run.as_posix(), status=status, exact_wall_seconds=None,
                known_wall_seconds_lower_bound=progress['elapsed'], process_cpu_seconds_lower_bound=progress['process_cpu_seconds'],
                sequence_forwards_lower_bound=progress['sequence_forwards'], token_forwards_lower_bound=progress['token_forwards'],
                evidence=identity(run/'progress.json'),
                directory_bytes=sum(path.stat().st_size for path in run.rglob('*') if path.is_file())))
    totals = {key: sum(run[key] for run in runs) for key in total_keys}
    totals['peak_allocated_bytes'] = max(run['peak_allocated_bytes'] for run in runs)
    cost = dict(pass_runs=runs, pass_total=totals, failed_runs=failures,
        full_wall_seconds_exact=None if failures else totals['wall_seconds'],
        full_wall_seconds_lower_bound=totals['wall_seconds']+sum(row['known_wall_seconds_lower_bound'] for row in failures),
        scope='Actual run summaries and retained failure progress; method calibration budgets exclude held-response evaluation acquisition and are reported separately')
    write(args.cost_output, cost)
    cost_lines = ['# 有限成员作用单元的实际投入', '',
        f"{len(runs)}个PASS运行合计{totals['wall_seconds']:.6f} driver秒、{totals['process_cpu_seconds']:.6f} process CPU秒、{totals['sequence_forwards']}条sequence forwards及{totals['token_forwards']}个token forwards。", '',
        f"PASS运行最大CUDA allocated为{totals['peak_allocated_bytes']} bytes，目录合计{totals['directory_bytes']} bytes。", '']
    for row in failures:
        cost_lines += [f"失败运行{Path(row['run']).name}保留有效数组及错误记录。最后已知进度为{row['known_wall_seconds_lower_bound']:.6f}秒、{row['sequence_forwards_lower_bound']}条sequence forwards，完整结束时长未知。", '']
    cost_lines += [f"完整单元driver时长已知下界为{cost['full_wall_seconds_lower_bound']:.6f}秒，不提供伪精确总时长。", '',
        '确认使用一个source W组、22个完整目标成员、两个背景及每成员8篇目标校准文档。两个固定目标共用256篇此前未用于干预选择的文档，包含192篇dataset-test与64篇dataset-train；该混合队列的结果属于当前干预确认，不作为官方benchmark结果。', '',
        '同预算目标拟合分别使用4496与4512次成员条件文档操作。实际采集还包含source调用、路径梯度、开发评价和完整组合执行，全部计入上述运行投入。', '']
    args.cost_output.with_suffix('.md').write_text('\n'.join(cost_lines), encoding='utf-8')
    print(json.dumps(dict(table=args.table.as_posix(), data=args.data.as_posix(), cost=args.cost_output.as_posix(),
        pass_runs=len(runs), pass_seconds=totals['wall_seconds'], failed_runs=len(failures), wall_seconds_lower_bound=cost['full_wall_seconds_lower_bound'])))


if __name__ == '__main__':
    main()
