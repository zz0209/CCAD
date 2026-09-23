import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from export_finite_member_confirmation import identity, read, write


def main():
    parser = argparse.ArgumentParser()
    for name in ('result', 'run-root', 'data', 'table', 'cost-output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = read(args.result)
    panel_path = Path(result['identities']['panel']['path'])
    panel = read(panel_path)
    assert len(panel['rows']) == result['documents']
    counts = dict(Counter(f"profession{row['profession']}_gender{row['gender']}" for row in panel['rows']))
    split_counts = dict(Counter(row['original_split'] for row in panel['rows']))
    calibration = []
    for group in result['identities']['groups']:
        run = Path(group['path']).parent
        config = read(run/'config.resolved.json')
        meta_path = Path(config['selected_groups_file']).with_name('selection_meta.json')
        meta = read(meta_path)
        calibration.append(dict(target_seed=meta['target_seed'], target_candidate_count=meta['target_candidate_count'],
            calibration_documents=len(meta['calibration_documents']), target_intervention_calls=meta['target_intervention_calls'],
            scalar_observation_dimensions=meta['scalar_observation_dimensions'], vector_observation_dimensions=meta['vector_observation_dimensions'],
            identity=identity(meta_path)))
    data = dict(result, export_written_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        calibration=calibration, panel_counts=counts, original_split_counts=split_counts,
        interpretation=dict(operation='Frozen 22 complete target members under none and common source-P backgrounds',
            source='Published associated-words W group with 11 members at six sites',
            observation='Same target intervention call budget; vector method reads 512 pooled coordinates and scalar calibration reads one original-head response',
            deployment='One frozen target group execution followed by the requested frozen later head; no per-member probing or refitting on evaluation documents',
            benchmark='Held intervention-cohort biographies with original dataset splits retained; not an official full-benchmark score'),
        export_provenance=dict(result=identity(args.result), exporter=identity(Path(__file__))))
    write(args.data, data)
    stats = result['statistics']
    source_values = stats['direct_calibration_n8/clean/source_accuracy']['value'], stats['direct_calibration_n8/conditional/source_accuracy']['value']
    lines = [r'\begingroup', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrr}', r'\toprule',
        r'Selection & nRMSE & Clean (\%) & After $P$ (\%) \\', r'\midrule']
    lines.append(' & '.join(['Source $W$', '0.000', f'{100*source_values[0]:.2f}', f'{100*source_values[1]:.2f}'])+r' \\')
    names = [('native_support', 'Native support'), ('direct_calibration_n8', 'Direct scalar'),
        ('activation_amplitude_n8', 'Activation gain'), ('path_calibration_n8', 'Path gain'), ('finite_vector_n8', 'Finite vector')]
    for method, label in names:
        error = stats[method+'/primary_nrmse']['value']
        assert error is not None
        cells = [label, f'{error:.3f}', f"{100*stats[method+'/clean/actual_accuracy']['value']:.2f}",
            f"{100*stats[method+'/conditional/actual_accuracy']['value']:.2f}"]
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    if args.table.exists():
        raise FileExistsError(args.table)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text('\n'.join(lines), encoding='utf-8')
    runs = []
    for path in sorted(args.run_root.iterdir()):
        if not path.is_dir():
            continue
        config, status = read(path/'config.resolved.json'), read(path/'status.json')
        assert config['run_parent'] == 'FINITE_VECTOR_REUSE_20260923' and status['status'] == 'PASS'
        summary = read(path/'metrics.summary.json')
        runs.append(dict(run=path.as_posix(), status=status,
            **{key: summary[key] for key in ('wall_seconds', 'process_cpu_seconds', 'sequence_forwards', 'token_forwards', 'peak_allocated_bytes')},
            directory_bytes=sum(file.stat().st_size for file in path.rglob('*') if file.is_file())))
    totals = {key: sum(row[key] for row in runs) for key in ('wall_seconds', 'process_cpu_seconds', 'sequence_forwards', 'token_forwards', 'directory_bytes')}
    totals['peak_allocated_bytes'] = max(row['peak_allocated_bytes'] for row in runs)
    cost = dict(pass_runs=runs, pass_total=totals, preparation_failure=dict(
        event='First launch exited before run creation because the D-drive run parent did not exist; the parent was then created',
        scientific_forward_calls=0, exact_elapsed_seconds=None,
        evidence=dict(path='E:/Projects/SAE_Lab/CCAD/master_log.md', line=11063, written_at_utc='2026-09-23T09:48:42Z')),
        full_elapsed_seconds_exact=None, measured_pass_driver_seconds=totals['wall_seconds'],
        scope='PASS summaries measure actual driver time; the pre-computation launch duration was not measured and is not invented')
    write(args.cost_output, cost)
    cost_lines = ['# 有限向量复用的实际投入', '',
        f"{len(runs)}个PASS运行合计{totals['wall_seconds']:.6f} driver秒、{totals['process_cpu_seconds']:.6f} process CPU秒、{totals['sequence_forwards']}条sequence forwards与{totals['token_forwards']}个token forwards。", '',
        f"峰值CUDA allocated为{totals['peak_allocated_bytes']} bytes，全部运行目录为{totals['directory_bytes']} bytes。", '',
        '首次父目录准备失败发生在run创建和模型计算之前，科学前向调用为0，准确耗时未知。上述PASS总耗时单独报告，完整启动经过不填写伪精确总时长。', '',
        f"最终确认使用{result['documents']}篇文档、{len(result['fixed_targets'])}个固定目标和四个冻结head。原始数据集划分与八个profession×gender格数量由最终panel读取并保存于论文JSON。", '',
        '方法校准预算与全部研究运行费用分别报告。向量方法读取512维实际作用，同预算标量方法读取原head的一维作用；完整组确认没有重新求解成员或训练head。', '']
    args.cost_output.with_suffix('.md').write_text('\n'.join(cost_lines), encoding='utf-8')
    print(dict(table=args.table.as_posix(), data=args.data.as_posix(), cost=args.cost_output.as_posix(),
        pass_runs=len(runs), pass_driver_seconds=totals['wall_seconds']), flush=True)


if __name__ == '__main__':
    main()
