from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
BULK = Path('D:/CCAD_Storage/runs/final_science_20260920')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bulk-root', type=Path, default=BULK)
    parser.add_argument('--round-id', default='FINAL_SCIENCE_01')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for run in sorted(args.bulk_root.iterdir()):
        if not run.is_dir():
            continue
        c = json.loads((run / 'config.resolved.json').read_text())
        status = json.loads((run / 'status.json').read_text())
        path = run / 'metrics.summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {}
        state = status['status']
        note = None
        if run.name == 'INTERVENTION_PROGRAM_SMOKE_20260920':
            state = 'ARTIFACT_FINALIZATION_FAILURE'
            note = 'Completed predictions retained. Final summary raised KeyError for missing component; original RUNNING status retained. Corrected smoke has independent R2 identity. Master log records the failure.'
        if run.name == 'CONDITIONAL_PROFILE_DEVELOPMENT_T2_20260920':
            state = 'INTERRUPTED_AFTER_COMPLETED_GLOBAL_CONTROL'
            note = 'All20 ordinary-profile requests completed; owned worker stopped during redundant inactive-token inference. Original status and predictions retained; R2 completes the two cached methods. Stop-time interval is in master_log.'
        row = dict(run_id=run.name, path=str(run), status=state, original_status=status,
            wall_seconds=summary.get('wall_seconds'), cpu_seconds=summary.get('process_cpu_seconds'),
            peak_allocated_bytes=summary.get('peak_allocated_bytes'),
            sequence_forwards=summary.get('sequence_forwards'), token_forwards=summary.get('token_forwards'),
            bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),
            seed=c.get('target_seed', c.get('source_field_evaluation', {}).get('target_seeds')),
            member_allowance=c.get('member_budget', c.get('source_field_evaluation', {}).get('members', '2p per site')),
            inverse_steps=c.get('inverse_steps', c.get('source_field_evaluation', {}).get('inverse_steps')),
            source_metric=bool(c.get('source_metric_rows') or c.get('source_metric_cache') or c.get('source_field_evaluation')),
            confirm=c.get('evidence_level','').startswith('frozen'), smoke='SMOKE' in run.name.upper(),
            config_sha256=hashlib.sha256((run / 'config.resolved.json').read_bytes()).hexdigest(), note=note)
        if path.exists():
            row['summary_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), round_id=args.round_id, runs=rows,
        known_driver_seconds=sum(r['wall_seconds'] for r in rows if r['wall_seconds'] is not None),
        bulk_bytes=sum(r['bytes'] for r in rows),
        max_peak_allocated_bytes=max(r['peak_allocated_bytes'] or 0 for r in rows),
        scope='Driver duration includes loading, computation and export. Effective reasoning time, queue time and missing failed-run duration are not inferred. Wakeups and smokes are not scientific rounds.',
        metadata_clarifications=[
            'Confirmation dataset_revision and generic component labels are inherited development strings. Frozen panel paths, hashes, exact membership and source snapshots identify the new contexts.',
            'excluded_test_functions checks action-bank exclusions in generic training. Source-profile construction deliberately uses the published source functions on their old fit contexts; no leave-function-out claim applies to this arm.',
            'source_function_training=False in evaluation membership refers to absence of parameter fitting in that run. Cached source functional information was estimated in its recorded development run.',
            'The metric uses scalar source-response gradient second moments. Only the natural model-distribution score control is described as Fisher.',
            'Older inherited textual purpose/budget fields do not override executed member_budget, inverse_steps, variants, generic_program_steps or source_metric fields.',
            'Current support is shared across requests at a fixed state. Multi-site trajectories and subsequent states depend on earlier requested interventions.',
            'Task-adapted controls are available only for targets2to5 and retain their distinct trained dictionaries.',
            '900 seconds is the planning estimate for each confirmation driver; no runtime timeout is applied inside evaluation. Actual durations are reported.',
            'PROFILE_CONFIRMATION_ANALYSIS_V2.json uses ordered execution records for query axes; the original sorted-membership analysis and partial human-family summaries are superseded. Original prediction arrays are unchanged.',
            'PROGRAM_PATH_SMOKE completed its model run; the first analysis exposed the query-order error and cross-batch float32 differences. R2 verifies direct-versus-prefix equality on identical batches, with all earlier assets retained.'
        ])
    if args.round_id=='FINAL_SCIENCE_02':
        result['metadata_clarifications']=[
            'Agreement uses original public prefix/answer pairs, ten fixed published source members and its source-only profile; this is an annotated-member program, not the complete circuit benchmark.',
            'Agreement targets1to4 used the original frozen driver and target5 the optional-conditional implementation. Full target1 replay has seven bitwise-identical prediction arrays; CODE_EQUIVALENCE.json retains identities.',
            'The subject_pair field denotes the leading-noun number pair. The original primary bootstrap is retained, with a post-confirmation crossed-noun sensitivity analysis registered separately.',
            'The first conditional smoke used edited embeddings to classify active tokens and is superseded by R2. Full conditional development R2 and the frozen confirmation use functional lookup from clean embedding weights.',
            'The full conditional development panel was previously evaluated. Conditional confirmation uses new documents and requests, with a separate pre-outcome freeze.',
            'Cached and direct embedding execution differ slightly through numerical batching; completed global-control arrays and measured differences remain available.',
            'Known driver totals exclude the interrupted development duration; the stop-time bounds and last completed progress are in the master log. No active run is inferred from its retained RUNNING file.',
            'The configured budget is a planning bound. Evaluation reports actual duration; the per-update timeout applies to training.'
        ]
    if args.round_id=='FINAL_SCIENCE_03':
        result['metadata_clarifications']=[
            'The frozen primary compares source-profile and Euclidean inference using the same exact source-amplitude computation, member selection rule, capacity constraint and128steps.',
            'Qwen cycle directions use different source functions and their own source profiles; the dependent five-dictionary pool remains fixed in the uncertainty calculation.',
            'Original readout predictions omit fitted inputs and are invalid controls. READOUT_CORRECT runs restore the full inherited64-input bank without refitting.',
            'CONFIRMATION_ANALYSIS_V2 and its arrays replace only the original readout predictions. The primary and other methods remain exactly unchanged.',
            'Two-term question identities were already used and receive new contexts. Three-term operand multisets and all192complete prompts were absent from retained panels.',
            'Both symbolic and English formats were present in source fitting. New contexts are not claimed as new format families.',
            'Semantic hybrid success and agreement with source intervention answers are distinct endpoints. Invalid source answers remain failures under the frozen scoring convention.',
            'Failed smoke output and the original incomplete readout are retained with their original source snapshots.'
        ]
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['runs', 'metadata_clarifications']}, indent=2))


if __name__ == '__main__':
    main()
