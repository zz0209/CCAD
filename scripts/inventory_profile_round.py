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
    auxiliary = []
    for run in sorted(args.bulk_root.iterdir()):
        if not run.is_dir():
            continue
        if args.round_id in ['FINAL_SCIENCE_04', 'FINAL_SCIENCE_05'] and run.name.endswith('_checkpoints'):
            assert not (run/'config.resolved.json').exists()
            auxiliary.append(dict(path=str(run), bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())))
            continue
        c = json.loads((run / 'config.resolved.json').read_text())
        status = json.loads((run / 'status.json').read_text())
        path = run / 'metrics.summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {}
        state = status['status']
        note = None
        if run.name == 'FS05_HUMAN_BALANCED_SMOKE_20260921':
            state = 'INVALID_EMPTY_EVALUATION'
            note = 'Training completed, but the original panel filter selected no evaluation documents. Original PASS status is retained. V2 uses the actual development split and a nonempty-evaluation assertion.'
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
        bulk_bytes=sum(r['bytes'] for r in rows)+sum(r['bytes'] for r in auxiliary),
        auxiliary=auxiliary,
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
    if args.round_id == 'FINAL_SCIENCE_04':
        result['metadata_clarifications'] = [
            'Source-column identity smokes verify equality with the existing input-dependent rule. Shared-fit checks verify unchanged dictionaries and fixed relations.',
            'Human single-target fitting trains27early-site columns. Shared fitting trains all55columns across11sites and two dictionaries; these changes are not independently attributed.',
            'Shared scalar and full-column corrections receive the same512updates and balanced random two-step dictionary schedule. Each training dictionary receives256updates.',
            'Human shared fitting uses targets2/3 and confirms1/4/5. Grammar uses3/4 and confirms1/2/5. Target response fitting is absent when applying these corrections.',
            'Original source encoders and the base model remain required. New target members depend on the current code and dictionary; member allowance is matched, exact selected members can differ.',
            'Human confirmation has128fresh original-development biographies and new requests. Grammar has96fresh lexical contexts, new continuous coordinates and previously used binary mask families.',
            'Legacy evaluation split fields can readdev. The frozen panel, exact membership and checkpoint identities determine confirmation provenance.',
            'Per-target gain and program references retain512source-supervised updates and are evaluated on the identical fresh panels. Raw source-direction readout is retained.',
            'The initial missing-parent smoke attempt exited before run creation and model loading. Import and lexical-panel preparation failures are recorded inmaster_log and created no evaluated model outcomes.',
            'The configured time limit is checked during training. Evaluation reports measured wall time. Auxiliary checkpoint directories are included inbulk storage totals.'
        ]
    if args.round_id == 'FINAL_SCIENCE_05':
        result['metadata_clarifications'] = [
            'All new results are development. Human uses32exposed biographies and one target; infinitive uses32exposed contexts and one target; subject-number uses48exposed prefixes and the same grammatical target.',
            'The request comparison retains512updates and the same endpoint schedule. Balanced requests preserve nominal group mass; independent requests preserve it in expectation.',
            'Task-adapted labels in cross-explanation runs refer to the infinitive-trained checkpoint. No subject-number response fitting occurs in those runs.',
            'Generic objectives use the same natural-text source actions, excluding the evaluated member IDs. Only resid4 encoder and decoder are trained. All other loaded dictionaries remain frozen.',
            'Local state, final state and vocabulary distribution are distinct training targets. Their training losses are not comparable numerical endpoints.',
            'The source-coordinate covariance check is mathematical validation. It is not a Transformer generalization result.',
            'The first human smoke has empty evaluation and scientific status INVALID. Its original runtime metadata is preserved.',
            'Grammar member-mask bootstrap intervals are unavailable when fewer than90percent of resamples retain source-effect support. All point estimates and valid-draw counts remain available.'
        ]
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['runs', 'metadata_clarifications']}, indent=2))


if __name__ == '__main__':
    main()
