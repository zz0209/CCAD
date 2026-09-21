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
        if run.name == 'RG03_NUMBER_SOURCE_SMOKE_20260921':
            state = 'INVALID_ARTIFACT_FINALIZATION'
            note = 'Source predictions completed; summary creation raised KeyError for missing component. Original RUNNING record retained. NUMBER_SOURCE_SMOKE_FAILURE.json records the invalid status; the recorded smoke repeats the corrected path.'
        if run.name == 'RG03_NUMBER_TRAJECTORY_CONFIRMATION_T3_20260921':
            note = 'All64 prediction cases completed. An unobserved simple-sentence category produced NaN summaries and finite_results=False. R2 enumerates observed categories; original predictions and FAIL status remain available for exact replay.'
        if run.name == 'CONDITIONAL_PROFILE_DEVELOPMENT_T2_20260920':
            state = 'INTERRUPTED_AFTER_COMPLETED_GLOBAL_CONTROL'
            note = 'All20 ordinary-profile requests completed; owned worker stopped during redundant inactive-token inference. Original status and predictions retained; R2 completes the two cached methods. Stop-time interval is in master_log.'
        row = dict(run_id=run.name, path=str(run), status=state, original_status=status,
            wall_seconds=summary.get('wall_seconds'), cpu_seconds=summary.get('process_cpu_seconds'),
            peak_allocated_bytes=summary.get('peak_allocated_bytes'),
            sequence_forwards=summary.get('sequence_forwards'), token_forwards=summary.get('token_forwards'),
            prediction_rows=summary.get('rows'), source_seed=c.get('source_seed'),
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
    if args.round_id == 'REUSE_GENERALIZATION_01':
        result['metadata_clarifications'] = [
            'All runs are development, with one shared target and exposed function contexts. Evaluation members are excluded from ordinary-action fitting.',
            'Batch-global aggregate and column objectives share actions, text schedule and update budget. State-local sampling changes source action identities and energy.',
            'Writer-only training freezes every natural dictionary parameter. Its action matrix starts from the target encoder and is saved separately.',
            'The fixed-action audit uses paired batch-global aggregate and singleton requests on fit and held contexts for every trained rule.',
            'Checkpoint labels task_adapted_tangent in agreement evaluation denote generic-action training. Agreement responses never enter fitting.',
            'The first active-execution smoke lacks inverse_steps and has FAIL status. Its source snapshot and traceback are retained; SMOKE_R2 completes the path.',
            'Decoder pursuit and fixed-support refinement are standard inference references. The active-only variant restricts candidates without changing final member allowance.',
            'First smoke startup failed before creation of a run directory; no driver duration is inferred for that attempt.'
        ]
    if args.round_id == 'REUSE_GENERALIZATION_02':
        result['metadata_clarifications'] = [
            'Finite fitting changes embedding execution coefficients for the complete55-member explanation. All11natural SAE dictionaries and the language model remain fixed.',
            'The embedding has10source columns,119source-active vocabulary tokens and a20-member shared allowance. Other sites retain the original tangent rule.',
            'Fixed-direct and sparse fitting share parameter coordinates, requests, updates and learning rate. Their64-step outputs agree bitwise; support changes occur later.',
            'Development uses32previously exposed biographies. Frozen confirmation uses128new biographies and27requests, with12new participation requests as primary.',
            'Target2 supplies the fitted physical fields. Targets4/5 receive these fields through constrained geometric realization and use no response fitting.',
            'The bootstrap resamples biographies and participation requests, holding the source, target dictionaries and four later classifier heads fixed.',
            'The first smoke retained predictions but failed in summary generation. Its full duration is unknown. The first support smoke failed on a missing reference path.',
            'The source-only normalization diagnostic uses original fit tokens and does not enter confirmation method selection.',
            'Checkpoint exports are included in their parent run sizes. Panel preparation and summary scripts have separate logs and are excluded from measured model-driver duration.'
        ]
    if args.round_id == 'REUSE_GENERALIZATION_03':
        result['metadata_clarifications'] = [
            'Human confirmation uses64 new biographies,12 new participation requests,7 semantic requests and fixed targets3,4,5. The four later heads and source explanation are fixed.',
            'Number confirmation uses64 new prefixes in the original within-RC and across-RC test files, three named requests and fixed targets3,4,5. Original pair identities are bootstrap clusters.',
            'The58-member number program follows published annotations at10 existing sites. It is a separately defined annotated-member intervention, with all source members and identities retained.',
            'Recorded-action and state-feedback execution retain the source model and source encoders for an additional source pass. All natural target dictionaries remain unchanged.',
            'Native comparisons share the upper allowance of twice the source-member count. Actual changed-member counts differ and are retained in execution diagnostics.',
            'The two location controls use the original development biographies. Frozen confirmation data are excluded from selecting these controls.',
            'The first number-source smoke failed during finalization and has unknown complete driver duration. The last progress is retained without treating it as completion.',
            'Some inherited budget and scope strings retain earlier configurations. Actual source panels, original freeze files, run settings and the current9000-second unit allocation are recorded separately.'
        ]
    if args.round_id == 'REUSE_GENERALIZATION_04':
        result['metadata_clarifications'] = [
            'Number development splits24source-bank contexts and24evaluation contexts within the previously exposed panel.',
            'Number confirmation fixes64fresh official prefixes, three requests and target seeds3/4/5. The primary contrast exchanges equal-cardinality part sets within sites.',
            'The source-action bank requires no target dictionary. Per-target construction uses decoder weights and no target model responses.',
            'Fixed pools contain64candidate identities per nonzero part/site; the original per-state allowance remains twice the source-member count.',
            'Human local-action and propagated-effect bank comparisons use exposed development biographies and retain their distinct source representations.',
            'Human confirmation fixes32new biographies, eight semantic requests, four later heads and target seeds3/4/5 before evaluation. Paired bootstrap resamples profession/gender-stratified documents.',
            'The filled-action control matches propagated-pool cardinality at each singleton part/site. Union candidate counts can differ because selected sets overlap differently.',
            'The original source-bank exporter failed before output because CUDA initialization had not preceded peak-memory reset. R2completed the export.',
            'The initial development swap exchanged empty and nonempty sets at some sites. Formal confirmation matches the site-specific pool cardinalities.',
            'The initial smoke command ended before run creation when the new storage parent did not exist. Its driver duration is unknown.'
        ]
    if args.round_id == 'REUSE_GENERALIZATION_09':
        result['metadata_clarifications'] = [
            'Target whole/program checkpoints were trained against source1 in the preceding study. Additional source definitions2–5 supply their own complete TopK encoders and original192selected members.',
            'All twelve nonself source-target directions among2–5 are retained. Evaluation performs zero optimization updates and checks unchanged source, base-model and loaded target parameters.',
            'The primary comparison is saved part-program minus initial execution on seven requests. Saved complete-request training and both source-direction readouts are retained.',
            'The384confirmation sentence pairs are new draws from three fixed original grammar generators, excluding earlier retained sentences including the preceding384-pair confirmation.',
            'Inference resamples paired sentences within grammar, holding the four shared seed nodes fixed. Deleting one node removes every direction incident on that node.',
            'The first smoke launches failed before a model run due to resource-lease access and then a missing storage parent. Valid smoke, development and confirmation outputs retain separate identities.',
            'The initial grammar generation stopped after16saved examples. The local adapter rejects missing opposite-number verb forms according to the original accepted-pair logic; regenerated data have a separate version and pre-evaluation freeze.',
            'Each evaluation driver computes source-only normalization on old fitting examples. Target response fitting is absent. Run duration includes loading and all five methods plus source and unedited references.'
        ]
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['runs', 'metadata_clarifications']}, indent=2))


if __name__ == '__main__':
    main()
