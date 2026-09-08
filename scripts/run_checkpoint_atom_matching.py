"""Use the existing full-dictionary Hungarian implementation at fixed checkpoints."""
import argparse
import itertools
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import numpy as np
import scipy
from safetensors.numpy import load_file
from prepare_f4_global_matching import full_assignment
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.artifacts import sha256, validate_run_directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    run = ROOT/'runs'/cfg['run_id']
    run.mkdir(exist_ok=False)
    begin = time.perf_counter()
    cpu_begin = time.process_time()
    write(run/'config.resolved.json', cfg)
    inputs = [entry(args.config, 'fixed consumer configuration', 'configuration')]
    code = []
    for rel in ['scripts/run_checkpoint_atom_matching.py', 'scripts/prepare_f4_global_matching.py',
                'scripts/run_r011s1_raw_hook_asset.py', 'scripts/run_f4_source_reference_causal.py', 'src/ccad/artifacts.py']:
        p = ROOT/rel
        destination = run/'source_snapshot'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
        code.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha256(p), snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json', dict(files=code, aggregate_sha256=aggregate(code), snapshot_root='source_snapshot'))
    write(run/'manifest.json', dict(schema_version='checkpoint.atom.match.v1', run_id=cfg['run_id'],
          run_parent='FINAL_FIVE_R14', purpose=cfg['purpose'], milestone='controlled_atom_training_curve',
          evidence_level='controlled_development', started_utc=datetime.now(timezone.utc).isoformat(),
          project_root=str(ROOT), config_hash=sha256(run/'config.resolved.json'), code_snapshot_hash=aggregate(code),
          source_snapshot_required=True, audit_opened=False, candidate_family_frozen=True,
          mean_constants_source_split='not used: normalized decoder cosine',
          threshold_source_split='all atoms and all unordered seed pairs, no activity filtering',
          statistics_unit='five shared SAE seeds and dependent checkpoints; seed pairs are not independent',
          device='cpu', seeds=cfg['seeds'], resource_lease='cpu-heavy', resource_lease_reason=cfg['budget']))
    write(run/'environment.json', dict(python=sys.executable, python_version=platform.python_version(),
          numpy=np.__version__, scipy=scipy.__version__, compute_threads=4))
    for name in ['stdout.log', 'stderr.log', 'metrics.raw.jsonl']:
        (run/name).touch()
    write(run/'status.json', dict(status='RUNNING'))
    rows = []
    error = None
    try:
        for stage in cfg['stages']:
            weights = {}
            for seed in cfg['seeds']:
                path = Path(stage['checkpoint_root'])/f'seed_{seed}'/'sae.safetensors'
                inputs.append(entry(path, 'controlled training checkpoint', 'complete decoder'))
                weights[seed] = load_file(str(path))['W_dec'].astype(np.float64)
                if weights[seed].shape != (8192, 2048):
                    raise ValueError('Unexpected decoder shape')
            write(run/'inputs.json', dict(inputs=inputs))
            for source, target in itertools.combinations(cfg['seeds'], 2):
                if time.perf_counter()-begin > cfg['budget_seconds']:
                    raise TimeoutError('Declared matching budget exhausted; completed pairs retained')
                start_pair = time.perf_counter()
                mapping, scale, stats = full_assignment(weights[source], weights[target])
                # The reusable implementation includes all atoms, absolute cosine,
                # a one-to-one assignment, and signed physical unit conversion.
                filename = f"step{stage['step']}_s{source}_t{target}_assignment.npz"
                np.savez_compressed(run/filename, target_indices=mapping, geometric_scale=scale)
                row = dict(run_id=cfg['run_id'], metric_version='all_atom_hungarian_cosine_v1',
                           step=stage['step'], tokens=stage['step']*1024, source_seed=source, target_seed=target,
                           width=8192, assignment=filename, wall_seconds=time.perf_counter()-start_pair, **stats)
                rows.append(row)
                with (run/'metrics.raw.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                write(run/'progress.json', dict(stage='PAIR_COMPLETE', written_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed=time.perf_counter()-begin, completed_pairs=len(rows), latest=row))
                print(json.dumps(row), flush=True)
                if len(rows)==1 and row['wall_seconds']>cfg['first_pair_budget_seconds']:
                    raise TimeoutError('First full assignment exceeded throughput budget; result retained')
            del weights
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (run/'stderr.log').write_text(traceback.format_exc())
    checks = dict(all_pairs=len(rows)==len(cfg['stages'])*len(list(itertools.combinations(cfg['seeds'],2))),
                  finite=all(np.isfinite(r['mean_absolute_cosine']) for r in rows),
                  unique=len(rows)==len({(r['step'],r['source_seed'],r['target_seed']) for r in rows}))
    status = 'PASS' if error is None and all(checks.values()) else 'FAIL'
    summary = dict(status=status, error=error, rows=len(rows), checks=checks, wall_seconds=time.perf_counter()-begin,
                   process_cpu_seconds=time.process_time()-cpu_begin, scope=cfg['scope'],
                   metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'))
    write(run/'inputs.json', dict(inputs=inputs))
    write(run/'metrics.summary.json', summary)
    write(run/'stdout.log', summary)
    write(run/'status.json', dict(status=status, error=error, updated_utc=datetime.now(timezone.utc).isoformat()))
    validation = validate_run_directory(run)
    write(run/'contract_validation.json', dict(ok=validation.ok, errors=list(validation.errors)))
    print(json.dumps(dict(**summary, contract_ok=validation.ok)), flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__':
    raise SystemExit(main())
