import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DESCRIPTION = ('Run P2 formal-study step 1 (protocol: artifacts/newtopics/p2/STEP1_PROTOCOL.md) on the local GPU: '
               'timing run, J-lens fit of Pythia-2.8B, converged tuned lens (Muon) of OLMo-2-0425-1B main and '
               'Pythia-2.8B, development-set swaps with J, tuned, tuned_nm, logit and logit_nm, and the step-1 '
               'decision. Each stage is skipped when its output exists, so the run resumes after interruption. Meant '
               'to be started through the shared resource manager with the gpu-0 lease.')
HERE = Path(__file__).resolve().parent
METHODS = ['jlens', 'tuned', 'tuned_nm', 'logit', 'logit_nm']


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def stamp(log, msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(log, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def set_status(run_dir, **fields):
    # status.json and progress.json follow tmp/progress_viewer/server.py.
    path = run_dir / 'status.json'
    st = json.loads(path.read_text()) if path.exists() else {}
    st.update(fields)
    path.write_text(json.dumps(st, indent=1))


def stage(log, name, out, cmd, stdout_path):
    if out is not None and Path(out).exists():
        stamp(log, f'SKIP {name} ({out} exists)')
        return
    stamp(log, f'START {name}')
    (log.parent / 'progress.json').write_text(json.dumps(dict(stage=name)))
    t0 = time.time()
    with open(stdout_path, 'w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        stamp(log, f'FAIL {name} rc={rc} after {time.time() - t0:.0f}s, see {stdout_path}')
        set_status(log.parent, status='FAIL', ended_at_utc=now())
        raise SystemExit(rc)
    stamp(log, f'DONE {name} {time.time() - t0:.0f}s')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--fineweb', required=True)
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--olmo_jlens', required=True, help='the OLMo-1B main J-lens fitted in P2_WORKSPACE_v1')
    parser.add_argument('--arrays', type=Path, required=True, help='folder for lenses and checkpoints (D:)')
    parser.add_argument('--light', type=Path, required=True, help='folder for JSON outputs and logs (E:)')
    parser.add_argument('--gpu_cap_gib', type=float, default=14.0)
    args = parser.parse_args()
    args.arrays.mkdir(parents=True, exist_ok=True)
    args.light.mkdir(parents=True, exist_ok=True)
    log = args.light / 'progress.log'
    set_status(args.light, status='RUNNING', started_at_utc=now(), ended_at_utc=None)
    py = sys.executable
    common = ['--manifest', args.manifest]
    cap = ['--gpu_cap_gib', str(args.gpu_cap_gib)]
    stage(log, 'timing_jfit_pythia28b', None,
          [py, 'bench_jfit.py', *common, '--label', 'pythia28b', '--fineweb', args.fineweb, '--dim_batch', '8', *cap],
          args.light / 'timing_jfit_pythia28b.log')
    stage(log, 'jlens_pythia28b', args.arrays / 'pythia28b_jlens.pt',
          [py, 'fit_jlens.py', *common, '--label', 'pythia28b', '--fineweb', args.fineweb,
           '--out', str(args.arrays / 'pythia28b_jlens.pt'), '--dim_batch', '8', *cap],
          args.light / 'fit_jlens_pythia28b.log')
    jl = {'olmo1b_main': args.olmo_jlens, 'pythia28b': str(args.arrays / 'pythia28b_jlens.pt')}
    for label in ('olmo1b_main', 'pythia28b'):
        tl = args.arrays / f'{label}_tl_full.pt'
        stage(log, f'tl_full_{label}', tl,
              [py, 'fit_tuned_muon.py', *common, '--label', label, '--fineweb', args.fineweb, '--parquet', args.parquet,
               '--out', str(tl), *cap],
              args.light / f'fit_tuned_muon_{label}.log')
        stage(log, f'swap_{label}', args.light / f'{label}_swap.json',
              [py, 'swap_eval.py', *common, '--label', label, '--task', args.task, '--jlens', jl[label],
               '--tuned', str(tl), '--out', str(args.light / f'{label}_swap.json'), '--methods', *METHODS, *cap],
              args.light / f'swap_eval_{label}.log')
    stage(log, 'step1_analysis', args.light / 'STEP1_DECISION.json',
          [py, 'step1_analysis.py', '--runs', f"olmo1b_main={args.light / 'olmo1b_main_swap.json'}",
           f"pythia28b={args.light / 'pythia28b_swap.json'}", '--out', str(args.light / 'STEP1_DECISION.json')],
          args.light / 'step1_analysis.log')
    stamp(log, 'STEP1_DONE ' + json.loads((args.light / 'STEP1_DECISION.json').read_text())['decision']['verdict'])
    set_status(args.light, status='PASS', ended_at_utc=now())


if __name__ == '__main__':
    main()
