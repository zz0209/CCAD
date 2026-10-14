import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_step1 import now, set_status, stamp

DESCRIPTION = ('Run one phase of P2 formal-study step 2 (protocol: artifacts/newtopics/p2/STEP2_PROTOCOL.md) on the '
               'local GPU. Phase 1: the matched tuned lens and v1/v2 lens swaps at OLMo-1B main and Pythia-2.8B, the '
               'v2 stop rules (the run halts if one triggers), then per OLMo-1B intermediate checkpoint an eligibility '
               'count and, when n >= 20 on v1 or v2, J-lens fit, converged and matched tuned lens and v1/v2 swaps, then '
               'the lens-fit stability refit at OLMo-1B main. Phase 2: Pythia-410M (converged and matched tuned lens, '
               'swaps, stability refit) and Pythia-1.4B (J-lens fit, both tuned lenses, swaps). Phase 3: concept-vector '
               'swaps (C4) on the held-out relation pairs of v1 and v2. Each phase ends with step2_analysis.py. Every '
               'stage is skipped when its output exists, so a phase resumes after interruption. Meant to be started '
               'through the shared resource manager with the gpu-0 lease.')
HERE = Path(__file__).resolve().parent
OLMO_MAIN = 'olmo1b_main'
OLMO_INTER = [f'olmo1b_step{s}' for s in (10000, 20000, 50000, 100000, 200000, 430000, 900000, 1907359)]
FINALS = [OLMO_MAIN, 'pythia410m', 'pythia14b', 'pythia28b']
LENS_METHODS = ['jlens', 'logit', 'tuned', 'tmatch', 'logit_nm', 'tuned_nm', 'tmatch_nm']
C4_METHODS = ['dim', 'dim_outside_j', 'dim_outside_j_clamp', 'random']
HELD_OUT = ['language_to_continent', 'currency_to_capital']
H5_OFFSET = 3800
MIN_N = 20


class Runner:
    def __init__(self, args):
        self.a = args
        self.run_dir = args.light / f'phase{args.phase}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = self.run_dir / 'progress.log'
        self.done = 0
        self.py = sys.executable
        self.common = ['--manifest', str(args.manifest)]
        self.cap = ['--gpu_cap_gib', str(args.gpu_cap_gib)]
        self.tasks = dict(v1=str(args.task_v1), v2=str(args.task_v2))

    def stage(self, label, name, out, cmd):
        # One subprocess per stage; output is kept per checkpoint, and an existing output skips the stage.
        light = self.a.light / label
        light.mkdir(parents=True, exist_ok=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if Path(out).exists():
            stamp(self.log, f'SKIP {label} {name} ({out} exists)')
            return
        stamp(self.log, f'START {label} {name}')
        (self.run_dir / 'progress.json').write_text(json.dumps(dict(stage=f'{label} {name}',
                                                                    step=f'{self.done} stages done')))
        t0 = time.time()
        stdout_path = light / f'{name}.log'
        with open(stdout_path, 'w', encoding='utf-8') as f:
            rc = subprocess.run([self.py, *cmd], cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            stamp(self.log, f'FAIL {label} {name} rc={rc} after {time.time() - t0:.0f}s, see {stdout_path}')
            set_status(self.run_dir, status='FAIL', ended_at_utc=now())
            raise SystemExit(rc)
        self.done += 1
        stamp(self.log, f'DONE {label} {name} {time.time() - t0:.0f}s')

    def jlens_path(self, label):
        given = dict(olmo1b_main=self.a.olmo_main_jlens, pythia410m=self.a.pythia410m_jlens,
                     pythia28b=self.a.pythia28b_jlens)
        return str(given.get(label) or self.a.arrays / label / 'jlens.pt')

    def tl_full_path(self, label):
        given = dict(olmo1b_main=self.a.olmo_main_tl_full, pythia28b=self.a.pythia28b_tl_full)
        return str(given.get(label) or self.a.arrays / label / 'tl_full.pt')

    def jfit(self, label, offset=0):
        out = self.a.arrays / label / ('jlens.pt' if offset == 0 else 'jlens_h5.pt')
        self.stage(label, 'fit_jlens' if offset == 0 else 'fit_jlens_h5', out,
                   ['fit_jlens.py', *self.common, '--label', label, '--fineweb', str(self.a.fineweb), '--out', str(out),
                    '--doc_offset', str(offset), '--dim_batch', '8', *self.cap])

    def tl_full(self, label):
        out = self.tl_full_path(label)
        self.stage(label, 'tl_full', out,
                   ['fit_tuned_muon.py', *self.common, '--label', label, '--fineweb', str(self.a.fineweb), '--parquet',
                    str(self.a.parquet), '--out', out, '--train_data', 'parquet', '--max_steps', '1024', *self.cap])

    def tl_match(self, label):
        out = str(self.a.arrays / label / 'tl_match.pt')
        self.stage(label, 'tl_match', out,
                   ['fit_tuned_muon.py', *self.common, '--label', label, '--fineweb', str(self.a.fineweb), '--parquet',
                    str(self.a.parquet), '--out', out, '--train_data', 'jdocs', '--max_steps', '250', *self.cap])

    def swaps(self, label, kind):
        for s, task in self.tasks.items():
            base = ['swap_eval.py', *self.common, '--label', label, '--task', task, '--tuned', self.tl_full_path(label)]
            if kind == 'lens':
                out = self.a.light / label / f'swap_{s}.json'
                extra = ['--jlens', self.jlens_path(label), '--tuned_match', str(self.a.arrays / label / 'tl_match.pt'),
                         '--methods', *LENS_METHODS]
            elif kind == 'h5':
                out = self.a.light / label / f'h5_swap_{s}.json'
                extra = ['--jlens', str(self.a.arrays / label / 'jlens_h5.pt'), '--methods', 'jlens']
            else:
                out = self.a.light / label / f'c4_{s}.json'
                extra = ['--jlens', self.jlens_path(label), '--concept_fit', str(self.a.concept_fit),
                         '--pairs', *HELD_OUT, '--methods', *C4_METHODS]
            self.stage(label, f'{kind}_swap_{s}', out, [*base, *extra, '--out', str(out), *self.cap])

    def eligibility(self, label):
        out = self.a.light / label / 'eligibility.json'
        self.stage(label, 'eligibility', out,
                   ['eligibility.py', *self.common, '--label', label, '--tasks',
                    *[f'{s}={t}' for s, t in self.tasks.items()], '--out', str(out), *self.cap])
        return json.loads(out.read_text())

    def analysis(self, mode, name):
        out = self.run_dir / name
        self.stage(f'phase{self.a.phase}', f'analysis_{mode}', out,
                   ['step2_analysis.py', '--root', str(self.a.light), '--mode', mode, '--out', str(out)])
        return json.loads(out.read_text())


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--manifest', type=Path, required=True, help='frozen models_manifest.json of step 2')
    parser.add_argument('--fineweb', type=Path, required=True)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--task_v1', type=Path, required=True)
    parser.add_argument('--task_v2', type=Path, required=True)
    parser.add_argument('--concept_fit', type=Path, required=True)
    parser.add_argument('--olmo_main_jlens', type=Path, required=True)
    parser.add_argument('--olmo_main_tl_full', type=Path, required=True)
    parser.add_argument('--pythia410m_jlens', type=Path, required=True)
    parser.add_argument('--pythia28b_jlens', type=Path, required=True)
    parser.add_argument('--pythia28b_tl_full', type=Path, required=True)
    parser.add_argument('--arrays', type=Path, required=True, help='lens arrays and checkpoints (D:)')
    parser.add_argument('--light', type=Path, required=True, help='JSON outputs and logs (E:)')
    parser.add_argument('--gpu_cap_gib', type=float, default=14.0)
    args = parser.parse_args()
    r = Runner(args)
    set_status(r.run_dir, status='RUNNING', started_at_utc=now(), ended_at_utc=None)
    if args.phase == 1:
        for label in (OLMO_MAIN, 'pythia28b'):
            r.tl_match(label)
            r.swaps(label, 'lens')
        stops = r.analysis('stopcheck', 'STOP_CHECK.json')['stop_rules']
        if stops['whole_study_stop'] or stops['lens_comparison_stop']:
            stamp(r.log, f'STOP_RULE_TRIGGERED {json.dumps(stops)}; phase halted for the coordinating session')
            set_status(r.run_dir, status='PASS', ended_at_utc=now())
            return
        for label in OLMO_INTER:
            e = r.eligibility(label)
            if not e['eligible_any']:
                stamp(r.log, f'INELIGIBLE {label}: ' + ', '.join(f"{s} {v['n_correct']}" for s, v in e['sets'].items()))
                continue
            r.jfit(label)
            r.tl_full(label)
            r.tl_match(label)
            r.swaps(label, 'lens')
        r.jfit(OLMO_MAIN, H5_OFFSET)
        r.swaps(OLMO_MAIN, 'h5')
    elif args.phase == 2:
        r.tl_full('pythia410m')
        r.tl_match('pythia410m')
        r.swaps('pythia410m', 'lens')
        r.jfit('pythia410m', H5_OFFSET)
        r.swaps('pythia410m', 'h5')
        r.jfit('pythia14b')
        r.tl_full('pythia14b')
        r.tl_match('pythia14b')
        r.swaps('pythia14b', 'lens')
    else:
        for label in FINALS + OLMO_INTER:
            if label in OLMO_INTER:
                e = json.loads((args.light / label / 'eligibility.json').read_text())
                if not any(v['n_correct_held_out'] >= MIN_N for v in e['sets'].values()):
                    stamp(r.log, f'SKIP {label} c4: held-out n < {MIN_N} on both sets')
                    continue
            r.swaps(label, 'c4')
    res = r.analysis('full', f'STEP2_DECISION_phase{args.phase}.json')
    stamp(r.log, f'PHASE{args.phase}_DONE ' + json.dumps({k: res[k]['verdict'] for k in ('H1', 'H2', 'H3', 'H4')}))
    set_status(r.run_dir, status='PASS', ended_at_utc=now())


if __name__ == '__main__':
    main()
