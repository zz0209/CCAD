"""Preserve the current manuscript and prepare fixed training-curve consumers."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    now = dt.datetime.now(dt.timezone.utc)
    archive = ROOT/'archive/research_workflow_20260908'/('r14_paper_'+now.strftime('%Y%m%dT%H%M%SZ'))
    entries = []
    paths = [p for p in (ROOT/'paper').rglob('*') if p.is_file() and 'build' not in p.relative_to(ROOT/'paper').parts]
    paths += [ROOT/p for p in ['EXPERIMENT_TRACKER.md', 'EXPERIMENT_PLAN.md', 'REFERENCE_REGISTRY.md']]
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = archive/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        a = hashlib.sha256(path.read_bytes()).hexdigest()
        assert a == hashlib.sha256(destination.read_bytes()).hexdigest()
        entries.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=a))
    (archive/'manifest.json').write_text(json.dumps(dict(written_at_utc=now.isoformat(), files=entries), indent=2)+'\n')
    def save(name, cfg):
        path = ROOT/'configs'/name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(cfg, indent=2)+'\n')
    parent = 'FINAL5_R14_l15_continue32m_five_v1_20260908'
    cfg = json.loads((ROOT/'configs/ext_r10_continued_source_function_v1.json').read_text())
    cfg.update(run_id='FINAL5_R14_continued_source_function_v1_20260908', training_run='runs/'+parent,
               checkpoint_steps=[8192, 16384, 24576, 32768], budget_seconds=600,
               purpose='Fixed source-selection rule at 8M,16M,24M,32M for five controlled dictionaries',
               budget='At most 600 seconds GPU-managed wall after the training producer completes; no new training or task panel.',
               scope='Same exposed 384 temporal prompts and frozen source-only raw-path gradient field; members are reselected by exactly the same rule at each checkpoint. This measures source usefulness, not fixed-ID stability or convergence. Training quality and fixed-teacher target transfer are separate consumers.')
    save('final5_r14_continued_source_function_v1.json', cfg)
    for step in [16384, 24576, 32768]:
        label = str(step//1024)+'m'
        specs = [dict(seed=s, path=f'D:/CCAD_Storage/training_curves/{parent}/step_{step}/seed_{s}') for s in range(1,6)]
        cfg = json.loads((ROOT/'configs/ext_r10_continued_codes_v1.json').read_text())
        cfg.update(run_id=f'FINAL5_R14_codes_{label}_v1_20260908', sae_checkpoints=specs,
                   purpose=f'Encode identical historical contexts at fixed {label} target training budget',
                   scope='Historical development and exposed role panel; unchanged 4M source teachers. Only the target checkpoint budget changes. Not independent confirmation.')
        save(f'final5_r14_codes_{label}_v1.json', cfg)
        cfg = json.loads((ROOT/'configs/ext_r10_target8m_support_v1.json').read_text())
        cfg.update(run_id=f'FINAL5_R14_target{label}_support_v1_20260908', target_sae_checkpoints=specs,
                   target_code_run=f'runs/FINAL5_R14_codes_{label}_v1_20260908',
                   purpose=f'Fixed-source correspondence and native controls at target {label}',
                   scope='Same exposed historical inputs, original 4M source teachers, selection method and coefficient budgets. Target training checkpoint is the sole material replacement relative to the retained 8M comparison; training itself jointly adds fresh natural tokens and a declared learning-rate phase. No claim of new confirmation or convergence.')
        save(f'final5_r14_target{label}_support_v1.json', cfg)
    print(json.dumps(dict(archive=str(archive/'manifest.json'), preserved_files=len(entries), consumer_configs=7)))


if __name__ == '__main__':
    main()
