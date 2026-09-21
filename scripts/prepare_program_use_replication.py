import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/reuse_generalization_20260921_round07'


def identity(path):
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    base = ROOT / 'configs/final_science03_consumer_development.json'
    template = json.loads(base.read_text())
    outputs, checkpoints = [], []
    for target in [3, 4, 5]:
        cfg = copy.deepcopy(template)
        cfg.update(run_id=f'RG07_PROGRAM_USE_T{target}_20260921',
                   run_parent='REUSE_GENERALIZATION_07', target_seed=target, seeds=[target],
                   run_storage_root='D:/CCAD_Storage/runs/reuse_generalization_20260921_round07',
                   relation_run=f'runs/REFORM_R59_shift_confirm_seed{target}_v1_20260915',
                   feature_cache_runs=[f'runs/IR04_shift_consumer_seed{target}_v1_20260916'],
                   purpose='Evaluate the same frozen input-dependent program on later classifier use across additional target dictionaries',
                   scope='Target replication on the existing consumer cohort; methods and classifier recipe unchanged',
                   evidence_level='development_existing_consumer_target_replication',
                   statistics_unit='Joint target and profession/gender-stratified document resampling, fixed source and task pairs',
                   budget_seconds=1500,
                   budget='Three runs at most 4500 driver seconds; combined artifacts at most 4GB')
        for name, variant in [('input_gain', 'tangent_gain'), ('input_program', 'tangent_mixed')]:
            folder = Path(f'D:/CCAD_Storage/training_curves/SCIENCE04_shift_t{target}_v1_20260919/{variant}')
            cfg['adapted_input_programs'][name]['directory'] = folder.as_posix()
            for site in cfg['adapted_input_programs'][name]['sites']:
                path = folder / f'{site}_seed{target}.pt'
                assert path.is_file(), path
                checkpoints.append(identity(path))
            if variant == 'tangent_gain':
                checkpoints.append(identity(folder / 'source_gains.npz'))
        path = ROOT / f'configs/rg07_program_use_t{target}.json'
        assert not path.exists(), path
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        outputs.append(identity(path))
        if target == 3:
            smoke = copy.deepcopy(cfg)
            panel = 'artifacts/final_science_20260920_round03/CONSUMER_SMOKE_PANEL.json'
            smoke.update(run_id='RG07_PROGRAM_USE_SMOKE_T3_20260921',
                         training_panel=panel, evaluation_panel=panel, feature_cache_runs=[],
                         methods=['none', 'source', 'input_tangent_budget', 'input_gain', 'input_program'],
                         budget_seconds=180)
            path = ROOT / 'configs/rg07_program_use_smoke.json'
            assert not path.exists(), path
            path.write_text(json.dumps(smoke, indent=2) + '\n')
            outputs.append(identity(path))
    design = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                  question='Does the same learned member execution preserve later explanation use across target dictionaries?',
                  primary='Mean part accuracy difference between input_program and input_tangent_budget; input_gain is the matched simpler training comparison.',
                  secondary='Worst-group accuracy, full accuracy, fixed-head response fidelity, and source-relative pronoun/name distinction.',
                  analysis='Report additional targets 3-5 separately from the pooled 2-5 result; 2000 paired target/document bootstrap draws, fixed task pairs and head seed 42.',
                  consequence='Use one trained C(h)q construction for the main method, its two published explanation tests and its later-use experiment. Keep comparative execution mechanisms in the supplement.',
                  scope='Existing exposed cohort. No independent-document or new-operation-family claim.',
                  source_config=identity(base), configurations=outputs, checkpoints=checkpoints,
                  panel=identity(ROOT / template['training_panel']),
                  runner=identity(ROOT / 'scripts/run_shift_retraining.py'),
                  analyzer=identity(ROOT / 'scripts/shift_new_tasks_results.py'))
    path = ART / 'DESIGN.json'
    assert not path.exists(), path
    path.write_text(json.dumps(design, indent=2) + '\n')
    print(json.dumps(dict(configurations=outputs, checkpoint_count=len(checkpoints))))


if __name__ == '__main__':
    main()
