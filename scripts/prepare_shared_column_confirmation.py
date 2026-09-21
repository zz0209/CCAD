from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

from prepare_source_column_reuse import save


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round04'
BULK = Path('D:/CCAD_Storage/runs/final_science_20260921_round04')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', choices=['human', 'infinitive'])
    setting = parser.parse_args().setting
    fit = BULK/f'FS04_{setting.upper()}_SHARED_COLUMNS_FIT_20260921'
    assert json.loads((fit/'status.json').read_text())['status'] == 'PASS'
    bank = BULK/(fit.name+'_checkpoints') if setting == 'human' else fit
    columns = {name: str(bank/variant/'source_columns.npz') for name, variant in
               [('source_columns_shared', 'source_columns_mixed'), ('source_columns_shared_gain', 'source_columns_scalar_mixed')]}
    held = [1, 4, 5] if setting == 'human' else [1, 2, 5]
    paths = []
    for seed in held:
        base = 'science04_shift_t4_v1.json' if setting == 'human' else 'science04_infinitive_t3_v1.json'
        c = json.loads((ROOT/'configs'/base).read_text())
        rid = f'FS04_{setting.upper()}_SHARED_CONFIRM_T{seed}_20260921'
        c.update(run_id=rid, run_parent='FINAL_SCIENCE_04', target_seed=seed, seeds=[seed],
                 run_storage_root=str(BULK), bulk_output_dir=str(BULK/(rid+'_checkpoints')),
                 purpose='Evaluate a frozen shared source correction on dictionaries excluded from its fit, using new contexts and continuous requests.',
                 evidence_level='frozen_new_context_request_confirmation', audit_opened=True,
                 evaluate_source_columns=columns, source_columns_evaluation_only=False,
                 variants=[], budget_seconds=900,
                 budget='At most900driver seconds per target including any required512-update reference arms;13GB allocated VRAM ceiling.',
                 scope='Fixed source and two fit dictionaries; three held target seeds. Shared corrections use zero target fitting. Target-specific controls retain512source-supervised updates.',
                 statistics_unit='Paired held-target seeds, documents or crossed lexical items, and requests. Source explanations and fit dictionaries remain fixed.')
        if seed == 1:
            c['target_directory'] = 'D:/CCAD_Storage/training_curves/REFORM_R58_shift_dictionaries_curve_v1_20260915/step_8192'
        if setting == 'human':
            c['relation_run'] = ('runs/REFORM_R58_shift_context_curve_s8192_v1_20260915' if seed == 1
                                 else f'runs/REFORM_R59_shift_confirm_seed{seed}_v1_20260915')
            req_path = OUT/'SHARED_COLUMN_HUMAN_CONFIRMATION_REQUESTS.json'
            req = json.loads(req_path.read_text())
            c.update(queries=req['queries'], dose_queries=req['dose_queries'], member_queries=req['member_queries'],
                     request_panel=str(req_path), request_panel_sha256=hashlib.sha256(req_path.read_bytes()).hexdigest(),
                     evaluation_panel=str(OUT/'SHARED_COLUMN_HUMAN_CONFIRMATION_PANEL.json'),
                     evaluation_description='128 unused original-dev biographies, excluded from prior panel metadata.',
                     eval_batch_size=64, eval_token_budget=8192,
                     evaluate_baselines=['input_initial', 'raw_reconstruction'])
            if seed == 1:
                c['variants'] = ['tangent_gain', 'tangent_mixed']
            else:
                c['evaluate_checkpoints'] = {m: f'D:/CCAD_Storage/training_curves/SCIENCE04_shift_t{seed}_v1_20260919/{m}'
                                             for m in ['tangent_gain', 'tangent_mixed']}
        else:
            c['relation_run'] = ('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE04_infinitive_seed1_material_v1_20260919' if seed == 1
                                 else 'runs/IR01_infinitive_natural_dev_v1_20260916' if seed == 2
                                 else f'runs/IR01_infinitive_target{seed}_v1_20260916')
            if seed == 5:
                c['relation_run'] = json.loads((ROOT/'configs/science04_infinitive_t5_v1.json').read_text())['relation_run']
            c.update(panel=str(OUT/'SHARED_COLUMN_INFINITIVE_CONFIRMATION_PANEL.json'),
                     eval_batch_size=64,
                     baselines=['native', 'native_tangent_relation_8', 'raw_reconstruction'])
            if seed == 2:
                c['variants'] = ['tangent_gain', 'tangent_mixed']
            else:
                version = 2 if seed == 1 else 1
                c['evaluate_checkpoints'] = {m: f'D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE04_infinitive_t{seed}_v{version}_20260919/{m}'
                                             for m in ['tangent_gain', 'tangent_mixed']}
        assert (Path(c['relation_run'])/'relation.npz').is_file()
        for path in c.get('evaluate_checkpoints', {}).values():
            assert Path(path).is_dir(), path
        path = ROOT/f'configs/fs04_{setting}_shared_confirm_t{seed}.json'
        save(path, c)
        paths.append(path)
    source_paths = [ROOT/'scripts/train_shift_response_space.py', ROOT/'scripts/train_infinitive_program.py',
                    ROOT/'scripts/prepare_shared_column_confirmation.py', ROOT/'scripts/analyze_shared_column_confirmation.py',
                    ROOT/'scripts/analyze_source_column_transfer.py', ROOT/'scripts/analyze_science04_confirmation.py',
                    ROOT/'scripts/run_shift_transfer.py', ROOT/'scripts/run_shift_explanation.py',
                    OUT/'SHARED_COLUMN_PANEL_PROVENANCE.json']
    provenance=json.loads((OUT/'SHARED_COLUMN_PANEL_PROVENANCE.json').read_text())
    source_paths += [Path(p) for p in provenance['files']]
    save(OUT/f'SHARED_COLUMN_{setting.upper()}_CONFIRMATION_FREEZE.json', dict(
        written_at_utc=datetime.now(timezone.utc).isoformat(), setting=setting, held_targets=held,
        fit_targets=[2, 3] if setting == 'human' else [3, 4],
        primary='Mean per-request normalized finite-response error on twelve fresh continuous semantic requests, comparing shared columns with shared scalar correction and the initial input-dependent rule.',
        secondary='Exact semantic parts, finer member requests, source-direction reconstruction readout and target-specific program/gain references. Human fixed later-head predictions and task accuracy remain separate outcomes.',
        inference='2000 paired held-target and document/request draws; human profession/gender strata and four fixed later heads, grammar crossed verbs/nouns and fixed exhaustive binary masks.',
        selection='Checkpoints and comparisons fixed before these model outcomes. No configuration selection on confirmation.',
        data_provenance=str(OUT/'SHARED_COLUMN_PANEL_PROVENANCE.json'), configs=list(map(str, paths)),
        identities={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+source_paths+[Path(p) for p in columns.values()]}))
    print(json.dumps(dict(setting=setting, held_targets=held, fit_targets=[2, 3] if setting == 'human' else [3, 4])))


if __name__ == '__main__':
    main()
