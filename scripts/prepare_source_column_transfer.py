from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np

from prepare_source_column_reuse import save


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/final_science_20260921_round04'
BULK = Path('D:/CCAD_Storage/runs/final_science_20260921_round04')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', choices=['human', 'infinitive'])
    args = parser.parse_args()
    setting = args.setting
    learned_run = BULK / f'FS04_{setting.upper()}_SOURCE_COLUMNS_FIT_20260921'
    assert json.loads((learned_run/'status.json').read_text())['status'] == 'PASS'
    if setting == 'human':
        learned = BULK / 'FS04_HUMAN_SOURCE_COLUMNS_FIT_20260921_checkpoints/source_columns_mixed/source_columns.npz'
        base = json.loads((ROOT/'configs/science04_shift_t2_v1.json').read_text())
        parameters = np.load(base['source_parameters'])
        sites = json.loads(Path(base['source_manifest']).read_text())['members']
        gains = np.load('D:/CCAD_Storage/training_curves/SCIENCE04_shift_t2_v1_20260919/tangent_gain/source_gains.npz')
        gain_columns = {s: parameters[s+'__decoder']*gains[s][:, None] for s in sites}
        seeds = [3, 4, 5]
    else:
        learned = learned_run/'source_columns_mixed/source_columns.npz'
        base = json.loads((ROOT/'configs/science04_infinitive_t3_v1.json').read_text())
        parameters = np.load(base['source_parameters'])
        gains = np.load('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE04_infinitive_t3_v1_20260919/tangent_gain/source_gains.npy')
        gain_columns = dict(decoder=parameters['decoder']*gains[:, None])
        seeds = [1, 4, 5]
    control = OUT/f'{setting.upper()}_TRANSFERRED_GAIN_COLUMNS.npz'
    if control.exists():
        raise FileExistsError(control)
    np.savez_compressed(control, **gain_columns)
    configurations = []
    for seed in seeds:
        name = f'science04_shift_t{seed}_v1' if setting == 'human' else f'science04_infinitive_t{seed}_v{2 if seed == 1 else 1}'
        c = json.loads((ROOT/'configs'/f'{name}.json').read_text())
        rid = f'FS04_{setting.upper()}_SOURCE_COLUMNS_TRANSFER_T{seed}_20260921'
        c.update(run_id=rid, run_parent='FINAL_SCIENCE_04', variants=[],
                 run_storage_root=str(BULK), bulk_output_dir=str(BULK/(rid+'_checkpoints')),
                 source_columns_evaluation_only=True,
                 evaluate_source_columns={'source_columns_mixed': str(learned), 'source_columns_gain': str(control)},
                 purpose='Apply one frozen source-column correction to an independently trained target dictionary without new fitting.',
                 evidence_level='development', audit_opened=False, budget_seconds=900,
                 scope='Exposed contexts and requests; source-column checkpoint is fixed and learned on a different target seed.',
                 evaluate_baselines=['input_initial','raw_reconstruction'] if setting == 'human' else [],
                 baselines=['native','native_tangent_relation_8','raw_reconstruction'] if setting == 'infinitive' else [])
        path = ROOT/f'configs/fs04_{setting}_source_columns_transfer_t{seed}.json'
        save(path, c)
        configurations.append(str(path))
    save(OUT/f'{setting.upper()}_TRANSFER_DESIGN.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
         source_columns=str(learned), source_columns_sha256=hashlib.sha256(learned.read_bytes()).hexdigest(),
         transferred_gain=str(control), gain_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
         configs=configurations, target_fitting=False, evidence='Development on exposed panels'))


if __name__ == '__main__':
    main()
