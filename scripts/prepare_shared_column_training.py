from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import argparse

from prepare_source_column_reuse import save


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round04'
BULK = Path('D:/CCAD_Storage/runs/final_science_20260921_round04')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', choices=['human', 'infinitive'])
    setting = parser.parse_args().setting
    base = json.loads((ROOT/f'configs/fs04_{setting}_source_columns_fit.json').read_text())
    fit_targets = [2, 3] if setting == 'human' else [3, 4]
    targets = [dict(seed=s, directory=base['target_directory']) for s in fit_targets]
    paths = []
    for smoke in [True, False]:
        c = base.copy()
        stage = 'SMOKE' if smoke else 'FIT'
        rid = f'FS04_{setting.upper()}_SHARED_COLUMNS_{stage}_20260921'
        c.update(run_id=rid, source_column_training_targets=targets,
                 variants=['source_columns_scalar_mixed', 'source_columns_mixed'],
                 steps=16 if smoke else 512, budget_seconds=300,
                 purpose='Learn one shared source correction across two training dictionaries and preserve it for other dictionaries.',
                 scope='Fixed total updates and source requests; source-column and scalar arms see the same balanced randomized dictionary schedule.',
                 bulk_output_dir=str(BULK/(rid+'_checkpoints')))
        if setting == 'human':
            source = json.loads(Path(base['source_manifest']).read_text())
            c.update(source_column_sites=list(source['members']),
                     evaluation_panel=str(OUT/'HUMAN_SMOKE_PANEL.json'),
                     queries=['pronouns', 'names', 'full', 'interior_00', 'member_subset_00'],
                     budget_seconds=600)
        elif smoke:
            c['panel'] = str(OUT/'INFINITIVE_SMOKE_PANEL.json')
        path = ROOT/f'configs/fs04_{setting}_shared_columns_{stage.lower()}.json'
        save(path, c)
        paths.append(str(path))
    save(OUT/f'{setting.upper()}_SHARED_COLUMN_TRAINING_DESIGN.json', dict(
        written_at_utc=datetime.now(timezone.utc).isoformat(), fit_targets=fit_targets,
        held_targets=[s for s in range(1, 6) if s not in fit_targets], steps=512, parameterizations=['source columns', 'per-source scalar'],
        question='Does a correction trained across two target dictionaries preserve more of its functional gain in independent targets?',
        evidence='Development comparison followed by fixed-checkpoint new-context confirmation if supported.',
        reason='Single-target correction has heterogeneous transfer. Sharing its training objective across dictionary realizations directly tests a common correction.',
        fitting='Same source contexts and balanced endpoint/interior requests; randomized two-update dictionary blocks; 256 updates per dictionary in each arm.',
        preserved='Original source teacher and encoders, target dictionaries and language model remain frozen.',
        configs=paths,
        code={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
              ['scripts/train_infinitive_program.py', 'scripts/train_shift_response_space.py', 'scripts/prepare_shared_column_training.py']}))


if __name__ == '__main__':
    main()
