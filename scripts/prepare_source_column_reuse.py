from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/final_science_20260921_round04'
BULK = 'D:/CCAD_Storage/runs/final_science_20260921_round04'


def save(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    for setting in ['human', 'infinitive']:
        base_path = ROOT / ('configs/science04_shift_t2_v1.json' if setting == 'human'
                            else 'configs/science04_infinitive_t3_v1.json')
        config = json.loads(base_path.read_text())
        if setting == 'human':
            source = json.loads(Path(config['source_manifest']).read_text())
            bank = np.load(config['source_parameters'])
            identity = {s: bank[s+'__decoder'] for s in source['members']}
            old_panel = json.loads(Path(config['evaluation_panel']).read_text())
            rows = old_panel['rows']
            chosen = []
            for profession in sorted({r['profession'] for r in rows}):
                for gender in [0, 1]:
                    chosen.extend([r for r in rows if r['profession'] == profession and r['gender'] == gender][:4])
            smoke_panel = OUT / 'HUMAN_SMOKE_PANEL.json'
            save(smoke_panel, dict(rows=chosen, evidence='Previously exposed development documents'))
        else:
            identity = dict(decoder=np.load(config['source_parameters'])['decoder'])
            old_panel = json.loads(Path(config['panel']).read_text())
            smoke_panel = OUT / 'INFINITIVE_SMOKE_PANEL.json'
            small_queries = {k: old_panel['queries'][k] for k in ['predicate', 'object', 'full', 'interior_00', 'member_subset_01']}
            save(smoke_panel, dict(rows=old_panel['rows'][:8], queries=small_queries, answer=' to'))
        identity_path = OUT / f'{setting.upper()}_SOURCE_COLUMNS_IDENTITY.npz'
        if identity_path.exists():
            raise FileExistsError(identity_path)
        np.savez_compressed(identity_path, **identity)
        for smoke in [True, False]:
            c = config.copy()
            stage = 'SMOKE' if smoke else 'FIT'
            run_id = f'FS04_{setting.upper()}_SOURCE_COLUMNS_{stage}_20260921'
            c.update(run_id=run_id, run_parent='FINAL_SCIENCE_04',
                     run_storage_root=BULK, bulk_output_dir=f'{BULK}/{run_id}_checkpoints',
                     purpose='Learn source-column compensation on one development target, preserving callable source parts and frozen target dictionaries.',
                     evidence_level='development', scope='One target fit; other dictionaries receive the same source columns without fitting.',
                     variants=['source_columns_mixed'], source_column_lr=.001,
                     steps=16 if smoke else 512, budget_seconds=240 if smoke else 900,
                     audit_opened=False,
                     evaluate_source_columns={'source_columns_identity': str(identity_path)} if smoke else {})
            if setting == 'human':
                c.update(evaluate_baselines=['input_initial', 'raw_reconstruction'])
                if smoke:
                    c.update(evaluation_panel=str(smoke_panel), queries=['pronouns', 'names', 'full', 'interior_00', 'member_subset_00'])
            else:
                c.update(baselines=['native_tangent_relation_8', 'raw_reconstruction'])
                if smoke:
                    c['panel'] = str(smoke_panel)
            save(ROOT / f'configs/fs04_{setting}_source_columns_{stage.lower()}.json', c)
    save(OUT / 'SOURCE_COLUMN_DESIGN.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        question='Do seed-shared functional distortions permit a source-column correction trained once and reused by independently trained target dictionaries?',
        mechanism='Keep the original source encoder and participation requests. Learn the decoder columns supplied to the target encoder tangent, then use its existing common support and capacity allocation.',
        frozen='Base language model, original source teacher, target encoders and decoders, fixed relations, membership annotations.',
        fit=dict(human_target=2, infinitive_target=3, steps=512, source_column_lr=.001,
                 human_sites=['embed', 'mlp_0', 'resid_0'], infinitive_site='resid_4',
                 losses='Existing mixed endpoint/interior source-program loss, original head plus hidden states; no new target evaluation labels.'),
        decision='Proceed to other targets on exposed panels only after representative smoke and single-target execution. A shared improvement requires fixed transferred columns to retain gains on those other targets. Independent confirmation uses new contexts and requests.',
        controls=['Initial input-dependent rule', 'Original fixed member relation', 'Source-direction readout',
                  'Existing target-specific program training', 'Transferred scalar gains'],
        budgets=dict(round_driver_seconds=9000, peak_allocated_bytes=13000000000, additional_bulk_bytes=5000000000),
        code={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [ROOT/'scripts/train_shift_response_space.py', ROOT/'scripts/train_infinitive_program.py', Path(__file__)]}))


if __name__ == '__main__':
    main()
