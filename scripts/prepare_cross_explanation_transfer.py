from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round05'
BULK = Path('D:/CCAD_Storage/runs/final_science_20260921_round05')


def main():
    template = json.loads((ROOT/'configs/agreement_reuse_development_r2_20260920.json').read_text())
    records = []
    for design in ['semantic','balanced','independent']:
        origin = BULK/f'FS05_INFINITIVE_{design.upper()}_DEV_20260921'
        checkpoint = origin/'tangent_mixed/dictionary.pt'
        assert checkpoint.is_file()
        assert json.loads((origin/'status.json').read_text())['status']=='PASS'
        c = dict(template)
        c.update(run_id=f'FS05_CROSS_EXPLANATION_{design.upper()}_DEV_20260921',
                 run_parent='FINAL_SCIENCE_05', run_storage_root=str(BULK),
                 purpose='Evaluate a frozen infinitive-trained dictionary on the distinct subject-number explanation without target-response adaptation.',
                 scope='One target seed, 48 previously exposed development prefixes, all 21 original subject-number requests. Training used only the four infinitive members and their fit contexts.',
                 target_seed=3,seeds=[3],variants=[],executions=['tangent','raw_readout'],
                 source_metric_rows=0,fisher_sequences=0,query_progress=False,
                 task_adapted_reference={'grammar':str(checkpoint)},
                 transfer_training_run=str(origin),request_design=design,
                 budget='Frozen execution only; shared target initialization and source-number teacher across methods.',
                 evidence_level='development',save_pooled=False)
        path = ROOT/f'configs/fs05_cross_explanation_{design}_dev.json'
        assert not path.exists()
        path.write_text(json.dumps(c,indent=2)+'\n')
        records.append(dict(design=design,config=str(path),checkpoint=str(checkpoint),
                            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    dest=OUT/'CROSS_EXPLANATION_DESIGN.json'
    assert not dest.exists()
    dest.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Infinitive program training improves the execution of a separately supplied subject-number explanation.',
        evaluation='All prior development requests retained; no subject-number responses enter the loaded checkpoint.',
        records=records),indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
