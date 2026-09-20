import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/final_science_20260920_round03'


def main():
    cfg=json.loads((ROOT/'configs/ir04_shift_consumer_seed2_v1.json').read_text())
    cfg.update(run_id='FS03_TRAINED_PROGRAM_CONSUMER_DEV_T2_20260920',
        run_parent='FINAL_SCIENCE_03', run_storage_root='D:/CCAD_Storage/runs/final_science_20260920_round03',
        evidence_level='development_existing_consumer', budget_seconds=1500,
        purpose='Test the frozen input-dependent program in the existing explanation-use consumer',
        scope='Existing four tasks and exposed consumer cohort; no new correspondence training or labels',
        budget='One target development comparison within the current round allowance',
        audit_opened=False, adapted_programs={},
        methods=['none','source','native','geometry_gain','raw','raw_reconstruction',
            'input_tangent_budget','input_gain','input_program'],
        feature_cache_runs=['runs/IR04_shift_consumer_seed2_v1_20260916'])
    cfg.pop('confirmation_freeze')
    cfg['adapted_input_programs']={
        name:dict(directory=f'D:/CCAD_Storage/training_curves/SCIENCE04_shift_t2_v1_20260919/{variant}',
            sites=['embed','mlp_0','resid_0'],source_gains=variant=='tangent_gain')
        for name,variant in [('input_gain','tangent_gain'),('input_program','tangent_mixed')]}
    original=ROOT/cfg['training_panel']
    panel=json.loads(original.read_text())
    chosen=[]
    for split in ['train','test']:
        for profession in [5,25,12,24]:
            for gender in [0,1]:
                rows=[r for r in panel['rows'] if r['split']==split and
                    r['profession']==profession and r['gender']==gender]
                assert len(rows)>=8
                chosen.extend(sorted(rows,key=lambda r:r['document_sha256'])[:8])
    small={**panel,'rows':chosen}
    small_path=ART/'CONSUMER_SMOKE_PANEL.json'
    assert not small_path.exists()
    small_path.write_text(json.dumps(small,indent=2)+'\n')
    destinations=[]
    for mode in ['development','smoke']:
        current=json.loads(json.dumps(cfg))
        if mode=='smoke':
            current.update(run_id='FS03_TRAINED_PROGRAM_CONSUMER_SMOKE_T2_20260920',
                budget_seconds=180,training_panel=small_path.as_posix(),evaluation_panel=small_path.as_posix(),
                feature_cache_runs=[],methods=['none','source','input_tangent_budget','input_gain','input_program'])
        path=ROOT/f'configs/final_science03_consumer_{mode}.json'
        assert not path.exists()
        path.write_text(json.dumps(current,indent=2)+'\n')
        destinations.append(dict(path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    decision=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),configurations=destinations,
        source_panel=dict(path=original.as_posix(),sha256=hashlib.sha256(original.read_bytes()).hexdigest()),
        question='Does the frozen program that improves part fidelity also preserve the use of the same published explanation after classifier retraining?',
        primary='Mean retrained accuracy over the three named parts and four existing tasks; program-minus-gain and program-minus-initial.',
        secondary='Full accuracy, worst-group accuracy and source-relative part ordering; same head recipe and source metadata for all methods.',
        scope='Development on an already exposed consumer cohort, one target. Confirm promising effects on unused documents and additional targets.',
        decision='Use this direct consumer result to choose between retaining the strongest program construction and changing the learned execution objective.')
    (ART/'CONSUMER_DEVELOPMENT_DESIGN.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(dict(configurations=destinations,smoke_rows=len(chosen))))


if __name__=='__main__':
    main()
