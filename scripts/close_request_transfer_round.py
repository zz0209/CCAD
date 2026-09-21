from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_science_20260921_round05'


def identity(path):
    return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    destination=OUT/'CLOSEOUT.json'
    assert not destination.exists()
    names=['REPORT.md','PROJECT_REVIEW.md','METHOD_AND_DECISION.md','RUN_INVENTORY_V2.json',
           'HUMAN_REQUEST_DEVELOPMENT.json','INFINITIVE_REQUEST_DEVELOPMENT.json',
           'CROSS_PROGRAM_GENERALIZATION.json','GENERIC_PROGRAM_GENERALIZATION.json',
           'PROGRAM_ACTION_AUDIT.json','NUMERICAL_REPLAY.json','COVARIANCE_CHECK.json','SMOKE_VALIDITY.json']
    paths=[OUT/name for name in names]+[ROOT/'paper/main.pdf',ROOT/'paper/main.tex',
        ROOT/'paper/sections/reuse_main.tex',ROOT/'paper/sections/reuse_program_confirmation.tex',
        ROOT/'paper/data/request_transfer_development.csv',ROOT/'paper/tables/request_training_development.tex',
        ROOT/'paper/tables/program_transfer_development.tex',ROOT/'delivery/final_science_20260921/README.md']
    inventory=json.loads((OUT/'RUN_INVENTORY_V2.json').read_text())
    assert len(inventory['runs'])==25
    assert sum(r['status']=='PASS' for r in inventory['runs'])==24
    assert json.loads((OUT/'NUMERICAL_REPLAY.json').read_text())['status']=='PASS'
    now=datetime.now(timezone.utc).isoformat()
    evidence=ROOT/'paper/EVIDENCE_INDEX.json'
    original=OUT/'pre_integration/EVIDENCE_INDEX.json'
    assert not original.exists()
    shutil.copyfile(evidence,original)
    current=json.loads(evidence.read_text())
    assert 'final_science_round05' not in current
    current['final_science_round05']=dict(written_at_utc=now,evidence='Development',
        location='Appendix E.7, Tables53–54',
        scope='One target per study and previously exposed functional contexts. Fixed-action checks use paired dependent natural-text requests. No independent function confirmation.',
        exporter=identity(ROOT/'scripts/export_request_transfer_results.py'),
        analyses=[identity(OUT/name) for name in names if name.endswith('.json')])
    evidence.write_text(json.dumps(current,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report=dict(written_at_utc=now,round_id='FINAL_SCIENCE_05',
        scientific_status='Development results complete; overall peer and venue judgments are not inferred from completion.',
        assets=[identity(path) for path in paths],evidence_index=identity(evidence),
        run_root='D:/CCAD_Storage/runs/final_science_20260921_round05',
        known_driver_seconds=inventory['known_driver_seconds'],bulk_bytes=inventory['bulk_bytes'],
        peak_allocated_bytes=inventory['max_peak_allocated_bytes'],
        active_model_workers=0,manuscript=json.loads((OUT/'manuscript_review/IDENTITY.json').read_text()))
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['assets','manuscript']}))


if __name__=='__main__':
    main()
