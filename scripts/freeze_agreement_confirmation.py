from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np


def save(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')


def main():
    root=Path('artifacts/final_science_20260920_round02')
    base=json.loads(Path('configs/agreement_reuse_development_r2_20260920.json').read_text())
    panel=json.loads((root/'AGREEMENT_CONFIRMATION.json').read_text())
    rng=np.random.default_rng(2026092002)
    for j in range(8):
        panel['queries'][f'participation_{j:02d}']=rng.uniform(.1,.9,10).tolist()
    for row in panel['rows']:
        clean=row['clean_prefix'].split()
        patch=row['patch_prefix'].split()
        assert clean[0]=='The' and patch[0]=='The'
        assert clean[1]!=patch[1] or row['structure']=='within_rc'
        row['subject_pair']='|'.join(sorted([clean[1],patch[1]]))
    panel_path=root/'AGREEMENT_FROZEN_PANEL.json'
    save(panel_path,panel)
    configurations=[]
    for seed in range(1,6):
        c=dict(base)
        prior=json.loads(Path(f'configs/profile_confirmation_t{seed}_20260920.json').read_text())
        c.update(run_id=f'AGREEMENT_CONFIRM_T{seed}_20260920',target_seed=seed,seeds=[seed],
                 target_directory=prior['target_directory'],grammar_rows=len(panel['rows']),
                 grammar_panel=str(panel_path),source_metric_rows=0,
                 source_metric_cache='D:/CCAD_Storage/runs/final_science_20260920_round02/AGREEMENT_REUSE_DEVELOPMENT_R2_20260920/source_metric_roots.pt',
                 other_source_metric_cache='D:/CCAD_Storage/runs/final_science_20260920/INTERVENTION_SOURCE_METRIC_DEVELOPMENT_20260920/source_metric_roots.pt',
                 task_adapted_reference=None,audit_opened=True,candidate_family_frozen=True,
                 evidence_level='frozen_confirmation_after_whole_function_development',
                 executions=['tangent','raw_readout','refined_common','refined_source_metric','refined_other_metric'],
                 budget_seconds=1200,budget='152 unused clean-prefix units and eight fresh participation queries; ten frozen public source members; five unchanged target dictionaries',
                 scope='Source profile acquired from48 source-only fit contexts; no target response fitting; same members and solver steps for metric comparison')
        path=Path(f'configs/agreement_confirm_t{seed}_20260920.json')
        save(path,c)
        configurations.append(str(path))
    files=configurations+[str(panel_path),str(root/'AGREEMENT_SOURCE_MANIFEST.json'),str(root/'agreement_source_parts.npz'),
                          base['grammar_fit_panel'],c['source_metric_cache'],c['other_source_metric_cache'],
                          'scripts/train_intervention_changes.py','src/ccad/intervention_transport.py',
                          'scripts/prepare_agreement_reuse.py','scripts/freeze_agreement_confirmation.py']
    freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),round_id='FINAL_SCIENCE_02',
                primary='Mean nRMSE across eight fresh participation requests and five target seeds, normalized across the152 fixed prefix units',
                primary_comparison='source profile versus same-support Euclidean refinement',
                other_comparisons=['active tangent','source-direction reconstruction readout','profile from prior infinitive function'],
                secondary=['three semantic endpoints','ten single members','each syntactic structure','source effect and answer decisions'],
                inference='2000 paired draws of target seeds, subject-number lexical clusters, and fresh requests. Same subject-pair cluster weight across structures; fixed source explanation. Prefix bootstrap additionally descriptive.',
                scope='Fixed public ten-member subject-number explanation at one Pythia70M site; held out from earlier method development. Confirmation contexts and participation requests unused before freeze. Three original benchmark structures; no full-circuit benchmark claim.',
                configuration_paths=configurations,
                inputs=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),bytes=Path(p).stat().st_size) for p in files])
    save(root/'AGREEMENT_CONFIRMATION_FREEZE.json',freeze)
    print(json.dumps(dict(written_at_utc=freeze['written_at_utc'],contexts=len(panel['rows']),
                         lexical_clusters=len({r['subject_pair'] for r in panel['rows']}),configs=configurations),indent=2))


if __name__=='__main__':
    main()
