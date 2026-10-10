import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'naacl_sprint_20261010'
bulk = 'D:/CCAD_Storage/runs/naacl_sprint_20261010'
for seed in [2, 3, 4, 5]:
    base = json.loads((root / f'runs/IR04_shift_consumer_seed{seed}_v1_20260916/config.resolved.json').read_text())
    cfg = dict(base)
    cfg.update(
        run_id=f'NAACL_E3_shift_use_seed{seed}_v1_20261010',
        run_parent='NAACL_SPRINT_20261010',
        run_storage_root=bulk,
        feature_cache_runs=[f'runs/IR04_shift_consumer_seed{seed}_v1_20260916'],
        extra_relations={name: f'{bulk}/matching_relations_seed{seed}/matching_relations.npz'
                         for name in ['hungarian_decoder', 'hungarian_activation', 'activation_top2']},
        methods=['none', 'source', 'hungarian_decoder', 'hungarian_activation', 'activation_top2', 'geometry',
                 'geometry_gain', 'native', 'raw', 'raw_reconstruction', 'whole', 'random_parts', 'parts_relation', 'parts'],
        budget='Later classifier use with cached representations for previously evaluated methods; at most 1800 driver seconds per target.',
        budget_seconds=1800,
        purpose='E3 later classifier use of label-free matching relations next to the frozen learned correspondences.',
        evidence_level='development_existing_consumer_target_replication',
        scope='Same consumer cohort, tasks, classifier recipe and frozen IR04 checkpoints. New label-free matching relations are fitted on natural text only.',
    )
    (out / f'e3_shift_use_seed{seed}.json').write_text(json.dumps(cfg, indent=2))
print('written')
