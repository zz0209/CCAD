import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'naacl_sprint_20261010'
bulk = 'D:/CCAD_Storage/runs/naacl_sprint_20261010'
runs = {1: 'SCIENCE04_infinitive_t1_v2_20260919', 3: 'SCIENCE04_infinitive_t3_v1_20260919',
        4: 'SCIENCE04_infinitive_t4_v1_20260919', 5: 'SCIENCE04_infinitive_t5_v1_20260919'}
first = json.loads(Path(f'D:/CCAD_Storage/runs/science_upgrade_20260919/{runs[3]}/config.resolved.json').read_text())
matching = dict(
    output_dir=f'{bulk}/infinitive_matching',
    device='cpu',
    dictionary_source_dir=first['dictionary_source_dir'],
    dictionary_overlay_dir=first['dictionary_overlay_dir'],
    source_parameters=str(root / first['source_parameters']),
    natural_states=str(root / first['natural_states']),
    targets=[[seed, json.loads(Path(f'D:/CCAD_Storage/runs/science_upgrade_20260919/{run}/config.resolved.json').read_text())['target_directory']]
             for seed, run in sorted(runs.items())],
    members_per_source=2,
)
(out / 'e1_infinitive_matching.json').write_text(json.dumps(matching, indent=2))
for seed, run in runs.items():
    base = json.loads(Path(f'D:/CCAD_Storage/runs/science_upgrade_20260919/{run}/config.resolved.json').read_text())
    cfg = dict(base)
    cfg.update(
        run_id=f'NAACL_E2_infinitive_t{seed}_v1_20261010',
        run_parent='NAACL_SPRINT_20261010',
        run_storage_root=bulk,
        extra_relations={name: f'{bulk}/infinitive_matching/infinitive_matching_seed{{seed}}.npz'
                         for name in ['hungarian_decoder', 'hungarian_activation', 'activation_top2']},
        baselines=['native', 'geometry', 'geometry_gain', 'hungarian_decoder', 'hungarian_activation',
                   'activation_top2', 'native_tangent_relation_8', 'raw_reconstruction'],
        variants=['tangent_full', 'tangent_mixed'],
        budget='Two 512-update arms and 39 requests on 96 fresh contexts, at most 600 driver seconds per target.',
        budget_seconds=600,
        purpose='E1 label-free matching and E2 complete-request versus part-request training of the same input-dependent executor on the published infinitive explanation.',
        scope='Same panel, source, targets, schedule seed and update budget as SCIENCE04. tangent_full trains only on the complete request; tangent_mixed repeats the original part-request schedule.',
    )
    (out / f'e2_infinitive_t{seed}.json').write_text(json.dumps(cfg, indent=2))
print('written')
