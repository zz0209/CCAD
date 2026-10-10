import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'naacl_sprint_20261010'
out.mkdir(parents=True, exist_ok=True)
bulk = 'D:/CCAD_Storage/runs/naacl_sprint_20261010'
written = []
for seed in [2, 3, 4, 5]:
    base = json.loads((root / f'runs/IR04_shift_response_seed{seed}_v1_20260916/config.resolved.json').read_text())
    match = dict(
        output_dir=f'{bulk}/matching_relations_seed{seed}',
        device='cuda:0',
        dictionary_source_dir=base['dictionary_source_dir'],
        dictionary_overlay_dir=base['dictionary_overlay_dir'],
        source_manifest=base['source_manifest'],
        source_parameters=base['source_parameters'],
        target_directory=base['target_directory'],
        target_seed=seed,
        model_local_dir=base['model_local_dir'],
        paired_tokens=base['paired_tokens'],
        natural_sequences=512,
        batch_sequences=16,
        members_per_source=base['members_per_source'],
    )
    path = out / f'e1_matching_relations_seed{seed}.json'
    path.write_text(json.dumps(match, indent=2))
    written.append(path)
    for panel, panel_path, split in [
        ('r59panel', r'artifacts\correspondence_reform_20260913\R59_CONFIRMATION_PANEL.json', 'test_confirmation'),
        ('ir04panel', base['evaluation_panel'], base['evaluation_split']),
    ]:
        cfg = dict(base)
        cfg.update(
            run_id=f'NAACL_E1_shift_{panel}_seed{seed}_v2_20261010',
            run_parent='NAACL_SPRINT_20261010',
            run_storage_root=bulk,
            evaluation_panel=panel_path,
            evaluation_split=split,
            extra_relations={name: f'{bulk}/matching_relations_seed{seed}/matching_relations.npz'
                             for name in ['hungarian_decoder', 'hungarian_activation', 'activation_top2']},
            methods=['none', 'source', 'hungarian_decoder', 'hungarian_activation', 'activation_top2',
                     'geometry', 'geometry_gain', 'native', 'raw', 'raw_reconstruction',
                     'whole', 'random_parts', 'parts_relation', 'parts'],
            budget='1800 driver seconds, 16 documents and 4096 padded tokens per batch, gpu-0 lease. No fitting.',
            budget_seconds=1800,
            purpose='E1 standard feature matching versus learned correspondences on complete and part deletions of the published SHIFT explanation.',
            evidence_level='frozen_methods_confirmation_panel',
            scope='Label-free matching relations are fitted on natural text only. Learned correspondences are the frozen IR04 checkpoints. No method or threshold is selected on this panel.',
        )
        path = out / f'e1_shift_{panel}_seed{seed}.json'
        path.write_text(json.dumps(cfg, indent=2))
        written.append(path)
for p in written:
    print(p)
