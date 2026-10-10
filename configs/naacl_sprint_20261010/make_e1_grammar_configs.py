import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'naacl_sprint_20261010'
bulk = 'D:/CCAD_Storage/runs/naacl_sprint_20261010'
base = json.loads(Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round08/RG08_GPT2_PROGRAM_CONFIRM_T2_20260921/config.resolved.json').read_text())
maps = dict(
    output_dir=f'{bulk}/grammar_matching_maps_s1',
    device='cuda:0',
    dictionary_source_dir=base['dictionary_source_dir'],
    dictionary_overlay_dir=base['dictionary_overlay_dir'],
    source_checkpoint=base['source_checkpoint'],
    source_run=base['source_run'],
    source_seed=1,
    target_checkpoint_template=base['target_checkpoint_template'],
    target_seeds=[2, 3, 4, 5],
    natural_states=base['natural_states'],
    natural_count=32768,
    members_per_source=base['members_per_source'],
)
(out / 'e1_grammar_matching_maps.json').write_text(json.dumps(maps, indent=2))
cfg = {k: v for k, v in base.items() if k not in ('resume_by_seed', 'resume_checkpoints', 'primary_comparisons')}
cfg.update(
    run_id='NAACL_E1_GPT2_T2_v1_20261010',
    run_id_template='NAACL_E1_GPT2_T{seed}_v1_20261010',
    run_parent='NAACL_SPRINT_20261010',
    run_storage_root=bulk,
    variants=[],
    steps=0,
    matching_maps={name: f'{bulk}/grammar_matching_maps_s1/{name}_s{{source_seed}}_t{{target_seed}}.npz'
                   for name in ['hungarian_decoder', 'hungarian_activation', 'activation_top2']},
    budget='Evaluation only, 1500 driver seconds per target, gpu-0 lease.',
    budget_seconds=1500,
    purpose='E1 label-free feature matching on the frozen GPT2Medium confirmation panel, reusing the RG08 rows and requests.',
    evidence_level='frozen_methods_confirmation_panel',
    scope='Matching maps use natural states only. Rows, requests, model, source and targets equal RG08. No training.',
)
(out / 'e1_grammar_eval.json').write_text(json.dumps(cfg, indent=2))
print('written')
