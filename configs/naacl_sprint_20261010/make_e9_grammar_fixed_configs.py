import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'naacl_sprint_20261010'
bulk = 'D:/CCAD_Storage/runs/naacl_sprint_20261010'
base = json.loads(Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round08/RG08_GPT2_PROGRAM_CONFIRM_T2_20260921/config.resolved.json').read_text())
cfg = {k: v for k, v in base.items() if k not in ('resume_by_seed', 'resume_checkpoints', 'primary_comparisons')}
cfg.update(
    run_parent='NAACL_SPRINT_20261010',
    run_storage_root=bulk,
    variants=['membership_whole', 'membership'],
    membership_candidates=16,
    membership_init=f'{bulk}/grammar_matching_maps_s1/hungarian_decoder_s{{source_seed}}_t{{target_seed}}.npz',
    budget='1500 driver seconds per target, gpu-0 lease.',
    evidence_level='fixed_target_only_correspondence',
    scope=('Fixed part-level memberships over target features, run with the target SAE alone and the target dictionary '
           'frozen. Complete and part supervision share inputs, request schedule, updates and parameters. Rows, requests, '
           'model, source and targets equal RG08.'),
    purpose=('E9 fixed correspondence for the GPT2Medium grammatical explanation, testing whether part supervision of a '
             'target-only membership improves part fidelity over complete supervision.'),
)
for lr in (0.005, 0.02):
    dev = dict(cfg, membership_lr=lr, evaluation_panel='', evaluation_split='development', eval_pairs_per_task=32,
               run_id=f'NAACL_E9_GPT2_FIXED_DEV_LR{lr}_T2_v1_20261010',
               run_id_template=f'NAACL_E9_GPT2_FIXED_DEV_LR{lr}_T{{seed}}_v1_20261010',
               evidence_level='development')
    (out / f'e9_gpt2_fixed_dev_lr{lr}.json').write_text(json.dumps(dev, indent=2))
# Learning rate .02 gave the lower part error of the part-supervised membership on the development split of target 2.
confirm = dict(cfg, membership_lr=0.02, run_id='NAACL_E9_GPT2_FIXED_T2_v1_20261010',
               run_id_template='NAACL_E9_GPT2_FIXED_T{seed}_v1_20261010',
               dev_choice='membership_lr=0.02 from NAACL_E9_GPT2_FIXED_DEV_LR{0.005,0.02}_T2_v1_20261010')
(out / 'e9_gpt2_fixed_confirm.json').write_text(json.dumps(confirm, indent=2))
print('written')
