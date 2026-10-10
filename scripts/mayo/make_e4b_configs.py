import json
from pathlib import Path

LOCAL = Path(r'E:\Projects\SAE_Lab\CCAD')
ROOT = '/mnt/nvme3-data/harvardhds/ccad-m364569'
BULK = f'{ROOT}/runs/naacl_sprint_20261010'
OUT = LOCAL / 'scripts' / 'mayo' / 'configs'
OUT.mkdir(parents=True, exist_ok=True)
LAYER = 14
PREP = f'{BULK}/qwen17_prepare_l{LAYER}_v2'
prepare = dict(
    output_dir=PREP,
    model_dir=f'{ROOT}/assets/qwen3-1.7b-base',
    hook_module_path=f'model.layers.{LAYER}',
    saes={'1': dict(path=f'{ROOT}/assets/qwen-scope-1.7b-w32k-l50/layer{LAYER}.sae.pt', k=50),
          '2': dict(path=f'{ROOT}/assets/qwen-scope-1.7b-w32k-l100/layer{LAYER}.sae.pt', k=100)},
    panels={'fit': f'{ROOT}/data/grammar/R26_panel.json', 'confirmation': f'{ROOT}/data/grammar/RG08_confirmation_panel.json'},
    natural_tokenizer_dir=f'{ROOT}/assets/pythia-70m-deduped',
    natural_tokens=f'{ROOT}/data/grammar/discovery.uint16.bin',
    natural_sequences=1024,
    natural_seed=20261010,
    states_per_batch=512,
    members_per_group=64,
    sentence_prefix_text='\n',
    scope='Public Qwen-Scope TopK SAEs for Qwen3-1.7B-Base at one residual layer; source groups selected only from the source_selection split with source gradients.',
)
(OUT / 'e4b_qwen17_prepare.json').write_text(json.dumps(prepare, indent=2))
maps = dict(
    output_dir=f'{BULK}/qwen17_matching_maps_l{LAYER}_v2',
    device='cuda:0',
    dictionary_source_dir=f'{ROOT}/code/dictionary_learning_60ec6bf',
    dictionary_overlay_dir=f'{ROOT}/code/dictionary_learning_60ec6bf',
    source_checkpoint=None,
    source_run=PREP,
    source_seed=None,
    target_checkpoint_template=f'{PREP}/topk_seed{{seed}}.pt',
    target_seeds=None,
    natural_states=f'{PREP}/natural_states.npz',
    natural_count=32768,
    members_per_source=2,
)
for source, target in [(1, 2), (2, 1)]:
    m = dict(maps, output_dir=f'{maps["output_dir"]}_s{source}', source_checkpoint=f'{PREP}/topk_seed{source}.pt',
             source_seed=source, target_seeds=[target])
    (OUT / f'e4b_qwen17_maps_s{source}.json').write_text(json.dumps(m, indent=2))
base = json.loads(Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round08/RG08_GPT2_PROGRAM_CONFIRM_T2_20260921/config.resolved.json').read_text())
drop = {'resume_by_seed', 'resume_checkpoints', 'primary_comparisons', 'exclusion_manifests', 'generator_adapter', 'generation_seed'}
for source, target in [(1, 2), (2, 1)]:
    cfg = {k: v for k, v in base.items() if k not in drop}
    cfg.update(
        run_id=f'NAACL_E4B_QWEN17_S{source}_T{target}_v2_20261010',
        run_id_template=f'NAACL_E4B_QWEN17_S{{source_seed}}_T{{seed}}_v2_20261010',
        # Selected on development pairs: both arms reach their lowest training-family error at 3e-5.
        dictionary_lr=3e-5,
        run_parent='NAACL_SPRINT_20261010',
        run_storage_root=BULK,
        source_revision='867d1b8 plus NAACL_SPRINT_20261010 working-tree scripts; the run stores its source snapshot',
        environment_spec=f'{ROOT}/code/CCAD/.aris/compute/mayo-ccad-env.json',
        device='cuda:0',
        dictionary_source_dir=f'{ROOT}/code/dictionary_learning_60ec6bf',
        dictionary_overlay_dir=f'{ROOT}/code/dictionary_learning_60ec6bf',
        model_local_dir=f'{ROOT}/assets/qwen3-1.7b-base',
        model_revision='see assets/DOWNLOAD_MANIFEST.json entry qwen3-1.7b-base',
        hook_module_path=f'model.layers.{LAYER}',
        backbone_module='model',
        source_checkpoint=f'{PREP}/topk_seed{source}.pt',
        source_checkpoint_template=f'{PREP}/topk_seed{{seed}}.pt',
        source_seed=source,
        source_seeds=[1, 2],
        target_checkpoint=f'{PREP}/topk_seed{target}.pt',
        target_checkpoint_template=f'{PREP}/topk_seed{{seed}}.pt',
        target_seed=target,
        target_seeds=[1, 2],
        seeds=[source, target],
        source_run=PREP,
        panel=f'{PREP}/fit_panel.json',
        evaluation_panel=f'{PREP}/confirmation_panel.json',
        evaluation_split='confirmation',
        natural_states=f'{PREP}/natural_states.npz',
        variants=['gain', 'whole', 'program'],
        matching_maps={name: f'{BULK}/qwen17_matching_maps_l{LAYER}_v2_s{{source_seed}}/{name}_s{{source_seed}}_t{{target_seed}}.npz'
                       for name in ['hungarian_decoder', 'hungarian_activation', 'activation_top2']},
        budget='Three 512-update arms plus evaluation on 384 fresh pairs, at most 7200 driver seconds on one RTX 6000 Ada.',
        budget_seconds=7200,
        purpose='E4b transfer of a three-function grammatical explanation between public Qwen-Scope SAEs that differ in sparsity on Qwen3-1.7B-Base.',
        evidence_level='frozen_protocol_public_targets',
        scope='Source groups use only source gradients on source_selection pairs. Fitting uses the original 64 fit pairs per grammar; confirmation uses the 384 RG08 pairs. Matched update budget across arms.',
        statistics_unit='Paired sentences within three fixed grammatical paradigms; one public target per direction.',
    )
    (OUT / f'e4b_qwen17_s{source}_t{target}.json').write_text(json.dumps(cfg, indent=2))
    # Learning-rate selection on the development split, which shares no sentence with fitting or confirmation.
    for tag, lr in [('1e-5', 1e-5), ('3e-5', 3e-5)]:
        dev = dict(cfg)
        dev.pop('matching_maps')
        dev.update(
            run_id=f'NAACL_E4B_QWEN17_DEV_LR{tag}_S{source}_T{target}_v1_20261010',
            run_id_template=f'NAACL_E4B_QWEN17_DEV_LR{tag}_S{{source_seed}}_T{{seed}}_v1_20261010',
            dictionary_lr=lr,
            evaluation_panel=f'{PREP}/fit_panel.json',
            evaluation_split='development',
            eval_pairs_per_task=64,
            variants=['whole', 'program'],
            budget='Two 512-update arms evaluated on 192 development pairs, at most 3600 driver seconds.',
            budget_seconds=3600,
            purpose='E4b learning-rate selection for the Qwen3-1.7B target dictionary on development pairs.',
            evidence_level='development_selection',
        )
        (OUT / f'e4b_qwen17_dev_lr{tag}_s{source}_t{target}.json').write_text(json.dumps(dev, indent=2))
print('written')
