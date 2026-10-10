import json
from pathlib import Path

LOCAL = Path(r'E:\Projects\SAE_Lab\CCAD')
ROOT = '/mnt/nvme3-data/harvardhds/ccad-m364569'
BULK = f'{ROOT}/runs/naacl_sprint_20261010'
OUT = LOCAL / 'scripts' / 'mayo' / 'configs'
OUT.mkdir(parents=True, exist_ok=True)
trainers = [f'trainer_{i}' for i in range(16)]
prepare = dict(
    output_dir=f'{BULK}/public_targets',
    saebench_dir=f'{ROOT}/assets/saebench_pythia70m/pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4',
    trainers=trainers,
    first_id=101,
    source_parameters=f'{ROOT}/data/infinitive/source_parts.npz',
    natural_states=f'{ROOT}/data/infinitive/fit_states.npz',
    members_per_source=2,
)
(OUT / 'e4a_prepare_public_targets.json').write_text(json.dumps(prepare, indent=2))
base = json.loads(Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE04_infinitive_t3_v1_20260919/config.resolved.json').read_text())
for index in range(len(trainers)):
    seed = 101 + index
    cfg = dict(base)
    cfg.update(
        run_id=f'NAACL_E4A_public_t{seed}_v1_20261010',
        run_parent='NAACL_SPRINT_20261010',
        run_storage_root=BULK,
        source_revision='867d1b8 plus NAACL_SPRINT_20261010 working-tree scripts; the run stores its source snapshot',
        environment_spec=f'{ROOT}/code/CCAD/.aris/compute/mayo-ccad-env.json',
        device='cuda:0',
        dictionary_source_dir=f'{ROOT}/code/dictionary_learning_60ec6bf',
        dictionary_overlay_dir=f'{ROOT}/code/dictionary_learning_60ec6bf',
        model_local_dir=f'{ROOT}/assets/pythia-70m-deduped',
        fit_panel=f'{ROOT}/data/infinitive/IR01_fit_panel.json',
        panel=f'{ROOT}/data/infinitive/ROUND04_INFINITIVE_PANEL.json',
        natural_states=f'{ROOT}/data/infinitive/fit_states.npz',
        source_parameters=f'{ROOT}/data/infinitive/source_parts.npz',
        target_directory=f'{BULK}/public_targets/targets',
        target_seed=seed,
        seeds=[seed],
        relation_run=f'{BULK}/public_targets/relations/seed{seed}',
        extra_relations={name: f'{BULK}/public_targets/infinitive_matching_seed{{seed}}.npz'
                         for name in ['hungarian_decoder', 'hungarian_activation', 'activation_top2']},
        baselines=['geometry', 'hungarian_decoder', 'hungarian_activation', 'activation_top2',
                   'native_tangent_relation_8', 'raw_reconstruction'],
        variants=['tangent_full', 'tangent_mixed', 'tangent_gain'],
        budget='Three 512-update arms and 39 requests on 96 fresh contexts, at most 900 driver seconds per public target on one RTX 6000 Ada.',
        budget_seconds=900,
        purpose='E4a transfer of the published infinitive explanation to public SAEBench TopK dictionaries that differ in width and sparsity.',
        evidence_level='frozen_protocol_public_targets',
        scope='Source, panels, schedule seed and update budget equal SCIENCE04. Targets are public SAEBench TopK dictionaries at resid_post_layer_4. Geometry relations only satisfy the trainer interface and are evaluated as decoder matching.',
        statistics_unit='Public target dictionary, crossed verb/noun and paired request family; fixed published source.',
    )
    (OUT / f'e4a_public_t{seed}.json').write_text(json.dumps(cfg, indent=2))
print('written', len(trainers))
