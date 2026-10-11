import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
data = f'{M}/data/cross_seed'
qwen = f'{M}/runs/naacl_sprint_20261010/qwen17_prepare_l14_v2'
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
base = json.loads((root / 'configs/cross_seed/mayo/x1c_s2_t3.json').read_text())
base.update(source_revision=revision, run_parent='CONTAINMENT', matched_gain_max=4., budget='Mayo GPU, one job per run.')

# CONTAINMENT experiment 4: the grammatical explanation of one public Qwen-Scope dictionary of Qwen3-1.7B carried into
# the other, with the task-free natural-text family (many-to-many), part supervision, and one-to-one controls with gains
# up to four under both objectives, three training seeds per direction.
for source, target in [(1, 2), (2, 1)]:
    for seed in [0, 1, 2]:
        cfg = dict(base)
        cfg.update(run_id=f'CSM_C4_QWEN17_s{source}_t{target}_ts{seed}_v1_20261011', seeds=[source, target],
                   model_revision='see assets/DOWNLOAD_MANIFEST.json entry qwen3-1.7b-base',
                   source=dict(model_dir=f'{M}/assets/qwen3-1.7b-base', model_label='Qwen/Qwen3-1.7B-Base',
                               sae_checkpoint=f'{qwen}/topk_seed{source}.pt', sae_label=f'Qwen-Scope layer 14 dictionary {source}',
                               hook_module_path='model.layers.14', gate=f'{qwen}/topk_s{source}_source.npz',
                               model_files=['config.json', 'model.safetensors']),
                   target=dict(same_model=True, sae_checkpoint=f'{qwen}/topk_seed{target}.pt',
                               sae_label=f'Qwen-Scope layer 14 dictionary {target}', hook_module_path='model.layers.14'),
                   natural_tokens=f'{data}/qwen_natural_train.uint32.bin', natural_eval_tokens=f'{data}/qwen_natural_eval.uint32.bin',
                   token_dtype='<u4', panel=f'{qwen}/fit_panel.json',
                   evaluation_panel=f'{M}/runs/naacl_sprint_20261010/qwen17_fresh_panel_rg09/confirmation_panel.json',
                   evaluation_split='rg09_new_sentence_confirmation', eos_token_id=151643,
                   families=['matched_natural', 'membership_natural', 'matched_parts', 'membership_parts'], training_seed=seed,
                   purpose=('CONTAINMENT experiment 4: a correspondence trained only on generic natural text with random feature '
                            'subsets, against part supervision and one-to-one controls with gains up to four, for the '
                            'Qwen3-1.7B grammatical explanation between public Qwen-Scope dictionaries.'),
                   scope=(f'Qwen3-1.7B-Base layer 14, source Qwen-Scope dictionary {source} with its R26 source groups, '
                          f'target dictionary {target}. Natural text is the exported FineWeb slice tokenized for Qwen; '
                          'fitting uses the R26 fit pairs and evaluation the RG09 confirmation pairs.'))
        (configs / f'c4_qwen17_s{source}_t{target}_ts{seed}.json').write_text(json.dumps(cfg, indent=1))
        print(cfg['run_id'])

# One-to-one controls with gains up to four, natural-text and part-supervised, for the eight GPT-2 Medium directions
# of X1 and X1c, with the X1 recipe otherwise unchanged; the many-to-many arms are the existing X1 and X1c runs.
for source, target in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5), (5, 1)]:
    cfg = dict(base)
    cfg.update(run_id=f'CSM_C7_GPT2_matched4_s{source}_t{target}_v1_20261011', seeds=[source, target],
               families=['matched_natural', 'matched_parts'],
               purpose='One-to-one controls with gains up to four for the GPT-2 Medium grammatical explanations.',
               scope=(f'GPT-2 Medium block 11, source TopK SAE seed {source} with its R26 source groups, target SAE seed '
                      f'{target}. Same candidates, data, steps and evaluation pairs as X1 and X1c.'))
    cfg['source'] = dict(base['source'], sae_checkpoint=f'{data}/gpt2_saes/topk_seed{source}.pt',
                         sae_label=f'GPT-2 Medium TopK SAE seed {source}', gate=f'{data}/gpt2_saes/topk_s{source}_source.npz')
    cfg['target'] = dict(base['target'], sae_checkpoint=f'{data}/gpt2_saes/topk_seed{target}.pt',
                         sae_label=f'GPT-2 Medium TopK SAE seed {target}')
    (configs / f'c7_gpt2_matched4_s{source}_t{target}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
    # CONTAINMENT experiment 5: the natural-text many-to-many family on candidates replaced by random target features
    # of matched activation frequency, with the same number of candidates per source feature.
    cfg = dict(cfg, run_id=f'CSM_C5_GPT2_capacity_s{source}_t{target}_v1_20261011', families=['capacity_natural'],
               capacity_bins=64,
               purpose=('CONTAINMENT experiment 5: capacity control for the natural-text many-to-many correspondence, '
                        'with each candidate replaced by a random target feature from the same activation-frequency bin.'))
    (configs / f'c5_gpt2_capacity_s{source}_t{target}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
