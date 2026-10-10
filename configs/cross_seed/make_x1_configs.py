import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'cross_seed'
bulk = 'D:/CCAD_Storage/runs/cross_seed_models'
gpt2 = 'D:/CCAD_Storage/models/gpt2-medium/6dcaa7a952f72f9298047fd5137cd6e4f05f41da'
saes = 'D:/CCAD_Storage/training_curves/FINAL3_R19_gpt2_dual_objective_five_seed_16m_v1_20260909/step_16384'
e9 = json.loads((root / 'configs/naacl_sprint_20261010/e9_gpt2_fixed_confirm.json').read_text())
# X1: within GPT-2 Medium, the grammatical explanation of SAE seed 1 is carried into SAE seeds 2 to 5. The natural
# families never see task data; the task families reuse the E9 fitting rows and requests.
for target in [2, 3, 4, 5]:
    cfg = dict(
        run_id=f'CSM_X1_gpt2_s1_t{target}_v1_20261010', run_parent='CROSS_SEED_MODELS', run_storage_root=bulk,
        generator_script='scripts/cross_seed_transfer.py', device='cuda:0', seeds=[1, target],
        purpose=('X1 within-model test: does a membership trained on natural text with random feature subsets, '
                 'without task data, carry the parts of the grammatical explanation as well as task supervision?'),
        scope=('GPT-2 Medium block 11, source TopK SAE seed 1 with the R26 source groups, target SAE seed '
               f'{target}. Candidates and one-to-one matches by natural-text activation correlation. Evaluation on '
               'the E9 confirmation pairs, which no family uses for fitting.'),
        budget='Local gpu-0 lease, one target per run.', audit_opened=True, candidate_family_frozen=True,
        model_revision=e9['model_revision'], dataset_revision=e9['dataset_revision'],
        dictionary_source_dir=e9['dictionary_source_dir'], dictionary_overlay_dir=e9['dictionary_overlay_dir'],
        source=dict(model_dir=gpt2, model_label='openai-community/gpt2-medium '+e9['model_revision'],
                    sae_checkpoint=f'{saes}/topk_seed1.pt', sae_label='GPT-2 Medium TopK SAE seed 1',
                    hook_module_path='transformer.h.11',
                    gate=str(root/'runs/REFORM_R26_contrast_components_v2_20260913/topk_s1_source.npz')),
        target=dict(same_model=True, sae_checkpoint=f'{saes}/topk_seed{target}.pt',
                    sae_label=f'GPT-2 Medium TopK SAE seed {target}', hook_module_path='transformer.h.11'),
        natural_tokens=str(root/'artifacts/final_three_research_20260909/training_material/train.uint16.bin'),
        natural_eval_tokens=str(root/'artifacts/final_three_research_20260909/training_material/validation.uint16.bin'),
        context=128, natural_batch=4, correlation_sequences=512, natural_eval_sequences=128, natural_eval_seed=1,
        candidates_per_source=16, panel=str(root/e9['panel']), evaluation_panel=str(root/e9['evaluation_panel']),
        evaluation_split=e9['evaluation_split'], tasks=e9['tasks'], queries=e9['queries'], eos_token_id=50256,
        fit_pairs_per_task=64, eval_pairs_per_task=128, eval_batch_pairs=16, batch_pairs=4,
        families=['matched_natural', 'membership_natural', 'matched_parts', 'membership_parts', 'membership_complete'],
        steps=512, lr=0.02, training_seed=0, progress_every=32)
    (out / f'x1_gpt2_t{target}.json').write_text(json.dumps(cfg, indent=2))
    print(cfg['run_id'])
    # X1b doubles the updates: the natural family sees 1,024 natural batches, and the mixed family alternates 512
    # natural batches with the 512 task updates of part supervision in X1.
    cfg_b = dict(cfg, run_id=f'CSM_X1B_gpt2_s1_t{target}_v1_20261010', steps=1024,
                 families=['membership_natural', 'membership_mixed'],
                 purpose=('X1b within-model test: does longer natural-text training, or alternating natural-text subsets '
                          'with task parts, keep both part fidelity and natural-text fidelity?'))
    (out / f'x1b_gpt2_t{target}.json').write_text(json.dumps(cfg_b, indent=2))
    print(cfg_b['run_id'])
# X1c repeats X1 with the explanations of source SAEs 2 to 5, each carried into the next seed in the R26 cycle.
for source_seed, target in [(2, 3), (3, 4), (4, 5), (5, 1)]:
    cfg = json.loads((out / 'x1_gpt2_t2.json').read_text())
    cfg.update(run_id=f'CSM_X1C_gpt2_s{source_seed}_t{target}_v1_20261010', seeds=[source_seed, target],
               families=['matched_natural', 'membership_natural', 'membership_parts'],
               purpose='X1c: the X1 comparison for the explanations of four further source SAEs.',
               scope=(f'GPT-2 Medium block 11, source TopK SAE seed {source_seed} with its R26 source groups, target SAE '
                      f'seed {target}. Same recipe and evaluation pairs as X1.'))
    cfg['source'] = dict(cfg['source'], sae_checkpoint=f'{saes}/topk_seed{source_seed}.pt',
                         sae_label=f'GPT-2 Medium TopK SAE seed {source_seed}',
                         gate=str(root/f'runs/REFORM_R26_contrast_components_v2_20260913/topk_s{source_seed}_source.npz'))
    cfg['target'] = dict(cfg['target'], sae_checkpoint=f'{saes}/topk_seed{target}.pt', sae_label=f'GPT-2 Medium TopK SAE seed {target}')
    (out / f'x1c_gpt2_s{source_seed}_t{target}.json').write_text(json.dumps(cfg, indent=2))
    print(cfg['run_id'])
