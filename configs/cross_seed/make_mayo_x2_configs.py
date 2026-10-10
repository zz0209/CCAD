import json
import subprocess
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'cross_seed' / 'mayo'
out.mkdir(parents=True, exist_ok=True)
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
data = f'{M}/data/cross_seed'
runs = f'{M}/runs/cross_seed_models'
models = f'{M}/assets/seed_models'
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
manifest = json.loads((root / 'artifacts/final_three_research_20260909/training_material/TOKEN_MANIFEST.json').read_text())
seed_manifest = {
    'x21': dict(repo='stanford-crfm/arwen-gpt2-medium-x21', commit='096da99a0ffe50dc3b5c8618e3e7b0bfbff84b77'),
    'x49': dict(repo='stanford-crfm/beren-gpt2-medium-x49', commit='7c3302ebe5bf8e622661b8c96a639a8ada3a7d16'),
}
e9 = json.loads((root / 'configs/naacl_sprint_20261010/e9_gpt2_fixed_confirm.json').read_text())
common = dict(source_revision=revision, run_parent='CROSS_SEED_MODELS', run_storage_root=runs, device='cuda:0',
              audit_opened=False, candidate_family_frozen=True, dataset_revision=e9['dataset_revision'],
              dictionary_source_dir=f'{M}/code/dictionary_learning_60ec6bf',
              dictionary_overlay_dir=f'{M}/code/dictionary_learning_60ec6bf', budget='Mayo GPU, one job per run.')

# The same natural token stream as the GPT-2 Medium SAEs, with Mayo paths.
mayo_manifest = dict(manifest)
mayo_manifest['outputs'] = {split: dict(info, path=f'{data}/{split}.uint16.bin') for split, info in manifest['outputs'].items()}
(out / 'TOKEN_MANIFEST_MAYO.json').write_text(json.dumps(mayo_manifest, indent=2))
base = json.loads((root / 'configs/final3_r19_gpt2_dual_objective_five_seed_16m_v1.json').read_text())
for model in ['x21', 'x49']:
    cfg = dict(base)
    cfg.update(run_id=f'CSM_SAE_gpt2_{model}_v1_20261010', round_id='CROSS_SEED_MODELS', objectives=['topk'], seeds=[1, 2],
               model_local_dir=f'{models}/gpt2-medium-{model}', model_id=seed_manifest[model]['repo'],
               model_revision=seed_manifest[model]['commit'], model_license='not stated on the model card',
               token_manifest=f'configs/cross_seed/mayo/TOKEN_MANIFEST_MAYO.json',
               bulk_output_dir=f'{runs}/saes/CSM_SAE_gpt2_{model}_v1_20261010',
               dictionary_source_dir=common['dictionary_source_dir'], dictionary_overlay_dir=common['dictionary_overlay_dir'],
               checkpoint_steps=[4096, 16384], budget_seconds=7200,
               purpose=f'Two TopK SAE seeds on the {model} GPT-2 Medium seed model, with the recipe of the OpenAI GPT-2 Medium SAEs.',
               scope=('Same 16,777,216 natural training tokens, block 11, 8,192 features, k=64 and schedule as '
                      'FINAL3_R19; only the language model differs.'))
    (out / f'sae_{model}.json').write_text(json.dumps(cfg, indent=2))
    print(cfg['run_id'])


def sae_path(model, seed):
    if model == 'openai':
        return f'{data}/gpt2_saes/topk_seed{seed}.pt'
    return f'{runs}/saes/CSM_SAE_gpt2_{model}_v1_20261010/step_16384/topk_seed{seed}.pt'


def model_dir(model):
    return f'{models}/gpt2-medium-openai' if model == 'openai' else f'{models}/gpt2-medium-{model}'


for model in ['x21', 'x49']:
    cfg = dict(common, run_id=f'CSM_SELECT_gpt2_{model}_s1_v1_20261010', generator_script='scripts/select_grammar_explanation.py',
               seeds=[1], model_revision=seed_manifest[model]['commit'], model_dir=model_dir(model),
               model_label=seed_manifest[model]['repo'], sae_checkpoint=sae_path(model, 1), sae_label=f'{model} TopK SAE seed 1',
               hook_module_path='transformer.h.11', panel=f'{data}/r26_panel.json',
               tasks=e9['tasks'], members_per_component=64, max_length=64, gradient_batch_pairs=2, eos_token_id=50256,
               purpose=f'Grammatical explanation of {model} SAE seed 1 by the R26 contrast rule.',
               scope='Source-selection rows of the R26 panel only.')
    (out / f'select_{model}.json').write_text(json.dumps(cfg, indent=2))
    print(cfg['run_id'])


def transfer(name, source_model, source_seed, target_model, target_seed, gate):
    same = source_model == target_model
    cfg = dict(common, run_id=f'CSM_X2_gpt2_{name}_v1_20261010', generator_script='scripts/cross_seed_transfer.py',
               seeds=[source_seed, target_seed], model_revision=seed_manifest.get(source_model, dict(commit=e9['model_revision']))['commit'],
               purpose=('X2: carry the grammatical explanation of one GPT-2 Medium seed model into a new SAE of the same model '
                        'or into an SAE of an independently trained seed model, without and with task data.'),
               scope=(f'Source {source_model} SAE seed {source_seed}, target {target_model} SAE seed {target_seed}, block 11. '
                      'Candidates and one-to-one matches by natural-text activation correlation on shared token streams.'),
               source=dict(model_dir=model_dir(source_model), model_label=source_model, sae_checkpoint=sae_path(source_model, source_seed),
                           sae_label=f'{source_model} TopK SAE seed {source_seed}', hook_module_path='transformer.h.11', gate=gate),
               target=dict(same_model=same, sae_checkpoint=sae_path(target_model, target_seed), sae_label=f'{target_model} TopK SAE seed {target_seed}',
                           hook_module_path='transformer.h.11', **({} if same else dict(model_dir=model_dir(target_model), model_label=target_model))),
               natural_tokens=f'{data}/train.uint16.bin', natural_eval_tokens=f'{data}/validation.uint16.bin',
               context=128, natural_batch=4, correlation_sequences=512, natural_eval_sequences=128, natural_eval_seed=1,
               candidates_per_source=16, panel=f'{data}/r26_panel.json', evaluation_panel=f'{data}/e9_confirmation_panel.json',
               evaluation_split=e9['evaluation_split'], tasks=e9['tasks'], queries=e9['queries'], eos_token_id=50256,
               fit_pairs_per_task=64, eval_pairs_per_task=128, eval_batch_pairs=16, batch_pairs=4,
               families=['matched_natural', 'membership_natural', 'matched_parts', 'membership_parts'],
               steps=512, lr=0.02, training_seed=0, progress_every=32)
    if not same:
        cfg['source']['model_files'] = cfg['target']['model_files'] = ['config.json', 'model.safetensors']
    (out / f'x2_{name}.json').write_text(json.dumps(cfg, indent=2))
    print(cfg['run_id'])


gate = {model: f'{runs}/CSM_SELECT_gpt2_{model}_s1_v1_20261010/source.npz' for model in ['x21', 'x49']}
for source, other in [('x21', 'x49'), ('x49', 'x21')]:
    transfer(f'{source}s1_{source}s2', source, 1, source, 2, gate[source])
    transfer(f'{source}s1_{other}s1', source, 1, other, 1, gate[source])
    transfer(f'{source}s1_{other}s2', source, 1, other, 2, gate[source])
# The interrupted X1c runs on the OpenAI model, now on Mayo.
for source_seed, target_seed in [(2, 3), (3, 4), (4, 5), (5, 1)]:
    cfg = json.loads((root / f'configs/cross_seed/x1c_gpt2_s{source_seed}_t{target_seed}.json').read_text())
    cfg.update({k: common[k] for k in ['source_revision', 'run_storage_root', 'dictionary_source_dir', 'dictionary_overlay_dir']},
               run_id=f'CSM_X1C_gpt2_s{source_seed}_t{target_seed}_v2_20261010',
               natural_tokens=f'{data}/train.uint16.bin', natural_eval_tokens=f'{data}/validation.uint16.bin',
               panel=f'{data}/r26_panel.json', evaluation_panel=f'{data}/e9_confirmation_panel.json')
    cfg['source'] = dict(cfg['source'], model_dir=model_dir('openai'), sae_checkpoint=sae_path('openai', source_seed),
                         gate=f'{data}/gpt2_saes/topk_s{source_seed}_source.npz', model_files=['config.json', 'model.safetensors'])
    cfg['target'] = dict(cfg['target'], sae_checkpoint=sae_path('openai', target_seed))
    (out / f'x1c_s{source_seed}_t{target_seed}.json').write_text(json.dumps(cfg, indent=2))
    print(cfg['run_id'])
