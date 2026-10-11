import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
SEEDS = [0, 1]
GRID = ['w16k_lower', 'w16k_higher', 'w65k_lowest', 'w16k_highest', 'w65k_highest']
# READER_COVERAGE independent confirmation on the five E10 grid targets, none of which took part in choosing the main
# method. The main method (annotated subsets fitted to Ledoit-Wolf whitened layer-22 states), its one-to-one control
# with gains up to four and the task-free natural-text map are trained with the E10 recipe and training seeds; the
# scalar part and subset maps and Hungarian matching are the saved E10 maps. Held-out readers use test biographies.
readers_base = json.loads((configs / 'exp1_gemma_readers_w65k.json').read_text())
for key in GRID:
    e10 = json.loads((configs / f'e10_gemma_grid_{key}.json').read_text())
    cfg = dict(e10, run_id=f'CSM_CONF_GEMMA2B_{key}_v1_20261011', run_root=f'{M}/runs/cross_seed_models',
               fit_families=['subsets_state', 'matched_subsets_state', 'natural'], fit_inits=['hungarian'],
               fit_seeds=SEEDS, eval_split='test', skip_use=True, state_reader='ledoit_wolf', whiten_bios=1024,
               whiten_tokens=60000, matched_gain_max=4., full_model=True,
               natural_lm_texts=f'{M}/data/cross_seed/natural_fineweb_slice.json', natural_lm_len=128,
               natural_lm_batch=4, natural_steps=512, natural_eval_seed=1,
               purpose=('READER_COVERAGE confirmation: the main method, its one-to-one control and the task-free '
                        'natural-text map on a Gemma Scope target that took no part in choosing the method.'))
    (configs / f'conf_gemma_{key}.json').write_text(json.dumps(cfg, indent=1))
    e10_run = f'{M}/runs/naacl_sprint_20261010/{e10["run_id"]}'
    conf_run = f'{M}/runs/cross_seed_models/{cfg["run_id"]}'
    readers = dict(readers_base, **{k: e10[k] for k in ['target_width', 'target_choice', 'target_label']},
                   run_id=f'CSM_CONF_GEMMA2B_readers_{key}_v1_20261011', eval_split='test', eval_bios=256,
                   probe_run=e10_run,
                   map_runs=[dict(run=e10_run, prefix='e10_', methods=['hungarian_decoder'] +
                                  [f'{f}_hungarian_s{s}' for s in SEEDS for f in ['part_sup', 'subset_sup']]),
                             dict(run=conf_run, prefix='conf_',
                                  methods=[f'{f}_hungarian_s{s}' for s in SEEDS
                                           for f in ['subset_state_sup', 'matched_subset_state_sup', 'natural_sup']])],
                   purpose='READER_COVERAGE confirmation: held-out readers on test biographies for a grid target.')
    (configs / f'conf_gemma_readers_{key}.json').write_text(json.dumps(readers, indent=1))
    print(cfg['run_id'], readers['run_id'])
    if key == 'w16k_lower':
        smoke = dict(cfg, run_id=cfg['run_id'] + '_SMOKE', natural_bios=64, fit_bios=32, fit_steps=2, natural_steps=2,
                     eval_bios=16, limit_use=64, fit_seeds=[0], whiten_bios=32)
        (configs / 'conf_gemma_smoke.json').write_text(json.dumps(smoke, indent=1))
        smoke_readers = dict(readers, run_id=readers['run_id'] + '_SMOKE', eval_bios=8, natural_docs=4, random_requests=2,
                             later_epochs=1,
                             map_runs=[dict(run=e10_run, prefix='e10_', methods=['hungarian_decoder', 'subset_sup_hungarian_s0']),
                                       dict(run=conf_run + '_SMOKE', prefix='conf_',
                                            methods=[f'{f}_hungarian_s0' for f in
                                                     ['subset_state_sup', 'matched_subset_state_sup', 'natural_sup']])])
        (configs / 'conf_gemma_readers_smoke.json').write_text(json.dumps(smoke_readers, indent=1))
        print(smoke['run_id'], smoke_readers['run_id'])
