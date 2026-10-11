import json
from pathlib import Path

from make_reader_configs import FEMALE, MALE

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
SEEDS = [0, 1, 2]
# READER_COVERAGE EXP-2 on the two formal Gemma Scope targets. The classifier-reader cells of the 2x2 are the saved
# E4c subsets maps (annotated) and E8a random maps (label-free), trained with the same candidates, initialization,
# draws, steps and learning rate; the new runs train the same two request families against whitened layer-22 token
# states, once with the Ledoit-Wolf covariance and once with per-dimension standardization (one development choice).
for key in ['w65k', 'l0low16k']:
    base = json.loads((configs / f'e4c_gemma_final_{key}.json').read_text())
    state_runs = {}
    for reader in ['ledoit_wolf', 'standardized']:
        cfg = dict(base, run_id=f'CSM_EXP2_GEMMA2B_state_{key}_{reader}_v1_20261011', run_root=f'{M}/runs/cross_seed_models',
                   fit_families=['subsets_state', 'random_state'], fit_inits=['hungarian'], fit_seeds=SEEDS,
                   eval_split='dev', skip_use=True, state_reader=reader, whiten_bios=1024, whiten_tokens=60000,
                   purpose=('READER_COVERAGE EXP-2: annotated and label-free request families fitted to whitened layer-22 '
                            'token states, the reader-complete cells of the reader-by-request comparison.'))
        (configs / f'exp2_gemma_state_{key}_{reader}.json').write_text(json.dumps(cfg, indent=1))
        state_runs[reader] = f'{M}/runs/cross_seed_models/{cfg["run_id"]}'
        if key == 'w65k' and reader == 'ledoit_wolf':
            smoke = dict(cfg, run_id=cfg['run_id'] + '_SMOKE', natural_bios=64, fit_bios=32, fit_steps=2, eval_bios=16,
                         limit_use=64, fit_seeds=[0], whiten_bios=32)
            (configs / 'exp2_gemma_state_smoke.json').write_text(json.dumps(smoke, indent=1))
        print(cfg['run_id'])
    # Held-out readers on development biographies for all four cells and the references of rule DR2.
    readers = json.loads((configs / f'exp1_gemma_readers_{key}.json').read_text())
    c1 = readers['probe_run']
    readers.update(
        run_id=f'CSM_EXP2_GEMMA2B_readers_{key}_v1_20261011',
        map_runs=[dict(run=f'{M}/runs/naacl_sprint_20261010/{base["run_id"]}', prefix='e4c_',
                       methods=[f'subset_sup_hungarian_s{s}' for s in SEEDS]),
                  dict(run=f'{M}/runs/naacl_sprint_20261010/NAACL_E8A_GEMMA2B_labelfree_{key}_v1_20261010', prefix='e8a_',
                       methods=[f'random_sup_hungarian_s{s}' for s in SEEDS]),
                  dict(run=state_runs['ledoit_wolf'], prefix='lw_',
                       methods=[f'{f}_hungarian_s{s}' for s in SEEDS for f in ['subset_state_sup', 'random_state_sup']]),
                  dict(run=state_runs['standardized'], prefix='std_',
                       methods=[f'{f}_hungarian_s{s}' for s in SEEDS for f in ['subset_state_sup', 'random_state_sup']]),
                  dict(run=c1, prefix='', methods=['hungarian_decoder'] +
                       [f'{f}_hungarian_s{s}' for s in SEEDS for f in ['natural_sup', 'part_sup']])],
        female_words=FEMALE, male_words=MALE,
        purpose='READER_COVERAGE EXP-2: held-out readers for the four reader-by-request cells on development biographies.')
    (configs / f'exp2_gemma_readers_{key}.json').write_text(json.dumps(readers, indent=1))
    print(readers['run_id'])
    if key == 'l0low16k':
        # The same readers on GPU 1 in parallel with the 65k readers on GPU 0; the version 1 identifier stays with the
        # GPU 0 lane, which is stopped before it reaches this target.
        parallel = dict(readers, run_id=f'CSM_EXP2_GEMMA2B_readers_{key}_v2_20261011')
        (configs / f'exp2_gemma_readers_{key}_v2.json').write_text(json.dumps(parallel, indent=1))
        print(parallel['run_id'])
    # Controls under the chosen main objective (reader-complete, annotated subsets, Ledoit-Wolf whitening): the
    # one-to-one map with gains up to four, the capacity control on frequency-matched random candidates, and the
    # classifier-loss one-to-one map trained with its own deletion cap.
    ext = dict(base, run_id=f'CSM_EXT_GEMMA2B_controls_{key}_v1_20261011', run_root=f'{M}/runs/cross_seed_models',
               fit_families=['matched_subsets_state', 'capacity_subsets_state', 'parts_matched'], fit_inits=['hungarian'],
               fit_seeds=SEEDS, eval_split='dev', skip_use=True, state_reader='ledoit_wolf', whiten_bios=1024,
               whiten_tokens=60000, matched_gain_max=4., capacity_bins=64,
               purpose=('READER_COVERAGE extension: one-to-one and capacity controls under the reader-complete main '
                        'objective, and the classifier-loss one-to-one control with gains up to four.'))
    (configs / f'ext_gemma_controls_{key}.json').write_text(json.dumps(ext, indent=1))
    if key == 'w65k':
        smoke = dict(ext, run_id=ext['run_id'] + '_SMOKE', natural_bios=64, fit_bios=32, fit_steps=2, eval_bios=16,
                     limit_use=64, fit_seeds=[0], whiten_bios=32)
        (configs / 'ext_gemma_controls_smoke.json').write_text(json.dumps(smoke, indent=1))
    ext_readers = dict(readers, run_id=f'CSM_EXT_GEMMA2B_readers_{key}_v1_20261011',
                       map_runs=[dict(run=f'{M}/runs/cross_seed_models/{ext["run_id"]}', prefix='ext_',
                                      methods=[f'{f}_hungarian_s{s}' for s in SEEDS
                                               for f in ['matched_subset_state_sup', 'capacity_subset_state_sup', 'matched_part_sup']]),
                                 dict(run=state_runs['ledoit_wolf'], prefix='lw_',
                                      methods=[f'subset_state_sup_hungarian_s{s}' for s in SEEDS]),
                                 dict(run=c1, prefix='', methods=['hungarian_decoder'])],
                       purpose='READER_COVERAGE extension: held-out readers for the controls under the main objective.')
    (configs / f'ext_gemma_readers_{key}.json').write_text(json.dumps(ext_readers, indent=1))
    print(ext['run_id'], ext_readers['run_id'])
