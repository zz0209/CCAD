import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
# Task use of the main method: the state-fit maps of EXP-2 delete each named part before the SHIFT classifier and four
# later classifiers are evaluated and retrained, with the E8b design, test biographies and tasks unchanged, so the rows
# join the E8b table of the other correspondences.
for key in ['w65k', 'l0low16k']:
    e8b = json.loads((configs / f'e8b_gemma_taskuse_{key}.json').read_text())
    cfg = dict(e8b, run_id=f'CSM_USE_GEMMA2B_state_{key}_v1_20261011', run_root=f'{M}/runs/cross_seed_models',
               source_run=f'{M}/runs/cross_seed_models/CSM_EXP2_GEMMA2B_state_{key}_ledoit_wolf_v1_20261011',
               methods=['source', 'readout', 'hungarian_decoder'] + [f'subset_state_sup_hungarian_s{s}' for s in [0, 1, 2]],
               purpose=('READER_COVERAGE use: deleting each transferred part with the state-fit maps before evaluating and '
                        'retraining the SHIFT classifier and four later classifiers, joined to the E8b comparison.'))
    (configs / f'use_gemma_state_{key}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
