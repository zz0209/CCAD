import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
SEEDS = [0, 1, 2]
# Final held-out reader numbers on test biographies for the two development targets: every map frozen by the EXP-2
# decisions and the controls under the main objective, read exactly as on development biographies.
for key in ['w65k', 'l0low16k']:
    dev = json.loads((configs / f'exp2_gemma_readers_{key}.json').read_text())
    ext = f'{M}/runs/cross_seed_models/CSM_EXT_GEMMA2B_controls_{key}_v1_20261011'
    final = dict(dev, run_id=f'CSM_FINAL_GEMMA2B_readers_{key}_v1_20261011', eval_split='test',
                 map_runs=dev['map_runs'] + [dict(run=ext, prefix='ext_', methods=[
                     f'{f}_hungarian_s{s}' for s in SEEDS
                     for f in ['matched_subset_state_sup', 'capacity_subset_state_sup', 'matched_part_sup']])],
                 purpose=('READER_COVERAGE final numbers: held-out readers on test biographies for every map frozen by '
                          'the EXP-2 decisions and the controls under the main objective.'))
    (configs / f'final_gemma_readers_{key}.json').write_text(json.dumps(final, indent=1))
    print(final['run_id'])
