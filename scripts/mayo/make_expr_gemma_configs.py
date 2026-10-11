import json
from pathlib import Path

from make_reader_configs import FEMALE, MALE

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
# EXP-R (research lead decision 4): on each development target, the explanation is rediscovered in the target dictionary
# by a fixed attribution and annotation procedure, then the rediscovered explanation, the transferred maps of the main
# method, Hungarian matching and the source edit run SHIFT with the E8b task design on test biographies.
for key in ['w65k', 'l0low16k']:
    state = json.loads((configs / f'exp2_gemma_state_{key}_ledoit_wolf.json').read_text())
    lw_run = f'{M}/runs/cross_seed_models/{state["run_id"]}'
    probe_run = json.loads((configs / f'exp2_gemma_readers_{key}.json').read_text())['probe_run']
    red = dict(state, run_id=f'CSM_REDISCOVER_GEMMA2B_{key}_v1_20261011', run_root=f'{M}/runs/cross_seed_models',
               probe_run=probe_run, count_run=lw_run, count_methods=[f'subset_state_sup_hungarian_s{s}' for s in [0, 1, 2]],
               batch=4, max_checked=2000, share=.5, pronoun_part='pronouns', description_part='female_descriptions',
               female_words=FEMALE, male_words=MALE,
               purpose=('READER_COVERAGE EXP-R: the SHIFT explanation rediscovered in the target dictionary by '
                        'attribution and fixed token-share annotation, as many features as the transferred maps use.'))
    for k in ['fit_families', 'fit_inits', 'fit_seeds', 'state_reader', 'whiten_bios', 'whiten_tokens']:
        red.pop(k)
    (configs / f'rediscover_gemma_{key}.json').write_text(json.dumps(red, indent=1))
    use = json.loads((configs / f'use_gemma_state_{key}.json').read_text())
    use.update(run_id=f'CSM_EXPR_GEMMA2B_{key}_v1_20261011', source_run=lw_run,
               methods=['source', 'hungarian_decoder'] + [f'subset_state_sup_hungarian_s{s}' for s in [0, 1, 2]] + ['rediscovered'],
               parts=['complete', 'pronouns', 'female_descriptions'],
               rediscovery=f'{M}/runs/cross_seed_models/{red["run_id"]}/REDISCOVERY.json',
               purpose=('READER_COVERAGE EXP-R: SHIFT on the target dictionary with the transferred explanation against '
                        'the rediscovered one, Hungarian matching and the source edit, for the complete deletion and '
                        'each part.'))
    (configs / f'expr_gemma_use_{key}.json').write_text(json.dumps(use, indent=1))
    print(red['run_id'], use['run_id'])
    # Smoke pair on the 16k target: a short rediscovery and a truncated use run of it.
    if key == 'l0low16k':
        red_smoke = dict(red, run_id=red['run_id'] + '_SMOKE', fit_bios=8, max_checked=100)
        (configs / 'rediscover_gemma_smoke.json').write_text(json.dumps(red_smoke, indent=1))
        use_smoke = dict(use, run_id=use['run_id'] + '_SMOKE', limit_task=32, methods=['source', 'rediscovered'],
                         rediscovery=f'{M}/runs/cross_seed_models/{red_smoke["run_id"]}/REDISCOVERY.json')
        (configs / 'expr_gemma_use_smoke.json').write_text(json.dumps(use_smoke, indent=1))
