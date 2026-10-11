import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
SEEDS = [0, 1]
# EXP-U for Gemma-2-2B SHIFT (research lead decision 4): the unified arm fits the Ledoit-Wolf whitened layer-22 states
# on biographies with parts, the complete request and random subsets, alternating with generic FineWeb text with random
# subsets, 400 steps each. Everything else is the EXP-2 state-fit setting. Development readers compare it with the
# biography-only state fit RC on development biographies.
for key in ['w65k', 'l0low16k']:
    base = json.loads((configs / f'exp2_gemma_state_{key}_ledoit_wolf.json').read_text())
    cfg = dict(base, run_id=f'CSM_U_GEMMA2B_{key}_v1_20261011', fit_families=['subsets_statemixed'], fit_seeds=SEEDS,
               full_model=True, natural_lm_texts=f'{M}/data/cross_seed/natural_fineweb_slice.json', natural_lm_len=128,
               natural_lm_batch=4, natural_steps=512, natural_eval_seed=1,
               purpose=('READER_COVERAGE EXP-U: the unified rule for the Gemma SHIFT explanation, whitened layer-22 states '
                        'on biographies and generic text alternately, 400 steps each.'))
    (configs / f'u_gemma_{key}.json').write_text(json.dumps(cfg, indent=1))
    if key == 'w65k':
        smoke = dict(cfg, run_id=cfg['run_id'] + '_SMOKE', natural_bios=64, fit_bios=32, fit_steps=2, eval_bios=16,
                     limit_use=64, fit_seeds=[0], whiten_bios=32)
        (configs / 'u_gemma_smoke.json').write_text(json.dumps(smoke, indent=1))
    readers = json.loads((configs / f'exp2_gemma_readers_{key}.json').read_text())
    lw = next(r for r in readers['map_runs'] if r['prefix'] == 'lw_')['run']
    c1 = readers['probe_run']
    readers.update(run_id=f'CSM_U_GEMMA2B_readers_{key}_v1_20261011',
                   map_runs=[dict(run=f'{M}/runs/cross_seed_models/{cfg["run_id"]}', prefix='u_',
                                  methods=[f'subset_statemixed_sup_hungarian_s{s}' for s in SEEDS]),
                             dict(run=lw, prefix='lw_', methods=[f'subset_state_sup_hungarian_s{s}' for s in [0, 1, 2]]),
                             dict(run=c1, prefix='', methods=['hungarian_decoder'] +
                                  [f'{f}_hungarian_s{s}' for s in [0, 1, 2] for f in ['natural_sup', 'part_sup']])],
                   purpose='READER_COVERAGE EXP-U: development readers for the unified Gemma arm against RC.')
    (configs / f'u_gemma_readers_{key}.json').write_text(json.dumps(readers, indent=1))
    print(cfg['run_id'], readers['run_id'])
