import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
OUT = ROOT / 'configs' / 'cross_seed_pilot'
DIRECTIONS = ['s1_t2', 's1_t3', 's1_t4', 's1_t5', 's2_t3', 's3_t4', 's4_t5', 's5_t1']
# EXP-U development readers for GPT-2 Medium: the reader setting of the earlier held-out runs (four BLiMP paradigms now
# counted as development data, development FineWeb documents), with U, S, the generic-text fit G, the explanation-pair
# fit P and Hungarian matching read in the same run so that every contrast is paired.
for d in DIRECTIONS:
    cfg = json.loads((RUNS / f'CSM_READERS_GPT2_{d}_v1_20261011' / 'config.resolved.json').read_text())
    keep = {e['name']: e for e in cfg['executors'] if e['name'] in ('hungarian_activation', 'membership_natural', 'rc_distparts')}
    u = str(RUNS / f'CSM_U_GPT2_{d}_v1_20261011' / 'maps.npz')
    cfg.update(run_id=f'CSM_UREADERS_GPT2_{d}_v1_20261011',
               executors=[keep['hungarian_activation'], keep['membership_natural'], keep['rc_distparts'],
                          dict(name='u_distmixed', maps=u, key='membership_distmixed', cap=1.),
                          dict(name='s_statemixed', maps=u, key='membership_statemixed', cap=1.)],
               purpose='READER_COVERAGE EXP-U: development readers for the unified GPT-2 arms against G and P.')
    (OUT / f'ureaders_gpt2_{d}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
