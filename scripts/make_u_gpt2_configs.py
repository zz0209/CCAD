import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs' / 'cross_seed_pilot'
# EXP-U for GPT-2 Medium (research lead decision 4): the unified rule with the full next-token distribution as reader
# (U, membership_distmixed) and with the whitened hook state as reader (S, membership_statemixed). Natural-text and
# task steps alternate, 512 of each, so every input type gets the steps of its dedicated fit. All other settings are
# those of the P1 configs, which carry the X1 and X1c settings with local paths.
for p1 in sorted(OUT.glob('p1_gpt2_distmixed_*.json')):
    direction = p1.stem.split('distmixed_')[1]
    cfg = json.loads(p1.read_text())
    cfg.update(run_id=f'CSM_U_GPT2_{direction}_v1_20261011', families=['membership_distmixed', 'membership_statemixed'],
               steps=1024, whiten_sequences=128,
               purpose=('READER_COVERAGE EXP-U: the unified fitting rule for the grammatical explanations, read through '
                        'the full next-token distribution (U) or the whitened hook state (S), on natural text and the '
                        'explanation pairs alternately with 512 steps each.'))
    (OUT / f'u_gpt2_{direction}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
