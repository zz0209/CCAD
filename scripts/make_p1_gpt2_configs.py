import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
OUT = ROOT / 'configs' / 'cross_seed_pilot'
# Pilot P1: one objective for the language-model consumer of the grammatical explanations. Generic-text random subsets
# and the explanation's parts and random subsets alternate, both read through the full next-token distribution. Every
# other setting is the X1 or X1c setting of the same direction, so the pilot is compared on the explanation's own
# confirmation pairs and on held-out generic text with the existing fits; the four held-out BLiMP paradigms stay unread.
SOURCES = {'s1_t2': 'CSM_X1_gpt2_s1_t2_v1_20261010', 's1_t3': 'CSM_X1_gpt2_s1_t3_v1_20261010',
           's1_t4': 'CSM_X1_gpt2_s1_t4_v1_20261010', 's1_t5': 'CSM_X1_gpt2_s1_t5_v1_20261010',
           's2_t3': 'CSM_X1C_gpt2_s2_t3_v2_20261010', 's3_t4': 'CSM_X1C_gpt2_s3_t4_v2_20261010',
           's4_t5': 'CSM_X1C_gpt2_s4_t5_v2_20261010', 's5_t1': 'CSM_X1C_gpt2_s5_t1_v2_20261010'}
OUT.mkdir(parents=True, exist_ok=True)
# The X1c runs ran on Mayo; their inputs have the same SHA256 as these local files, whose paths come from the local X1
# run and the local held-out reader run of the same direction.
LOCAL = json.loads((RUNS / SOURCES['s1_t2'] / 'config.resolved.json').read_text())
PATH_KEYS = ['dictionary_source_dir', 'dictionary_overlay_dir', 'natural_tokens', 'natural_eval_tokens', 'panel',
             'evaluation_panel']
for direction, run in SOURCES.items():
    cfg = json.loads((RUNS / run / 'config.resolved.json').read_text())
    if run.startswith('CSM_X1C'):
        readers = json.loads((RUNS / f'CSM_READERS_GPT2_{direction}_v1_20261011' / 'config.resolved.json').read_text())
        cfg.update({k: LOCAL[k] for k in PATH_KEYS})
        cfg['source'] = dict(cfg['source'], **{k: readers['source'][k] for k in ['model_dir', 'sae_checkpoint', 'gate']})
        cfg['target'] = dict(cfg['target'], sae_checkpoint=readers['target']['sae_checkpoint'])
        assert cfg['source']['sae_label'] == readers['source']['sae_label'] and cfg['target']['sae_label'] == readers['target']['sae_label']
    cfg.update(run_id=f'CSM_P1_GPT2_distmixed_{direction}_v1_20261011', run_parent='READER_COVERAGE',
               run_storage_root=str(RUNS), families=['membership_distmixed'], budget='Local gpu-0 lease, one job per run.',
               purpose=('READER_COVERAGE pilot P1: generic-text random subsets alternating with the explanation parts and '
                        'random subsets, both read through the full next-token distribution, as one objective for the '
                        'language-model consumer.'))
    (OUT / f'p1_gpt2_distmixed_{direction}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'])
