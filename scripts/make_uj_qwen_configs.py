import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAYO_CONFIGS = ROOT / 'scripts' / 'mayo' / 'configs'
OUT = ROOT / 'configs' / 'cross_seed_pilot'
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
MIRROR = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010/mayo_mirror/qwen_local')
BLIMP = ROOT / 'artifacts' / 'scientific_reform_20260913' / 'reference_material' / 'blimp' / '3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
GPT2_LOCAL = json.loads((RUNS / 'CSM_X1_gpt2_s1_t2_v1_20261010' / 'config.resolved.json').read_text())
# Mayo inputs and their local copies (SHA256 checked against Mayo for the model, the SAEs and the held-out paradigms).
PREFIXES = [(f'{M}/assets/qwen3-1.7b-base', MIRROR / 'qwen3-1.7b-base'),
            (f'{M}/runs/naacl_sprint_20261010/qwen17_prepare_l14_v2', MIRROR / 'qwen17_prepare_l14_v2'),
            (f'{M}/runs/naacl_sprint_20261010/qwen17_fresh_panel_rg09', MIRROR / 'qwen17_fresh_panel_rg09'),
            (f'{M}/data/cross_seed', MIRROR),
            (f'{M}/data/grammar/heldout', BLIMP),
            (f'{M}/runs/cross_seed_models', MIRROR / 'maps'),
            (f'{M}/code/dictionary_learning_60ec6bf', Path(GPT2_LOCAL['dictionary_source_dir']))]


def local(value):
    if isinstance(value, dict):
        return {k: local(v) for k, v in value.items()}
    if isinstance(value, list):
        return [local(v) for v in value]
    if isinstance(value, str) and value.startswith(M):
        for prefix, path in PREFIXES:
            if value.startswith(prefix):
                return str(path / value[len(prefix) + 1:]) if len(value) > len(prefix) else str(path)
        raise ValueError(value)
    return value


# EXP-U for Qwen3-1.7B with Qwen-Scope: the joint unified rule on the local GPU, with the C8 settings of each direction
# and training seed, and development readers of U, S, G (task-free natural-text map of C4), P (C8) and Hungarian.
for direction in ['s1_t2', 's2_t1']:
    for seed in [0, 1, 2]:
        base = local(json.loads((MAYO_CONFIGS / f'c8_qwen17_rc_{direction}_ts{seed}.json').read_text()))
        base.update(dictionary_source_dir=GPT2_LOCAL['dictionary_source_dir'], dictionary_overlay_dir=GPT2_LOCAL['dictionary_overlay_dir'],
                    run_storage_root=str(RUNS))
        cfg = dict(base, run_id=f'CSM_UJ_QWEN17_{direction}_ts{seed}_v1_20261011',
                   families=['membership_distjoint', 'membership_statejoint'], steps=512, whiten_sequences=128,
                   budget='Local gpu-0 lease, one job per run.',
                   purpose=('READER_COVERAGE EXP-U: the unified rule as one objective for the Qwen-Scope grammatical '
                            'explanations, read through the full next-token distribution (U) or the whitened hook state (S).'))
        (OUT / f'uj_qwen17_{direction}_ts{seed}.json').write_text(json.dumps(cfg, indent=1))
        readers = local(json.loads((MAYO_CONFIGS / f'readers_qwen17_{direction}_ts{seed}.json').read_text()))
        # Half the Mayo batches: full-vocabulary log-probabilities of five executors overflow the 16 GB local GPU.
        readers.update(dictionary_source_dir=GPT2_LOCAL['dictionary_source_dir'], dictionary_overlay_dir=GPT2_LOCAL['dictionary_overlay_dir'],
                       run_root=str(RUNS), device='cuda:0', batch_pairs=8, natural_batch=4)
        keep = {e['name']: e for e in readers['executors'] if e['name'] in ('hungarian_activation', 'membership_natural', 'rc_distparts')}
        u = str(RUNS / cfg['run_id'] / 'maps.npz')
        readers.update(run_id=f'CSM_UREADERS_QWEN17_{direction}_ts{seed}_v1_20261011',
                       executors=[keep['hungarian_activation'], keep['membership_natural'], keep['rc_distparts'],
                                  dict(name='u_distjoint', maps=u, key='membership_distjoint', cap=1.),
                                  dict(name='s_statejoint', maps=u, key='membership_statejoint', cap=1.)],
                       purpose='READER_COVERAGE EXP-U: development readers for the unified Qwen arms against G and P.')
        (OUT / f'ureaders_qwen17_{direction}_ts{seed}.json').write_text(json.dumps(readers, indent=1))
        print(cfg['run_id'], readers['run_id'])
