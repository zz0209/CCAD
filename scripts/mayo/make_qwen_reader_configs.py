import json
from pathlib import Path

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
RUNS = f'{M}/runs/cross_seed_models'
HELDOUT = ['irregular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2',
           'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause']
# READER_COVERAGE held-out readers for the six Qwen-Scope grammar runs: the same four held-out BLiMP paradigms and the
# Qwen tokenization of the held-out generic text, with the maps of C4 and the reader-complete maps of C8.
for direction in ['s1_t2', 's2_t1']:
    for seed in [0, 1, 2]:
        base = json.loads((configs / f'c4_qwen17_{direction}_ts{seed}.json').read_text())
        c4 = f'{RUNS}/CSM_C4_QWEN17_{direction}_ts{seed}_v1_20261011/maps.npz'
        c8 = f'{RUNS}/CSM_C8_QWEN17_rc_{direction}_ts{seed}_v1_20261011/maps.npz'
        cfg = dict(
            run_id=f'CSM_READERS_QWEN17_{direction}_ts{seed}_v1_20261011', run_root=RUNS, device='cuda:0',
            dictionary_source_dir=base['dictionary_source_dir'], dictionary_overlay_dir=base['dictionary_overlay_dir'],
            source=base['source'], target=base['target'], eos_token_id=base['eos_token_id'], sentence_prefix='\n',
            part_names=['verb', 'number', 'gender'], random_requests=16, random_seed=20261011,
            heldout_paradigms=[f'{M}/data/grammar/heldout/{p}.jsonl' for p in HELDOUT], pairs_per_paradigm=128,
            batch_pairs=16, natural_eval_tokens=base['natural_eval_tokens'], token_dtype=base['token_dtype'],
            context=base['context'], natural_docs=128, natural_batch=8,
            executors=[dict(name='hungarian_activation', maps=c4, key='hungarian_activation', cap=1.),
                       dict(name='membership_natural', maps=c4, key='membership_natural', cap=1.),
                       dict(name='membership_parts', maps=c4, key='membership_parts', cap=1.),
                       dict(name='matched_natural4', maps=c4, key='matched_natural', cap=None),
                       dict(name='matched_parts4', maps=c4, key='matched_parts', cap=None),
                       dict(name='rc_distparts', maps=c8, key='membership_distparts', cap=1.)],
            purpose='READER_COVERAGE: held-out BLiMP paradigms and generic text for the Qwen3-1.7B grammar correspondences.')
        (configs / f'readers_qwen17_{direction}_ts{seed}.json').write_text(json.dumps(cfg, indent=1))
        print(cfg['run_id'])
