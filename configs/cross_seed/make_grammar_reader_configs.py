import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'cross_seed' / 'readers'
out.mkdir(parents=True, exist_ok=True)
runs = Path('D:/CCAD_Storage/runs/cross_seed_models')
blimp = root / 'artifacts/scientific_reform_20260913/reference_material/blimp/3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
HELDOUT = ['irregular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2',
           'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause']
# READER_COVERAGE held-out readers for the eight GPT-2 Medium grammar directions, run on the local GPU: BLiMP paradigms
# that no correspondence or explanation selection used, and held-out generic text.
for source, target in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5), (5, 1)]:
    base_run = f'CSM_X1_gpt2_s{source}_t{target}_v1_20261010' if source == 1 else f'CSM_X1C_gpt2_s{source}_t{target}_v2_20261010'
    base = json.loads((root / 'configs/cross_seed' / (f'x1_gpt2_t{target}.json' if source == 1 else f'x1c_gpt2_s{source}_t{target}.json')).read_text())
    c7 = runs / f'CSM_C7_GPT2_matched4_s{source}_t{target}_v1_20261011' / 'maps.npz'
    c8 = runs / f'CSM_C8_GPT2_rc_s{source}_t{target}_v1_20261011' / 'maps.npz'
    x = runs / base_run / 'maps.npz'
    cfg = dict(
        run_id=f'CSM_READERS_GPT2_s{source}_t{target}_v1_20261011', run_root=str(runs), device='cuda:0',
        dictionary_source_dir=base['dictionary_source_dir'], dictionary_overlay_dir=base['dictionary_overlay_dir'],
        source=base['source'], target=base['target'], eos_token_id=base['eos_token_id'], sentence_prefix='',
        part_names=['verb', 'number', 'gender'], random_requests=16, random_seed=20261011,
        heldout_paradigms=[str(blimp / f'{p}.jsonl') for p in HELDOUT], pairs_per_paradigm=128, batch_pairs=16,
        natural_eval_tokens=base['natural_eval_tokens'], context=base['context'], natural_docs=128, natural_batch=8,
        executors=[dict(name='hungarian_activation', maps=str(x), key='hungarian_activation', cap=1.),
                   dict(name='membership_natural', maps=str(x), key='membership_natural', cap=1.),
                   dict(name='membership_parts', maps=str(x), key='membership_parts', cap=1.),
                   dict(name='matched_natural', maps=str(x), key='matched_natural', cap=1.),
                   dict(name='matched_natural4', maps=str(c7), key='matched_natural', cap=None),
                   dict(name='matched_parts4', maps=str(c7), key='matched_parts', cap=None),
                   dict(name='rc_distparts', maps=str(c8), key='membership_distparts', cap=1.)],
        purpose='READER_COVERAGE: held-out BLiMP paradigms and generic text for the GPT-2 Medium grammar correspondences.')
    (out / f'readers_gpt2_s{source}_t{target}.json').write_text(json.dumps(cfg, indent=1))
    if (source, target) == (1, 2):
        smoke = dict(cfg, run_id=cfg['run_id'] + '_SMOKE', pairs_per_paradigm=8, natural_docs=8, random_requests=2)
        (out / 'readers_gpt2_smoke.json').write_text(json.dumps(smoke, indent=1))
    print(cfg['run_id'])
