import json
from pathlib import Path

from make_reader_configs import FEMALE, MALE

configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
RUNS = f'{M}/runs/cross_seed_models'
SEEDS = [42, 43, 44]
# READER_COVERAGE held-out readers for Pythia-70M SHIFT: every relation of C2, C2b, C2c and C2d, the R59 relations and
# Hungarian matching, read on the 128 development biographies that no relation was fitted on (the evaluation rows of
# the program runs) and on held-out R011 text.
gemma = json.loads((configs / 'c1_gemma_natural_w65k.json').read_text())
e8b = json.loads((configs / 'e8b_gemma_taskuse_w65k.json').read_text())
for target in [2, 3, 4, 5]:
    base = json.loads((configs / f'c2_pythia_program_t{target}_ts42.json').read_text())
    executors = [dict(name='native', relation=f'{base["relation_run"]}/relation.npz', key='native'),
                 dict(name='geometry_gain', relation=f'{base["relation_run"]}/relation.npz', key='geometry_gain'),
                 dict(name='hungarian_decoder', relation=base['matched_relation'], key='hungarian_decoder')]
    arms = [('c2', 'parts_relation', 'parts_relation'), ('c2', 'natural_relation', 'natural_relation'),
            ('c2', 'natural_matched', 'matched_rows_free'), ('c2b', 'parts_relation', 'parts_relation_cap4'),
            ('c2b', 'natural_relation', 'natural_relation_cap4'), ('c2b', 'natural_matched', 'one_to_one'),
            ('c2c', 'subset_relation', 'subset_relation'), ('c2d', 'parts_relation', 'parts_relation_state'),
            ('c2d', 'subset_relation', 'subset_relation_state')]
    for seed in SEEDS:
        executors += [dict(name=f'{label}_s{seed}', relation=f'{RUNS}/{lane}_programs/t{target}_ts{seed}/{variant}/relation.npz',
                           key='native') for lane, variant, label in arms]
    cfg = {k: base[k] for k in ['model_local_dir', 'source_manifest', 'notebook', 'source_parameters', 'target_directory',
                                'target_seed', 'frozen_source_run', 'dictionary_source_dir', 'dictionary_overlay_dir',
                                'natural_lm_eval_tokens', 'model_revision', 'source_revision']}
    cfg.update(run_id=f'CSM_EXP1_PYTHIA70M_readers_t{target}_v1_20261011', run_root=RUNS, seed=0,
               evaluation_membership=f'{RUNS}/CSM_C2_PYTHIA70M_program_t{target}_ts42_v1_20261011/evaluation_membership.json',
               executors=executors, dataset_dir=gemma['dataset_dir'], data_seed=gemma['data_seed'],
               later_pairs=e8b['later_pairs'], head_train_bios=4000, head_epochs=4, probe_seed=42, probe_lr=.01,
               probe_batch=128, batch=16, max_length=2048, random_requests=16, random_seed=20261011, natural_docs=128,
               female_words=FEMALE, male_words=MALE,
               purpose='READER_COVERAGE: held-out readers for every saved Pythia-70M SHIFT relation.')
    (configs / f'exp1_pythia_readers_t{target}.json').write_text(json.dumps(cfg, indent=1))
    if target == 2:
        smoke = dict(cfg, run_id=cfg['run_id'] + '_SMOKE', executors=executors[:4], random_requests=2, natural_docs=16,
                     head_train_bios=256, later_pairs=cfg['later_pairs'][:1])
        (configs / 'exp1_pythia_readers_smoke.json').write_text(json.dumps(smoke, indent=1))
    print(cfg['run_id'])
