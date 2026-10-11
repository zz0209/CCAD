import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
configs = Path(__file__).resolve().parent / 'configs'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
P = f'{M}/data/pythia_shift'
runs = f'{M}/runs/cross_seed_models'
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
# CONTAINMENT experiment 2 on the published Pythia-70M SHIFT explanation and the four independently trained target
# dictionary stacks: the task-free natural-text family (many-to-many), its one-to-one control with gains up to four,
# and part supervision, all with the target dictionaries frozen, three training seeds each. Natural text comes from
# the R011 FineWeb documents, which the R14 corpus of the target dictionaries excludes by construction.
paths = dict(source_revision=revision, run_parent='CONTAINMENT', run_storage_root=runs, device='cuda:0',
             dictionary_source_dir=f'{M}/code/dictionary_learning_60ec6bf',
             dictionary_overlay_dir=f'{M}/code/dictionary_learning_60ec6bf',
             model_local_dir=f'{M}/assets/pythia-70m-deduped', source_manifest=f'{P}/PUBLISHED_EXPLANATION.json',
             notebook=f'{P}/bib_shift.ipynb', source_parameters=f'{P}/published_shift55.npz',
             frozen_source_run=f'{P}/REFORM_R58_shift_source_development_v2_20260915',
             target_directory=f'{P}/step_8192', paired_tokens=f'{M}/data/grammar/discovery.uint16.bin')
variants = ['parts_relation', 'natural_relation', 'natural_matched']
for target in [2, 3, 4, 5]:
    train_base = json.loads((root / f'runs/IR04_shift_program_seed{target}_v1_20260916/config.resolved.json').read_text())
    eval_base = json.loads((root / f'configs/naacl_sprint_20261010/e1_shift_r59panel_seed{target}.json').read_text())
    relation_run = f'{P}/REFORM_R59_shift_confirm_seed{target}_v1_20260915'
    matching = f'{P}/matching_relations_seed{target}/matching_relations.npz'
    programs = {}
    for training_seed in [42, 43, 44]:
        bulk = f'{runs}/c2_programs/t{target}_ts{training_seed}'
        cfg = dict(train_base, **paths)
        cfg.update(run_id=f'CSM_C2_PYTHIA70M_program_t{target}_ts{training_seed}_v1_20261011', bulk_output_dir=bulk,
                   relation_run=relation_run, matched_relation=matching, training_seed=training_seed, variants=variants,
                   natural_lm_tokens=f'{P}/r011_train.uint16.bin', natural_lm_eval_tokens=f'{P}/r011_validation.uint16.bin',
                   natural_eval_sequences=256, natural_eval_seed=1, matched_gain_max=4., budget_seconds=5400,
                   budget='5400 driver seconds on one Mayo GPU for three 512-step relation-only arms.',
                   purpose=('CONTAINMENT experiment 2: a correspondence trained only on generic natural text with random '
                            'feature subsets, against part supervision and a one-to-one control with gains up to four, '
                            'for the published Pythia-70M SHIFT explanation, with the target dictionaries frozen.'),
                   scope=('IR04 recipe for part supervision; the natural arms replace the source-use contexts and part '
                          'requests with R011 FineWeb sequences and random subsets. Every map is also scored on 256 '
                          'held-out R011 validation sequences.'))
        (configs / f'c2_pythia_program_t{target}_ts{training_seed}.json').write_text(json.dumps(cfg, indent=1))
        programs.update({f'{v}_ts{training_seed}': dict(directory=f'{bulk}/{v}', sites=train_base['adapt_sites'],
                                                         gain_max=4. if v == 'natural_matched' else 1.) for v in variants})
    evaluation = dict(eval_base, **paths)
    evaluation.update(run_id=f'CSM_C2_PYTHIA70M_eval_t{target}_v1_20261011', relation_run=relation_run,
                      evaluation_panel=f'{P}/R59_CONFIRMATION_PANEL.json', adapted_programs=programs,
                      extra_relations={'hungarian_decoder': matching},
                      methods=['none', 'source', 'hungarian_decoder', 'geometry_gain', 'native', 'raw_reconstruction']
                      + list(programs), budget_seconds=5400,
                      budget='5400 driver seconds on one Mayo GPU, no fitting.',
                      purpose='CONTAINMENT experiment 2 evaluation on complete and part deletions of the SHIFT explanation.',
                      scope='Frozen C2 checkpoints and natural-text matching relations. No method or threshold is selected on this panel.')
    (configs / f'c2_pythia_eval_t{target}.json').write_text(json.dumps(evaluation, indent=1))
    print(target)
# C2b: the same many-to-many arms with each target feature's total deletion allowed up to four, the bound of the
# one-to-one control, to separate the number of target features from the size of the edit; and the one-to-one control
# itself, which keeps only each Hungarian match (the version 1 arm also trained the other entries of the matched rows).
for target in [2, 3, 4, 5]:
    programs = {}
    for training_seed in [42, 43, 44]:
        cfg = json.loads((configs / f'c2_pythia_program_t{target}_ts{training_seed}.json').read_text())
        bulk = f'{runs}/c2b_programs/t{target}_ts{training_seed}'
        cfg.update(run_id=f'CSM_C2B_PYTHIA70M_program_t{target}_ts{training_seed}_v1_20261011', bulk_output_dir=bulk,
                   variants=['parts_relation', 'natural_relation', 'natural_matched'], relation_capacity=4.,
                   purpose=('CONTAINMENT experiment 2b: part-supervised and natural-text many-to-many correspondences '
                            'whose total deletion per target feature may reach four, and the one-to-one control with '
                            'gains up to four on the Hungarian matches.'))
        (configs / f'c2b_pythia_program_t{target}_ts{training_seed}.json').write_text(json.dumps(cfg, indent=1))
        names = dict(parts_relation='parts_relation_cap4', natural_relation='natural_relation_cap4', natural_matched='one_to_one')
        programs.update({f'{names[v]}_ts{training_seed}': dict(directory=f'{bulk}/{v}', sites=cfg['adapt_sites'], gain_max=4.)
                         for v in cfg['variants']})
        # C2c: annotation-free random feature subsets on the unlabeled source-use contexts, deletion bound one as in
        # part supervision, the main method of rule D1 when only the grammar families pass.
        sub = json.loads((configs / f'c2_pythia_program_t{target}_ts{training_seed}.json').read_text())
        sub_bulk = f'{runs}/c2c_programs/t{target}_ts{training_seed}'
        sub.update(run_id=f'CSM_C2C_PYTHIA70M_program_t{target}_ts{training_seed}_v1_20261011', bulk_output_dir=sub_bulk,
                   variants=['subset_relation'],
                   purpose=('CONTAINMENT experiment 2c: a many-to-many correspondence trained with random feature subsets '
                            'on unlabeled source-use contexts, without part annotation.'))
        (configs / f'c2c_pythia_program_t{target}_ts{training_seed}.json').write_text(json.dumps(sub, indent=1))
        programs[f'subset_relation_ts{training_seed}'] = dict(directory=f'{sub_bulk}/subset_relation', sites=sub['adapt_sites'])
    evaluation = json.loads((configs / f'c2_pythia_eval_t{target}.json').read_text())
    evaluation.update(run_id=f'CSM_C2B_PYTHIA70M_eval_t{target}_v1_20261011', adapted_programs=programs,
                      methods=['none', 'source'] + list(programs),
                      purpose='CONTAINMENT experiment 2b evaluation on complete and part deletions of the SHIFT explanation.')
    (configs / f'c2b_pythia_eval_t{target}.json').write_text(json.dumps(evaluation, indent=1))
smoke = json.loads((configs / 'c2_pythia_program_t2_ts42.json').read_text())
smoke.update(run_id=smoke['run_id'] + '_SMOKE', bulk_output_dir=smoke['bulk_output_dir'] + '_smoke', steps=4,
             natural_eval_sequences=8, development_per_group=2)
(configs / 'c2_pythia_program_smoke.json').write_text(json.dumps(smoke, indent=1))
smoke_eval = json.loads((configs / 'c2_pythia_eval_t2.json').read_text())
smoke_eval.update(run_id=smoke_eval['run_id'] + '_SMOKE', adapted_programs={
    f'{v}_ts42': dict(directory=f"{smoke['bulk_output_dir']}/{v}", sites=smoke['adapt_sites'],
                      gain_max=4. if v == 'natural_matched' else 1.) for v in variants},
    methods=['none', 'source', 'hungarian_decoder'] + [f'{v}_ts42' for v in variants], queries=['full', 'pronouns'])
(configs / 'c2_pythia_eval_smoke.json').write_text(json.dumps(smoke_eval, indent=1))
