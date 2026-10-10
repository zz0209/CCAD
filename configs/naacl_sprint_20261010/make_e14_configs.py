import json
from pathlib import Path

root = Path(r'E:\Projects\SAE_Lab\CCAD')
out = root / 'configs' / 'naacl_sprint_20261010'
bulk = 'D:/CCAD_Storage/runs/naacl_sprint_20261010'
# E14 repeats the Pythia-70M SHIFT training with two further training seeds per target dictionary. Everything except
# training_seed (request draws, fitting order and random subsets) equals the IR04 recipe, and evaluation uses the E1
# confirmation panel with the same methods as the reported runs.
for target in [2, 3, 4, 5]:
    train_base = json.loads((root / f'runs/IR04_shift_program_seed{target}_v1_20260916/config.resolved.json').read_text())
    eval_base = json.loads((out / f'e1_shift_r59panel_seed{target}.json').read_text())
    for training_seed in [43, 44]:
        programs = f'{bulk}/e14_programs/t{target}_ts{training_seed}'
        train = dict(train_base)
        train.update(
            run_id=f'NAACL_E14_shift_program_t{target}_ts{training_seed}_v1_20261010',
            run_parent='NAACL_SPRINT_20261010',
            run_storage_root=bulk,
            bulk_output_dir=programs,
            training_seed=training_seed,
            purpose='E14 training-seed replication of the Pythia-70M SHIFT correspondences.',
            scope='IR04 recipe with a different training seed. Same source information, dictionaries, steps and parameters.',
        )
        (out / f'e14_shift_program_t{target}_ts{training_seed}.json').write_text(json.dumps(train, indent=2))
        evaluation = dict(eval_base)
        evaluation.update(
            run_id=f'NAACL_E14_shift_eval_t{target}_ts{training_seed}_v1_20261010',
            adapted_programs={name: dict(directory=f'{programs}/{name}', sites=train_base['adapt_sites'])
                              for name in train_base['variants']},
            methods=['none', 'source', 'whole', 'random_parts', 'parts_relation', 'parts'],
            purpose='E14 evaluation of the replicated Pythia-70M SHIFT correspondences on the E1 confirmation panel.',
            scope='Frozen E14 checkpoints. No method or threshold is selected on this panel.',
        )
        (out / f'e14_shift_eval_t{target}_ts{training_seed}.json').write_text(json.dumps(evaluation, indent=2))
        print(target, training_seed)
