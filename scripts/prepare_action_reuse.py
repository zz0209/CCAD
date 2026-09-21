import json
from pathlib import Path


def main():
    root = Path('artifacts/reuse_generalization_20260921_round06')
    output = root/'replication'
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads((root/'EVAL_INITIAL.json').read_text())
    training = json.loads((root/'ACTION_DEVELOPMENT_V2.json').read_text())
    evaluation = json.loads((root/'action_v2/EVAL_ACTION.json').read_text())
    bulk = Path(base['run_storage_root'])
    sites = list(json.loads(Path(base['source_manifest']).read_text())['members'])
    queue = []

    def save(name, value):
        path = output/(name+'.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
        queue.append(dict(config=str(path), script=value['generator_script'], run_id=value['run_id']))

    for seed in [3, 4, 5]:
        initial = base | dict(target_seed=seed, seeds=[seed],
            run_id=f'RG06_action_initial_t{seed}_v1_20260921',
            evidence_level='target_replication_exposed_documents',
            purpose='Fit the same initial correspondence for the fixed local action recipe.',
            threshold_source_split='Source-only classification gradients weight the common initial relation; later local training uses no task outputs.')
        save(f'INITIAL_T{seed}', initial)
        train = base | training | dict(target_seed=seed, seeds=[seed],
            run_id=f'RG06_action_replication_t{seed}_v2_20260921',
            relation_run=str(bulk/initial['run_id']))
        train.pop('base_config')
        # 保留完整配置作为下一运行的唯一配置入口。
        train['base_config'] = str(output/f'INITIAL_T{seed}.json')
        save(f'TRAIN_T{seed}', train)
        ev = evaluation | dict(target_seed=seed, seeds=[seed],
            run_id=f'RG06_action_replication_evaluation_t{seed}_v2_20260921',
            relation_run=str(bulk/initial['run_id']),
            evidence_level='target_replication_exposed_documents',
            adapted_programs={f'local_{v}':dict(directory=str(bulk/train['run_id']/v/'step_512'),
                                              sites=sites) for v in ['whole','parts']})
        save(f'EVAL_T{seed}', ev)

    quality = json.loads(Path('configs/ir04_shift_material_v1.json').read_text())
    quality.update(run_id='RG06_action_natural_quality_t2_v1_20260921', run_parent='REUSE_GENERALIZATION_06',
        run_storage_root=str(bulk), sites=sites, seeds=[2],
        variants=['original','local_whole','local_parts'],
        purpose='Measure natural reconstruction and CE recovery after local action training.',
        scope='Sixty-four existing held natural sequences,eleven sites,one target;quality assessment only.',
        budget_seconds=600, evidence_level='development_quality',
        dictionary_directories={'original':base['target_directory'],
            **{f'local_{v}':str(bulk/training['run_id']/v/'step_512') for v in ['whole','parts']}})
    save('NATURAL_QUALITY', quality)

    consumer = json.loads(Path('configs/ir04_shift_consumer_seed2_v1.json').read_text())
    consumer.pop('confirmation_freeze')
    consumer.update(run_id='RG06_action_consumer_t2_v1_20260921', run_parent='REUSE_GENERALIZATION_06',
        run_storage_root=str(bulk), audit_opened=False,
        evidence_level='existing_consumer_development', budget_seconds=1500,
        purpose='Test local action supervision on four existing later classifiers with the published training recipe.',
        scope='Same existing later task cohort and source judgments;fixed-head fidelity and retrained-head use are separate outcomes.',
        methods=['none','source','native','geometry_gain','raw','parts','local_whole','local_parts'],
        feature_cache_runs=['runs/IR04_shift_consumer_seed2_v1_20260916'])
    consumer['adapted_programs']={'parts':consumer['adapted_programs']['parts'],
        **{f'local_{v}':dict(directory=str(bulk/training['run_id']/v/'step_512'), sites=sites)
           for v in ['whole','parts']}}
    save('CONSUMER', consumer)
    with (output/'QUEUE.json').open('x', encoding='utf-8') as stream:
        json.dump(queue, stream, indent=2)
    print(json.dumps(dict(queue=queue)))


if __name__ == '__main__':
    main()
