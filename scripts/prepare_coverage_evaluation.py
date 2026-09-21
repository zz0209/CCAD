import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action', action='store_true')
    args = parser.parse_args()
    training = json.loads(args.training.read_text())
    base = json.loads(Path(training['base_config']).read_text()) | training
    root = Path(base['run_storage_root'])/base['run_id']
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    if args.action:
        c = dict(base)
        c.pop('base_config')
        version = c['run_id'].split('_')[-2]
        c.update(run_id=f'RG06_action_evaluation_t{c["target_seed"]}_{version}_20260921',
                 generator_script='scripts/run_shift_transfer.py', budget_seconds=600,
                 purpose='Evaluate frozen local-action-trained target dictionaries and member relations.',
                 scope='Matched development documents and seven requests; adapted programs use target codes only.',
                 methods=['none', 'source', 'native', 'geometry_gain', 'raw',
                          *['local_'+v for v in training['action_variants']]],
                 adapted_programs={})
        manifest = json.loads(Path(c['source_manifest']).read_text())
        for variant in training['action_variants']:
            c['adapted_programs']['local_'+variant] = dict(
                directory=str(root/variant/f'step_{training["steps"]}'), sites=list(manifest['members']))
        path = args.output/'EVAL_ACTION.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(c, stream, indent=2)
        print(json.dumps(dict(configurations=[str(path)])))
        return
    for name in ['initial', *training['variants']]:
        c = dict(base)
        for key in ['base_config', 'relation_run', 'request_spec']:
            c.pop(key, None)
        c.update(run_id=f'RG06_coverage_transfer_{name}_t{c["target_seed"]}_v1_20260921',
                 generator_script='scripts/run_shift_transfer.py', budget_seconds=600,
                 purpose='Compare the same frozen-member functional fit after matched state-coverage training.',
                 scope='Exposed development biographies,seven part requests,one target; same relation fit across dictionary conditions.',
                 budget='cpu-heavy then gpu-0,600driver seconds,13GB VRAM,1GB output.',
                 fit_sequences=128, fit_batch_sequences=8, fit_iterations=1200,
                 functional_weight=.99, context_response=True, evaluation_split='dev',
                 eval_batch_size=8, development_per_group=training['development_per_group'],
                 methods=['none', 'source', 'native', 'geometry_gain', 'raw'],
                 dictionary_condition=name)
        if name != 'initial':
            c['target_directory'] = str(root/name/f'step_{training["steps"]}')
        path = args.output/f'EVAL_{name.upper()}.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(c, stream, indent=2)
        paths.append(str(path))
    print(json.dumps(dict(configurations=paths)))


if __name__ == '__main__':
    main()
