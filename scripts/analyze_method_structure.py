import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def parameters(config, run):
    if 'adapt_sites' in config:
        paths = [(Path(config['target_directory']) / f'{site}_seed{config["target_seed"]}.pt',
                  Path(config['bulk_output_dir']) / 'tangent_mixed' / f'{site}_seed{config["target_seed"]}.pt')
                 for site in config['adapt_sites']]
    else:
        paths = [(Path(config['target_checkpoint']), run / 'program_step512.pt')]
    encoder_count, total_count = 0, 0
    for original_path, fitted_path in paths:
        original = torch.load(original_path, map_location='cpu', weights_only=True)
        fitted = torch.load(fitted_path, map_location='cpu', weights_only=True)
        if 'dictionary' in fitted:
            fitted = fitted['dictionary']
        for key in ['decoder.weight', 'b_dec']:
            assert torch.equal(original[key], fitted[key]), (fitted_path, key)
        encoder_count += sum(original[key].numel() for key in ['encoder.weight', 'encoder.bias'])
        total_count += sum(original[key].numel() for key in ['encoder.weight', 'encoder.bias', 'decoder.weight', 'b_dec'])
    return dict(trained_encoder_parameters=encoder_count, original_joint_parameters=total_count,
                decoder_and_center_bit_equal=True)


def human(run, repetitions):
    config = json.loads((run / 'config.resolved.json').read_text())
    rows = json.loads((run / 'evaluation_membership.json').read_text())['rows']
    queries = config['queries']
    old = Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE03_shift_execution_reform_v1_20260919')
    assert rows == json.loads((old / 'evaluation_membership.json').read_text())['rows']
    assert json.loads((run / 'program_context_membership.json').read_text()) == json.loads((old / 'program_context_membership.json').read_text())
    assert json.loads((run / 'loss_scales.json').read_text()) == json.loads((old / 'loss_scales.json').read_text())
    clean = np.load(run / 'none__full__pooled.npy').astype(float)
    source = np.stack([np.load(run / f'source__{q}__pooled.npy').astype(float) for q in queries])
    assert np.array_equal(clean, np.load(old / 'none__full__pooled.npy'))
    for i, query in enumerate(queries):
        assert np.array_equal(source[i], np.load(old / f'source__{query}__pooled.npy'))
        assert np.array_equal(np.load(run / f'joint__{query}__pooled.npy'),
                              np.load(old / f'tangent_mixed__{query}__pooled.npy'))
    methods = ['input_initial', 'joint', 'encoding_hybrid', 'decoding_hybrid', 'tangent_mixed', 'raw_reconstruction']
    target = np.stack([np.stack([np.load(run / f'{method}__{q}__pooled.npy').astype(float)
                                for q in queries]) for method in methods])
    tasks = ['composer_surgeon_orientation0', 'composer_surgeon_orientation1',
             'model_software_engineer_orientation0', 'model_software_engineer_orientation1']
    heads = [np.load(ROOT / 'runs/IR04_shift_consumer_seed2_v1_20260916' / f'none__full__{task}__probe42.npz') for task in tasks]
    weights = np.stack([head['weight'].ravel() for head in heads], 1)
    differences = (target - source) @ weights
    effects = (source - clean) @ weights
    assert np.isfinite(differences).all()
    families = dict(full=[queries.index('full')], parts=[queries.index(q) for q in ['pronouns', 'names', 'associated_words']],
                    untrained=[i for i, q in enumerate(queries) if q.startswith(('interior', 'boundary'))])
    professions = np.array([row['profession'] for row in rows])
    cells = {(p, g): np.array([i for i, row in enumerate(rows) if row['profession'] == p and row['gender'] == g])
             for p, g in sorted({(row['profession'], row['gender']) for row in rows})}

    def measure(indices):
        values = np.empty((len(methods), len(families)))
        for fi, qi in enumerate(families.values()):
            per_head = []
            for hi, pair in enumerate([(5, 25), (5, 25), (12, 24), (12, 24)]):
                selected = indices[np.isin(professions[indices], pair)]
                denominator = np.square(effects[np.ix_(qi, selected, [hi])]).mean(1).ravel()
                assert (denominator > 1e-12).all()
                residual = np.take(np.take(differences[..., hi], qi, axis=1), selected, axis=2)
                per_head.append(np.sqrt(np.square(residual).mean(2) / denominator).mean(1))
            values[:, fi] = np.array(per_head).mean(0)
        return values

    point = measure(np.arange(len(rows)))
    rng = np.random.default_rng(92212)
    boot = np.stack([measure(np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in cells.values()]))
                     for _ in range(repetitions)])
    summary = {method: {family: float(point[mi, fi]) for fi, family in enumerate(families)}
               for mi, method in enumerate(methods)}
    focal = methods.index('tangent_mixed')
    comparisons = {method: {family: dict(mean=float(point[focal, fi] - point[mi, fi]),
        ci=np.quantile(boot[:, focal, fi] - boot[:, mi, fi], [.025, .975]).tolist())
        for fi, family in enumerate(families)} for mi, method in enumerate(methods) if mi != focal}
    quality = {}
    for name, directory in [('encoder_training', run), ('joint_training', old)]:
        records = [json.loads(line) for line in (directory / 'metrics.raw.jsonl').read_text().splitlines()]
        quality[name] = [record for record in records if record.get('kind') == 'quality' and record['method'] == 'tangent_mixed']
    return dict(summary=summary, encoder_training_minus=comparisons, contexts=len(rows), quality=quality,
        inference='Paired document bootstrap stratified by profession and gender, fixed source, target, heads and requests; exposed development.',
        source_and_joint_replay_bit_equal=True, training_membership_equal=True,
        parameters=parameters(config, run))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-run', type=Path, required=True)
    parser.add_argument('--grammar-analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert not args.output.exists()
    assert json.loads((args.human_run / 'status.json').read_text())['status'] == 'PASS'
    grammar = None
    if args.grammar_analysis:
        grammar = json.loads(args.grammar_analysis.read_text())
        assert len(grammar['inputs']) == 1
        grammar_run = Path(grammar['inputs'][0]['run'])
        grammar_config = json.loads((grammar_run / 'config.resolved.json').read_text())
        grammar['parameters'] = parameters(grammar_config, grammar_run)
        grammar['inference'] = ('Paired sentence-pair bootstrap stratified by task, fixed source and target, '
                                'heads and requests; exposed development. The single target provides '
                                'no estimate of between-target variation.')
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        human=human(args.human_run, args.bootstrap), grammar=grammar,
        human_run=args.human_run.as_posix(), bootstrap=args.bootstrap,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(human=result['human']['summary'],
        grammar={m: v['primary']['mean'] for m, v in grammar['summary'].items()} if grammar else None), indent=2))


if __name__ == '__main__':
    main()
