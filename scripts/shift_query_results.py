"""Paired development analysis of source-defined parts and their unfitted unions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PARTS = ['pronouns', 'names', 'associated_words']
UNIONS = ['pronouns+names', 'pronouns+associated_words', 'names+associated_words']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_logits(run, order):
    values = {}
    for line in (run / 'metrics.raw.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['kind'] != 'classification':
            continue
        key = row['method'] + '/' + row['operation']
        arr = values.setdefault(key, np.full(len(order), np.nan))
        index = order[row['component']]
        assert np.isnan(arr[index]), (key, index)
        arr[index] = row['logit']
    assert all(np.isfinite(v).all() for v in values.values())
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='REFORM_R59_shift_query_granularity_v1_20260915')
    parser.add_argument('--export-paper', action='store_true')
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    assert read(run / 'status.json')['status'] == 'PASS'
    assert read(run / 'contract_validation.json')['ok']
    cfg = read(run / 'config.resolved.json')
    source = Path(cfg['frozen_source_run'])
    panel = [r for r in read(source / 'panel.json')['rows'] if r['split'] == 'dev']
    order = {r['document_sha256']: i for i, r in enumerate(panel)}
    assert len(panel) == len(order) == 695
    y = np.array([r['label'] for r in panel])
    gender = np.array([r['gender'] for r in panel])
    values = load_logits(run, order)
    base = np.load(source / 'dev_none.npz')['logits']
    strata = [np.flatnonzero((y == label) & (gender == g)) for label in (0, 1) for g in (0, 1)]
    rng = np.random.default_rng(59515)
    counts = np.zeros((2000, len(panel)), np.float64)
    for ix in strata:
        samples = rng.choice(ix, size=(len(counts), len(ix)), replace=True)
        for i, sample in enumerate(samples):
            counts[i] += np.bincount(sample, minlength=len(panel))
    weights = counts / len(panel)
    summaries, cells, bootstrap = {}, {}, {}
    sigmoid = lambda a: 1 / (1 + np.exp(-np.clip(a, -60, 60)))
    interval = lambda a: np.quantile(a, [.025, .975]).tolist()
    for method in cfg['methods']:
        bootstrap[method] = {}
        for query in cfg['queries']:
            v = values[method + '/' + query]
            ref = values['source/' + query]
            changed = (ref > 0) != (base > 0)
            assert 0 < changed.sum() < len(panel)
            agree = (v > 0) == (ref > 0)
            agreement = .5 * (agree[changed].mean() + agree[~changed].mean())
            bs = .5 * ((counts @ (agree & changed)) / (counts @ changed) +
                       (counts @ (agree & ~changed)) / (counts @ ~changed))
            error = np.abs(sigmoid(v) - sigmoid(ref))
            correct = (v > 0) == y
            group_acc = np.array([correct[ix].mean() for ix in strata])
            samples = np.stack([(counts[:, ix] @ correct[ix]) / len(ix) for ix in strata], axis=1)
            bootstrap[method][query] = dict(balanced_agreement=bs, probability_mae=weights @ error,
                                            profession=weights @ correct, worst_group=samples.min(1))
            cells[method + '/' + query] = dict(
                balanced_agreement=float(agreement), agreement_ci=interval(bs),
                source_changed=int(changed.sum()), source_unchanged=int((~changed).sum()),
                probability_mae=float(error.mean()), probability_mae_ci=interval(weights @ error),
                profession=float(correct.mean()), profession_ci=interval(weights @ correct),
                worst_group=float(group_acc.min()), worst_group_ci=interval(samples.min(1)),
                groups=group_acc.tolist(), logit_mae=float(np.abs(v - ref).mean()))
        summaries[method] = {}
        for family, queries in [('parts', PARTS), ('unions', UNIONS), ('full', ['full'])]:
            summaries[method][family] = {}
            for metric in ['balanced_agreement', 'probability_mae', 'profession', 'worst_group']:
                point = np.mean([cells[method + '/' + q][metric] for q in queries])
                draws = np.mean([bootstrap[method][q][metric] for q in queries], axis=0)
                summaries[method][family][metric] = dict(value=float(point), ci=interval(draws))
    contrasts = {}
    pairs = [('native_groups', c) for c in ['native', 'native_refit', 'native_total', 'geometry', 'raw']]
    pairs += [('native', 'geometry'), ('native', 'raw')]
    for method, control in pairs:
        for family, queries in [('parts', PARTS), ('unions', UNIONS), ('full', ['full'])]:
            key = method + '-' + control + '/' + family
            contrasts[key] = {}
            for metric in ['balanced_agreement', 'probability_mae', 'profession', 'worst_group']:
                gain = summaries[method][family][metric]['value'] - summaries[control][family][metric]['value']
                bs = np.mean([bootstrap[method][q][metric] - bootstrap[control][q][metric]
                              for q in queries], axis=0)
                contrasts[key][metric] = dict(difference=float(gain), ci=interval(bs))
    old = ROOT / 'runs/REFORM_R58_shift_context_curve_s8192_v1_20260915'
    prior = load_logits(old, order)
    # Gain calibration was evaluated in R58 on the same documents. Reuse only
    # its actually measured cells; pair-union outcomes did not exist there.
    gain_cells = {}
    gain_draws = []
    native_draws = []
    for query in ['full', *PARTS]:
        ref = values['source/' + query]
        changed = (ref > 0) != (base > 0)
        v = prior['geometry_gain/' + query]
        agree = (v > 0) == (ref > 0)
        bs = .5 * ((counts @ (agree & changed)) / (counts @ changed) +
                   (counts @ (agree & ~changed)) / (counts @ ~changed))
        gain_cells[query] = dict(balanced_agreement=float(.5 * (agree[changed].mean() + agree[~changed].mean())),
                                ci=interval(bs), profession=float(((v > 0) == y).mean()),
                                worst_group=float(min(((v[ix] > 0) == y[ix]).mean() for ix in strata)),
                                probability_mae=float(np.abs(sigmoid(v) - sigmoid(ref)).mean()))
        if query in PARTS:
            gain_draws.append(bs)
            native_draws.append(bootstrap['native'][query]['balanced_agreement'])
    gain_mean = float(np.mean([gain_cells[q]['balanced_agreement'] for q in PARTS]))
    gain_supplement = dict(
        source=identity(old / 'metrics.raw.jsonl'), cells=gain_cells,
        parts_balanced=dict(value=gain_mean, ci=interval(np.mean(gain_draws, axis=0))),
        native_minus_gain_parts=dict(
            difference=summaries['native']['parts']['balanced_agreement']['value'] - gain_mean,
            ci=interval(np.mean(native_draws, axis=0) - np.mean(gain_draws, axis=0))),
        scope='Existing R58 gain-calibrated geometry outputs on identical development documents. '
              'Only full and single parts were measured; no union outcome is inferred.')
    replay = {method: max(float(np.max(np.abs(values[method + '/' + q] - prior[method + '/' + q])))
                         for q in ['full', *PARTS]) for method in ['source', 'geometry', 'native', 'raw']}
    fit = read(run / 'RELATION_FIT.json')
    support = {'native': sum(v['selected'] for v in fit.values())}
    support.update({name: sum(v['part_fits'][name]['selected'] for v in fit.values())
                    for name in ['native_refit', 'native_groups', 'native_total']})
    out = dict(written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               run_id=args.run, results=cells, summary=summaries, contrasts=contrasts,
               target_members=support, original_controls_max_logit_difference=replay,
               gain_calibrated_geometry_supplement=gain_supplement,
               sources=[identity(run / 'metrics.raw.jsonl'), identity(run / 'RELATION_FIT.json'),
                        identity(source / 'panel.json'), identity(source / 'dev_none.npz'),
                        identity(Path(__file__))],
               scope='Exposed development biographies; one target dictionary stack and fixed source head. '
                     '2000 paired document bootstrap draws, stratified by profession/gender, seed 59515. '
                     'Parts and union scores average their three queries; full is separate. '
                     'Balanced agreement equally weights source-changed and source-unchanged documents '
                     'relative to the unedited classifier for each query. Source is the intervention '
                     'reference, not a claim of perfect task accuracy. No independent confirmation.')
    (ART / 'R59_QUERY_ANALYSIS.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    if args.export_paper:
        (ROOT / 'paper/data/human_part_predictions_development.json').write_text(
            json.dumps(out, indent=2) + '\n', encoding='utf-8')
        table = [r'\begin{tabular}{lrrrrr}\toprule',
                 r' & \multicolumn{2}{c}{Source agreement} & \multicolumn{3}{c}{Single-part outcomes}\\',
                 r'Method & Parts & Unions & Profession & Worst group & Prob.\ MAE\\\midrule']
        labels = [('source', 'Source reference'), ('geometry', 'Geometry'),
                  ('geometry_gain', 'Geometry + gains'), ('native', 'Member relation'),
                  ('native_refit', 'Member refit'), ('native_groups', 'Semantic-part refit'),
                  ('native_total', 'Complete-effect refit'), ('raw', 'Source-direction readout')]
        for name, label in labels:
            if name == 'geometry_gain':
                row = [gain_mean * 100, None] + [100 * np.mean([gain_cells[q][metric] for q in PARTS])
                                               for metric in ['profession', 'worst_group', 'probability_mae']]
            else:
                row = [100 * summaries[name][f]['balanced_agreement']['value'] for f in ['parts', 'unions']]
                row += [100 * summaries[name]['parts'][m]['value'] for m in ['profession', 'worst_group', 'probability_mae']]
            table.append(label + ' & ' + ' & '.join('--' if v is None else f'{v:.2f}' for v in row) + r'\\')
        table.append(r'\bottomrule\end{tabular}')
        (ROOT / 'paper/tables/human_part_predictions_development.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')
    for method in summaries:
        print(method, {family: {metric: round(100 * score['value'], 3) for metric, score in scores.items()}
                       for family, scores in summaries[method].items()})
    print('replay', replay, 'support', support)
    print('parts-vs-refit', contrasts['native_groups-native_refit/parts'])
    print('unions-vs-refit', contrasts['native_groups-native_refit/unions'])


if __name__ == '__main__':
    main()
