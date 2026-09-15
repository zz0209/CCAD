"""Replay the frozen R59 part-prediction endpoint from retained model outputs.

The same stratified document resample is used for every target dictionary.
The primary cohort resamples the four new target seeds, not source seeds.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from shift_query_results import ART, ROOT, PARTS, UNIONS, read, identity, load_logits

FAMILIES = {'parts': PARTS, 'unions': UNIONS, 'full': ['full']}
METRICS = ['balanced_agreement', 'probability_mae', 'logit_mae', 'profession', 'worst_group']
METHODS = ['source', 'geometry', 'geometry_gain', 'native', 'raw']


def interval(x):
    assert np.isfinite(x).all()
    return np.quantile(x, [.025, .975]).tolist()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def summarize(values, labels, strata, counts):
    n = len(labels)
    base = values['none/full']
    cells, boot, families, family_boot = {}, {}, {}, {}
    for method in METHODS:
        for query in ['full', *PARTS, *UNIONS]:
            key = method + '/' + query
            v, ref = values[key], values['source/' + query]
            changed = (ref > 0) != (base > 0)
            agree = (v > 0) == (ref > 0)
            correct = (v > 0) == labels
            assert 0 < changed.sum() < n, ('undefined balanced endpoint', query)
            denominator_changed = counts @ changed
            denominator_unchanged = counts @ ~changed
            assert np.all(denominator_changed > 0) and np.all(denominator_unchanged > 0)
            group_acc = np.array([correct[ix].mean() for ix in strata])
            group_draws = np.stack([counts[:, ix] @ correct[ix] / len(ix) for ix in strata], axis=1)
            probability_error = np.abs(sigmoid(v) - sigmoid(ref))
            logit_error = np.abs(v - ref)
            boot[key] = dict(
                balanced_agreement=.5 * (counts @ (agree & changed) / denominator_changed +
                                         counts @ (agree & ~changed) / denominator_unchanged),
                probability_mae=counts @ probability_error / n,
                logit_mae=counts @ logit_error / n,
                profession=counts @ correct / n, worst_group=group_draws.min(1))
            point = dict(balanced_agreement=.5 * (agree[changed].mean() + agree[~changed].mean()),
                         probability_mae=probability_error.mean(), logit_mae=logit_error.mean(),
                         profession=correct.mean(), worst_group=group_acc.min())
            cells[key] = {metric: dict(value=float(value), document_ci=interval(boot[key][metric]))
                          for metric, value in point.items()}
            cells[key].update(source_changed=int(changed.sum()), source_unchanged=int((~changed).sum()),
                              group_accuracy=group_acc.tolist())
        for family, queries in FAMILIES.items():
            key = method + '/' + family
            family_boot[key], families[key] = {}, {}
            for metric in METRICS:
                draws = np.mean([boot[method + '/' + q][metric] for q in queries], axis=0)
                family_boot[key][metric] = draws
                families[key][metric] = dict(
                    value=float(np.mean([cells[method + '/' + q][metric]['value'] for q in queries])),
                    document_ci=interval(draws))
    unedited = (base > 0) == labels
    return dict(cells=cells, families=families,
                unedited=dict(profession=float(unedited.mean()),
                              worst_group=float(min(unedited[ix].mean() for ix in strata)),
                              group_accuracy=[float(unedited[ix].mean()) for ix in strata])), family_boot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ART / 'R59_CONFIRMATION_ANALYSIS.json')
    args = parser.parse_args()
    freeze_path = ART / 'R59_CONFIRMATION_FREEZE.json'
    freeze = read(freeze_path)
    # The protocol and the executables/configurations used by this confirmation
    # were fixed before test-document selection and all model responses.
    for item in [*freeze['code_identities'], *freeze['configs'], freeze['target_training_config']]:
        p = ROOT / item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
    panel_path = Path(freeze['panel_path'])
    panel = read(panel_path)['rows']
    assert panel and all(row['split'] == 'test_confirmation' for row in panel)
    order = {row['document_sha256']: i for i, row in enumerate(panel)}
    assert len(order) == len(panel)
    labels = np.array([row['label'] for row in panel])
    gender = np.array([row['gender'] for row in panel])
    strata = [np.flatnonzero((labels == y) & (gender == g)) for y in (0, 1) for g in (0, 1)]
    rng = np.random.default_rng(59516)
    counts = np.zeros((4000, len(panel)), dtype=np.float64)
    for ix in strata:
        samples = rng.choice(ix, size=(len(counts), len(ix)), replace=True)
        for i, sample in enumerate(samples):
            counts[i] += np.bincount(sample, minlength=len(panel))
    seed_draw = rng.integers(0, 4, size=(len(counts), 4))
    summaries, draws, sources, reference, counts_by_run = {}, {}, [], None, {}
    for seed in range(1, 6):
        cfg = read(ROOT / f'configs/reform_r59_shift_confirm_seed{seed}_v1.json')
        run = ROOT / 'runs' / cfg['run_id']
        assert read(run / 'status.json')['status'] == 'PASS', run
        assert read(run / 'contract_validation.json')['ok'], run
        values = load_logits(run, order)
        expected = {'none/full'} | {m + '/' + q for m in METHODS for q in ['full', *PARTS, *UNIONS]}
        assert set(values) == expected
        common = {key: values[key] for key in expected if key.startswith(('none/', 'source/'))}
        if reference is None:
            reference = common
        else:
            assert all(np.array_equal(reference[key], common[key]) for key in common), 'source/unedited changed across targets'
        summaries[str(seed)], draws[seed] = summarize(values, labels, strata, counts)
        counts_by_run[cfg['run_id']] = len(values) * len(panel)
        sources.extend(identity(run / name) for name in ['metrics.raw.jsonl', 'config.resolved.json',
                                                       'RELATION_FIT.json', 'status.json', 'contract_validation.json'])
    cohort, contrasts = {}, {}
    new_seeds = [2, 3, 4, 5]

    def cohort_draw(key, metric, control=None):
        a = np.stack([draws[s][key][metric] -
                      (draws[s][control][metric] if control else 0) for s in new_seeds])
        fixed = a.mean(0)
        joint = a[seed_draw, np.arange(len(counts))[:, None]].mean(1)
        return fixed, joint

    for key in draws[2]:
        cohort[key] = {}
        for metric in METRICS:
            fixed, joint = cohort_draw(key, metric)
            points = [summaries[str(s)]['families'][key][metric]['value'] for s in new_seeds]
            cohort[key][metric] = dict(value=float(np.mean(points)), per_seed=points,
                                       document_ci=interval(fixed), document_target_seed_ci=interval(joint))
    for control in ['geometry_gain', 'geometry', 'raw']:
        for family in FAMILIES:
            left, right = 'native/' + family, control + '/' + family
            key = 'native-' + control + '/' + family
            contrasts[key] = {}
            for metric in METRICS:
                fixed, joint = cohort_draw(left, metric, right)
                points = [summaries[str(s)]['families'][left][metric]['value'] -
                          summaries[str(s)]['families'][right][metric]['value'] for s in new_seeds]
                contrasts[key][metric] = dict(difference=float(np.mean(points)), per_seed=points,
                    document_ci=interval(fixed), document_target_seed_ci=interval(joint))
    out = dict(written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        round_id='R59', panel_documents=len(panel), group_counts=[len(ix) for ix in strata],
        primary=contrasts['native-geometry_gain/parts']['balanced_agreement'],
        new_target_seed_cohort=cohort, contrasts=contrasts, per_target_seed=summaries,
        raw_rows_by_run=counts_by_run, source_and_unedited_logits_identical_across_targets=True,
        sources=[identity(freeze_path), identity(panel_path), identity(Path(__file__)), *sources],
        scope='Frozen confirmation on new official biographies, fixed source head and public source explanation. '
              'Primary averages the three predefined single parts over new target seeds2-5. '
              'Seed1 is a separate replication of previously developed target material. '
              '4000 paired profession/gender-stratified document and target-seed resamples, RNG59516. '
              'Document draws are shared across targets, parts and methods; the fixed-cohort interval excludes '
              'target-seed resampling. Source seeds, source annotations and query families are fixed, not resampled. '
              'Worst-group family scores average per-query minima; they are not worst overall union scores. '
              'Probability and logit MAE target source intervention outputs, not task labels.')
    args.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(primary=out['primary'], rows=counts_by_run, output=str(args.output))))


if __name__ == '__main__':
    main()
