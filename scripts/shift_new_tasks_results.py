"""Paired analysis of explanation reuse in new classification tasks."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20260916)
    p.add_argument('--classifier', default='retrained', choices=['retrained', 'frozen'])
    a = p.parse_args()
    rows, identities, specs = {}, [], {}
    for run in a.runs:
        raw = run/'metrics.raw.jsonl'
        identities.append(dict(path=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
        cfg = json.loads((run/'config.resolved.json').read_text())
        specs.update({t['name']: t for t in cfg['tasks']})
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if r['classifier'] != a.classifier:
                continue
            key = tuple(r[k] for k in ['task', 'method', 'operation', 'target_seed', 'probe_seed', 'component'])
            if key in rows:
                assert r['prediction'] == rows[key]['prediction']
                assert abs(r['logit']-rows[key]['logit']) < 2e-5
            rows[key] = r
    methods = sorted({k[1] for k in rows})
    seeds = sorted({k[3] for k in rows})
    probes = sorted({k[4] for k in rows})
    queries = ['full', 'pronouns', 'names', 'associated_words']
    pairs = sorted({n.rsplit('_orientation', 1)[0] for n in specs})
    source_seed = min(k[3] for k in rows if k[1] == 'source')
    clean_seed = min(k[3] for k in rows if k[1] == 'none')
    arrays, strata, counts, changed_counts = {}, {}, {}, {}
    for pair in pairs:
        tasks = [pair+f'_orientation{o}' for o in [0, 1]]
        docs = sorted({k[-1] for k in rows if k[0] == tasks[0]})
        ref = [rows[tasks[0], 'none', 'full', clean_seed, probes[0], d] for d in docs]
        y = np.array([r['label'] for r in ref])
        g = np.array([r['gender'] for r in ref])
        strata[pair] = [np.flatnonzero((y == yy) & (g == gg)) for yy in [0, 1] for gg in [0, 1]]
        assert all(len(ix) for ix in strata[pair])
        counts[pair] = [len(ix) for ix in strata[pair]]
        pred = np.empty((len(methods), 4, len(seeds), len(probes), 2, len(docs)), bool)
        for mi, method in enumerate(methods):
            for qi, query in enumerate(queries):
                for si, seed in enumerate(seeds):
                    source = source_seed if method == 'source' else (clean_seed if method == 'none' else seed)
                    for pi, probe in enumerate(probes):
                        for oi, task in enumerate(tasks):
                            operation = 'full' if method == 'none' else query
                            pred[mi, qi, si, pi, oi] = [rows[task, method, operation, source, probe, d]['prediction'] for d in docs]
        arrays[pair] = pred, y
        changed_counts[pair] = (pred[methods.index('source'), :, 0] !=
                                pred[methods.index('none'), :, 0]).sum(-1).tolist()

    def measure(seed_indices=None, draws=None):
        summaries = []
        for pair in pairs:
            pred, labels = arrays[pair]
            # Strata keep both confounding orientations paired on each biography.
            groups = strata[pair] if draws is None else draws[pair]
            doc_indices = np.concatenate(groups)
            current = np.take(pred, np.arange(len(seeds)) if seed_indices is None else seed_indices, axis=2)
            correct = current == labels
            accuracy = correct[..., doc_indices].mean((2, 3, 4, 5))
            group_accuracy = np.stack([correct[..., ix].mean(-1) for ix in groups])
            worst = group_accuracy.min(0).mean((2, 3, 4))
            teacher = current[methods.index('source')]
            unchanged = current[methods.index('none')]
            changed = teacher != unchanged
            agree = current == teacher
            balanced = []
            for flag in [False, True]:
                eligible = (changed == flag)[..., doc_indices]
                numerator = (agree[..., doc_indices]*eligible).sum(-1)
                denominator = eligible.sum(-1)
                balanced.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0))
            b = np.nanmean(.5*(balanced[0]+balanced[1]), axis=(2, 3, 4))
            summaries.append(np.stack([accuracy, worst, b], axis=-1))
        return np.mean(summaries, axis=0), np.stack(summaries)

    observed, by_pair = measure()
    rng = np.random.default_rng(a.seed)
    samples = []
    for _ in range(a.bootstrap):
        draws = {pair: [rng.choice(ix, len(ix), replace=True) for ix in strata[pair]] for pair in pairs}
        samples.append(measure(rng.integers(len(seeds), size=len(seeds)), draws)[0])
    samples = np.array(samples)
    names = ['profession', 'worst_group', 'balanced_source_agreement']
    result = {}
    for mi, method in enumerate(methods):
        result[method] = {}
        for qi, query in enumerate(queries+['parts_mean']):
            value = observed[mi, qi] if qi < 4 else observed[mi, 1:].mean(0)
            sample = samples[:, mi, qi] if qi < 4 else samples[:, mi, 1:].mean(1)
            result[method][query] = {n: dict(mean=float(value[j]), ci95=np.quantile(sample[:, j], [.025, .975]).tolist()) for j,n in enumerate(names)}
    contrasts, interactions = {}, {}
    for method in methods:
        if method in ['none', 'source']:
            continue
        for control in ['none', 'geometry', 'geometry_gain', 'native', 'raw']:
            if control not in methods or method == control:
                continue
            mi, ci = methods.index(method), methods.index(control)
            key = method+' minus '+control
            contrasts[key] = {}
            for qi, query in enumerate(queries+['parts_mean']):
                v = observed[mi]-observed[ci]
                s = samples[:, mi]-samples[:, ci]
                value = v[qi] if qi < 4 else v[1:].mean(0)
                sample = s[:, qi] if qi < 4 else s[:, 1:].mean(1)
                contrasts[key][query] = {n: dict(mean=float(value[j]), ci95=np.quantile(sample[:, j], [.025, .975]).tolist()) for j,n in enumerate(names)}
            value = (observed[mi, 1:]-observed[ci, 1:]).mean(0)-(observed[mi, 0]-observed[ci, 0])
            sample = (samples[:, mi, 1:]-samples[:, ci, 1:]).mean(1)-(samples[:, mi, 0]-samples[:, ci, 0])
            interactions[key] = {n: dict(mean=float(value[j]), ci95=np.quantile(sample[:, j], [.025, .975]).tolist()) for j,n in enumerate(names)}
    output = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), classifier=a.classifier,
                  source_paths=identities, methods=methods, target_seeds=seeds, probe_seeds=probes,
                  task_pairs=pairs, profession_gender_counts=counts, source_changed_counts=changed_counts,
                  results=result, contrasts=contrasts, part_minus_full_interactions=interactions,
                  by_pair={pair: {m: by_pair[i,j].tolist() for j,m in enumerate(methods)} for i,pair in enumerate(pairs)},
                  bootstrap=a.bootstrap, bootstrap_seed=a.seed,
                  inference='Fixed task-pair cohort and source explanation; jointly resample target seeds and documents within each profession/gender stratum, sharing draws across methods, requests, orientations and classifier seeds. Average classifier seeds; task pairs are not treated as independent directions. A one-target development interval contains document uncertainty only.',
                  metric_order=names, query_order=queries)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), methods=methods, target_seeds=seeds, pairs=pairs)))


if __name__ == '__main__':
    main()
