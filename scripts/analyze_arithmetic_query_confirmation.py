"""Analyze frozen source-member queries with crossed question/partition units."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'


def analyze(freeze_name='R39_CONFIRMATION_FREEZE.json', output_prefix='r39_member_confirmation'):
    freeze = json.loads((ART / freeze_name).read_text())
    config_path = ROOT / freeze['config_path']
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == freeze['config_sha256']
    cfg = json.loads(config_path.read_text())
    run = ROOT / 'runs' / cfg['run_id']
    assert json.loads((run / 'status.json').read_text())['status'] == 'PASS'
    assert json.loads((run / 'config.resolved.json').read_text()) == cfg
    raw = run / 'metrics.raw.jsonl'
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest == json.loads((run / 'metrics.summary.json').read_text())['metrics_raw_sha256']
    panel = json.loads((run / 'panel.json').read_text())
    key = lambda q: tuple(panel['rows'][i][k] for i in [q['recipient'], q['donor']] for k in ['a', 'b'])
    ids = sorted({key(q) for q in panel['pairs']})
    lookup = {k: i for i, k in enumerate(ids)}
    nq, nb, ns = len(ids), cfg['member_queries']['partitions'], len(cfg['seeds'])
    shape = (nq, nb, 2, 2, 2, ns)
    names = ['source', 'member', 'assignment', 'two_assignment', 'wrong']
    answers = {n: np.full(shape, -99999, dtype=np.int64) for n in names}
    full = {n: np.full((nq, 2, 2, ns), -99999, dtype=np.int64) for n in ['source', 'member', 'assignment']}
    full_function = {n: np.full((nq, 2, 2, ns, 3), np.nan) for n in full}
    baseline = np.full((nq, 2), -99999, dtype=np.int64)
    rows = [json.loads(s) for s in raw.read_text().splitlines()]
    base = {r['row_id']: r['answer'] for r in rows if r['kind'] == 'base'}
    count = 0
    for r in rows:
        if r['kind'] != 'source_patch' or r['seed'] == 0:
            continue
        q = panel['pairs'][r['row_id']]
        qi, oi, ti, si = lookup[key(q)], ['unit', 'tens'].index(r['operation']), q['template'], cfg['seeds'].index(r['seed'])
        text = re.match(r'\s*(\d+)', r['generated_text'])
        value = int(text[1]) if text else -1
        assert value == (-1 if r['answer'] is None else r['answer'])
        recipient, donor = [panel['rows'][q[k]] for k in ['recipient', 'donor']]
        desired = (recipient['total']//10*10+donor['total']%10 if oi == 0
                   else donor['total']//10*10+recipient['total']%10)
        numeric = 10 <= value < 100
        target = numeric and (value%10 == donor['total']%10 if oi == 0 else value//10 == donor['total']//10)
        preserve = numeric and (value//10 == recipient['total']//10 if oi == 0 else value%10 == recipient['total']%10)
        flags = [value == desired, target, preserve]
        assert flags == [r[k] for k in ['exact_hybrid', 'target_digit_success', 'preserve_digit_success']]
        if r['method'].endswith('_full'):
            name = r['method'][:-5]
            ix = (qi, oi, ti, si)
            assert full[name][ix] == -99999
            full[name][ix] = value
            full_function[name][ix] = flags
        else:
            match = re.fullmatch(r'(.+)_part([01])_bank(\d+)', r['method'])
            assert match
            name, part, bank = match.groups()
            ix = (qi, int(bank), int(part), oi, ti, si)
            assert answers[name][ix] == -99999
            answers[name][ix] = value
        b = base[q['recipient']]
        baseline[qi, ti] = -1 if b is None else b
        count += 1
    assert all((v != -99999).all() for v in [*answers.values(), *full.values(), baseline])
    parent = ROOT / cfg['member_queries']['relation_run']
    relations = json.loads((parent / 'config.resolved.json').read_text())['relation_transfer']['seed_pairs']
    order = [cfg['seeds'].index(next(s for s, target in relations if target == t)) for t in cfg['seeds']]
    source = answers['source'][..., order]
    changed = source != baseline[:, None, None, None, :, None]
    agree = {n: v == source for n, v in answers.items() if n != 'source'}
    agree['full_component'] = full['member'][:, None, None, ...] == source
    agree['unchanged'] = ~changed
    # Each Q x B count retains both parts, operations, prompts and the fixed SAE cohort.
    den = np.stack([changed.sum(axis=(2, 3, 4, 5)), (~changed).sum(axis=(2, 3, 4, 5))], -1)
    counts = {n: np.stack([(v & changed).sum(axis=(2, 3, 4, 5)), (v & ~changed).sum(axis=(2, 3, 4, 5))], -1) for n, v in agree.items()}
    cells = [dict(method=n, exact_agreement=float(v.mean()),
                  changed_agreement=float(v[changed].mean()), unchanged_agreement=float(v[~changed].mean()),
                  balanced_agreement=float(.5*(v[changed].mean()+v[~changed].mean()))) for n, v in agree.items()]
    rng = np.random.default_rng(freeze['analysis']['bootstrap_seed'])
    draws = freeze['analysis']['draws']
    qw = rng.multinomial(nq, np.full(nq, 1/nq), size=draws)
    bw = rng.multinomial(nb, np.full(nb, 1/nb), size=draws)
    contrasts = []
    for label, weights in [('question_clusters', np.ones_like(bw)), ('question_and_partition', bw)]:
        denominators = np.einsum('dq,db,qbc->dc', qw, weights, den)
        assert (denominators > 0).all()
        estimates = {n: (np.einsum('dq,db,qbc->dc', qw, weights, v)/denominators).mean(1) for n, v in counts.items()}
        for other in ['assignment', 'two_assignment', 'wrong', 'full_component', 'unchanged']:
            diff = estimates['member'] - estimates[other]
            point = next(c['balanced_agreement'] for c in cells if c['method'] == 'member') - next(c['balanced_agreement'] for c in cells if c['method'] == other)
            contrasts.append(dict(comparator=other, units=label, difference_points=100*point,
                                  interval_points=(100*np.quantile(diff, [.025, .975])).tolist()))
    bank_scores = []
    for b in range(nb):
        dd = den[:, b].sum(0)
        bank_scores.append(dict(bank=b, scores={n: float((v[:, b].sum(0)/dd).mean()) for n, v in counts.items()}))
    functional = [dict(method=n, operation=op, **{m: float(v[:, o, ..., k].mean()) for k, m in enumerate(['H', 'T', 'P'])})
                  for n, v in full_function.items() for o, op in enumerate(['unit', 'tens'])]
    full_contrasts = []
    for other in ['source', 'assignment']:
        difference = (full_function['member'][..., 0]-full_function[other][..., 0]).mean((1, 2, 3))
        ci = np.quantile(qw @ difference/nq, [.025, .975])*100
        full_contrasts.append(dict(comparator=other, difference_points=float(100*difference.mean()), interval_points=ci.tolist()))
    out = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), scope=cfg['scope'], run=run.relative_to(ROOT).as_posix(),
               raw_sha256=digest, parsed_outputs=count, questions=nq, partition_banks=nb,
               source_query_count=int(source.size), source_changed_count=int(changed.sum()),
               source_changed_fraction=float(changed.mean()), source_numeric_fraction=float(((source>=10)&(source<100)).mean()),
               cells=cells, contrasts=contrasts, by_partition=bank_scores, full_function=functional,
               full_contrasts=full_contrasts, statistics=freeze['analysis']['uncertainty'])
    (ART / (output_prefix+'.json')).write_text(json.dumps(out, indent=2)+'\n')
    np.savez_compressed(ART / (output_prefix+'.npz'), methods=np.array(names), answers=np.stack(list(answers.values())),
                        source_aligned=source, changed=changed, baseline=baseline, operand_pairs=np.array(ids),
                        full_methods=np.array(list(full)), full_answers=np.stack(list(full.values())),
                        full_function=np.stack(list(full_function.values())))
    print(json.dumps(out))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',default='R39_CONFIRMATION_FREEZE.json')
    parser.add_argument('--output-prefix',default='r39_member_confirmation')
    args=parser.parse_args()
    analyze(args.freeze,args.output_prefix)
