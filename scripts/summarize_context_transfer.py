"""All-case frozen transfer summaries, retaining role, cue and lexical dependence.

Only reads saved scalar outcomes. No model, fitting, selection or GPU imports.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMS = ['correct', 'teacher_agree', 'kl', 'noop_kl', 'source_donor_kl',
        'number_correct', 'time_correct', 'number_shift', 'time_shift',
        'signed_number_shift', 'signed_time_shift', 'abs_number_shift',
        'abs_time_shift', 'delta_norm']


def partitions(row, familiar):
    role = row.get('cue_role', 'temporal')
    cue = 'familiar_cue' if row['cue_id'] in familiar else 'new_cue'
    return ['all', role, f'{role}/{cue}', f'{role}/{cue}/{row["template"]}']


def average(rows):
    n = sum(r['n'] for r in rows)
    result = dict(n=n)
    for name in SUMS:
        total = sum(r[name] for r in rows)
        result[name + '_sum'] = total
        result[name] = total / n
    result['accuracy'] = result.pop('correct')
    result['teacher_agreement'] = result.pop('teacher_agree')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--familiar-cues', nargs='+', type=int, default=[0, 1])
    args = ap.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    summary = json.loads((run / 'metrics.summary.json').read_text())
    assert summary['status'] == 'PASS', 'Only a complete frozen run can be summarized.'
    cfg = json.loads((run / 'config.resolved.json').read_text())
    material = ROOT / cfg['material_run']
    panel = json.loads((material / 'panel.json').read_text())
    rows, pairs = panel['rows'], panel['pairs']
    baseline, raw_records = {}, []
    for line in (material / 'metrics.raw.jsonl').open():
        r = json.loads(line)
        if r['kind'] == 'baseline':
            baseline[r['row_id']] = r
        raw_records.append(r)
    teachers = {}
    for path in run.glob('source*_endpoint.json'):
        for r in json.loads(path.read_text())['rows']:
            teachers[r['source_seed'], r['factor'], r['row_id']] = r
    groups = defaultdict(lambda: defaultdict(float))
    keys = ['partition', 'factor', 'method', 'source_seed', 'target_seed', 'block']
    count = 0
    examples = []
    for line in (run / 'metrics.raw.jsonl').open():
        r = json.loads(line)
        count += 1
        row, base = rows[r['row_id']], baseline[r['row_id']]
        teacher = teachers[r['source_seed'], r['factor'], r['row_id']]
        donor = rows[pairs[r['row_id']][r['factor']]]
        dn = r['number_logodds'] - base['number_logodds']
        dt = r['past_logodds'] - base['past_logodds']
        values = dict(correct=r['correct'], teacher_agree=r['label'] == teacher['teacher_label'],
                      kl=0. if r['method'] == 'source_teacher' else r['kl_reference'],
                      noop_kl=teacher['teacher_noop_kl'],
                      source_donor_kl=r['kl_reference'] if r['method'] == 'source_teacher' else 0.,
                      number_correct=(r['number_logodds'] > 0) == bool(r['expected'] % 2),
                      time_correct=(r['past_logodds'] > 0) == bool(r['expected'] // 2),
                      number_shift=dn, time_shift=dt,
                      signed_number_shift=dn * (1 if donor['number'] else -1),
                      signed_time_shift=dt * (1 if donor['past'] else -1),
                      abs_number_shift=abs(dn), abs_time_shift=abs(dt), delta_norm=r['delta_norm'])
        variants = [(r['method'], values)]
        if r['method'] == 'source_teacher':
            noop = dict(values, correct=base['label'] == r['expected'],
                        teacher_agree=base['label'] == teacher['teacher_label'],
                        kl=teacher['teacher_noop_kl'], source_donor_kl=0.,
                        number_correct=(base['number_logodds'] > 0) == bool(r['expected'] % 2),
                        time_correct=(base['past_logodds'] > 0) == bool(r['expected'] // 2),
                        number_shift=0., time_shift=0., signed_number_shift=0., signed_time_shift=0.,
                        abs_number_shift=0., abs_time_shift=0., delta_norm=0.)
            variants.append(('no_op', noop))
        for method, value_dict in variants:
            for part in partitions(row, args.familiar_cues):
                key = (part, r['factor'], method, r['source_seed'], r.get('target_seed', 0), row['block'])
                g = groups[key]
                g['n'] += 1
                for name, value in value_dict.items():
                    g[name] += value
        # Fixed first lexeme/value in each role/cue/syntax; no outcome selection.
        if row['block'] == 0 and row['number'] == row['past'] == row['distractor'] == 0 and r['source_seed'] == 1 and r.get('target_seed', 2) == 2:
            examples.append(dict(text=row['text'], donor_text=donor['text'], cue_role=row.get('cue_role', 'temporal'), baseline=base, **r))
    blocks = [dict(zip(keys, key), **value) for key, value in groups.items()]
    by_edge, by_method = defaultdict(list), defaultdict(list)
    for r in blocks:
        by_edge[tuple(r[k] for k in keys[:-1])].append(r)
        by_method[tuple(r[k] for k in keys[:3])].append(r)
    edges = [dict(zip(keys[:-1], key), **average(rr)) for key, rr in by_edge.items()]
    pooled = []
    for key, rr in by_method.items():
        by_source = defaultdict(list)
        for r in rr:
            by_source[r['source_seed']].append(r)
        seeds = [dict(source_seed=s, **average(ss)) for s, ss in sorted(by_source.items())]
        pooled.append(dict(zip(keys[:3], key), **average(rr), source_seed_means=seeds,
                           source_seed_kl_min=min(r['kl'] for r in seeds),
                           source_seed_kl_max=max(r['kl'] for r in seeds),
                           directions=len({(r['source_seed'], r['target_seed']) for r in rr})))
    lookup = {(r['partition'], r['factor'], r['method'], r['source_seed'], r['target_seed']): r for r in edges}
    comparisons = []
    for part, factor in sorted({(r['partition'], r['factor']) for r in edges}):
        fcc = [r for r in edges if r['partition'] == part and r['factor'] == factor and r['method'] == 'fcc_group']
        for method in sorted({r['method'] for r in edges} - {'fcc_group', 'source_teacher'}):
            matched = []
            for r in fcc:
                prefix = (part, factor, method, r['source_seed'])
                other = lookup.get(prefix + (r['target_seed'],), lookup.get(prefix + (0,)))
                if other is not None:
                    matched.append((r, other))
            if matched:
                comparisons.append(dict(partition=part, factor=factor, comparator=method,
                                        directions=len(matched), fcc_lower_kl=sum(a['kl'] < b['kl'] for a, b in matched),
                                        mean_kl_difference=sum(a['kl'] - b['kl'] for a, b in matched) / len(matched),
                                        comparator_source_only=all(b['target_seed'] == 0 for a, b in matched)))
    material_groups = defaultdict(list)
    for r in raw_records:
        for part in partitions(rows[r['row_id']], args.familiar_cues):
            material_groups[part, r.get('method', 'baseline')].append(r)
    material_summary = [dict(partition=key[0], method=key[1], n=len(rr),
                             correct=sum(r['correct'] for r in rr), accuracy=sum(r['correct'] for r in rr) / len(rr),
                             kl_to_full_donor=sum(r.get('kl_to_full_donor', 0.) for r in rr) / len(rr))
                        for key, rr in material_groups.items()]
    out.mkdir(parents=True, exist_ok=True)
    for name, rr in [('context_blocks', blocks), ('context_edges', edges)]:
        with (out / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rr[0]))
            writer.writeheader()
            writer.writerows(rr)
    result = dict(run=str(run), raw_rows=count, run_summary=summary, rows=pooled,
                  comparisons=comparisons, material_rows=material_summary, fixed_examples=examples,
                  source_raw_sha256=hashlib.sha256((run / 'metrics.raw.jsonl').read_bytes()).hexdigest(),
                  statistics='Full denominator. Four new lexical blocks, five shared SAE seeds, two syntaxes, two roles and four cue pairs. Twenty directions are dependent. Means and source-seed ranges are descriptive; no independent-edge confidence interval. Source-only controls are averaged over five sources and compared to corresponding source edges. time_shift is change in past-vs-present log odds; signed_time_shift follows donor cue past/present metadata, including quoted titles where expected tense stays present. abs_time_shift measures unsigned change, not necessarily error. At coincident positions, joint wrong_factor is the same addition with factor names exchanged and is not a valid joint negative control.',
                  derived_noop='No new inference: baseline label/logodds from material and exact teacher_noop_kl from frozen endpoint, once per source/factor/input. Reported in addition to original raw rows.',
                  row_selection='None. Fixed examples choose source1,target2,block0,all initial factor values0 before reading output.')
    (out / 'R4_L15_CONTEXT_SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
    keep = {'source_teacher', 'fcc_group', 'full_code_ridge', 'raw_native_units', 'das_style_raw_rank1', 'global_factor_mean', 'cue_factor_mean', 'same_members_native', 'direct_target_native'}
    for r in pooled:
        if r['partition'].count('/') == 1 and r['factor'] in ['time', 'joint'] and r['method'] in keep:
            print(json.dumps({k: r[k] for k in ['partition', 'factor', 'method', 'n', 'accuracy', 'teacher_agreement', 'kl', 'signed_time_shift', 'abs_time_shift']}))


if __name__ == '__main__':
    main()
