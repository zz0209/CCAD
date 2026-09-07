"""Retain the prechosen complete examples alongside their successes and failures."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/seven_round_rebuild_20260906/r5_mechanism'
LABELS = ['is', 'are', 'was', 'were']


def main():
    source = OUT / 'R5_MECHANISM_SUMMARY.json'
    data = json.loads(source.read_text())
    lookup = {(r['row_id'], r['factor'], r['method'], r.get('member', -1)): r
              for r in data['fixed_examples']}
    ids = sorted({r['row_id'] for r in data['fixed_examples']})
    assert len(ids) == 16
    records = []
    lines = [
        '# Fixed member-mechanism examples', '',
        'Source seed 1 to target seed 2; first lexical block; initial number, past and distractor values all zero. '
        'All four cue pairs, both syntaxes and both roles are retained. This rule was recorded before the member run. '
        'The panel was already exposed during frozen confirmation, so this is mechanism analysis, not a new confirmation.', '',
        'Labels are the highest-probability member of {is, are, was, were}, not the highest-probability token in the full vocabulary. '
        'Every KL is over the full vocabulary. Source-teacher KL uses the full donor distribution; FCC KL uses the source teacher; '
        'member/native/swap KL uses intact FCC. These reference types must not be combined into a single fidelity ranking.', '',
        '## All prechosen cases', '',
        '| Row / role / syntax | Recipient | Factor | Expected | Source | FCC | Equal-norm role swap |',
        '|---|---|---|---|---|---|---|',
    ]
    for i in ids:
        for factor in ['number', 'time', 'joint']:
            src = lookup[i, factor, 'source_teacher', -1]
            fcc = lookup[i, factor, 'fcc_group', -1]
            method = 'joint_time_role_swap_norm_matched' if factor == 'joint' else 'role_swap_norm_matched'
            swap = lookup[i, factor, method, -1]
            record = dict(row_id=i, role=fcc['cue_role'], syntax=fcc['template'], cue_id=fcc['cue_id'],
                          recipient=fcc['text'].replace('<|endoftext|>', ''), donor=fcc['donor_text'].replace('<|endoftext|>', ''),
                          factor=factor, expected=LABELS[fcc['expected']], source=LABELS[src['label']],
                          fcc=LABELS[fcc['label']], equal_norm_swap=LABELS[swap['label']],
                          source_correct=src['correct'], fcc_correct=fcc['correct'],
                          source_past_logodds=src['past_logodds'], fcc_past_logodds=fcc['past_logodds'],
                          swap_past_logodds=swap['past_logodds'], source_reference=src['reference_kind'],
                          fcc_reference=fcc['reference_kind'], swap_reference=swap['reference_kind'])
            records.append(record)
            lines.append(f"| {i} / {record['role']} / {record['syntax']} | {record['recipient']} | {factor} | "
                         f"{record['expected']} | {record['source']} | {record['fcc']} | {record['equal_norm_swap']} |")
    lines += ['', '## What the first cue examples establish', '',
              'The main figure shows the first prepositional examples, rows 0 and 256, for the time-only operation. '
              'Both source and FCC choose the intended tense. Their joint number-and-time operations both fail: '
              'row 0 should produce were but source and FCC choose was; row 256 should produce are but both choose is. '
              'These are source failures that high source fidelity cannot fix.', '',
              'With the same first cue in object-relative syntax (row 8), source and FCC jointly choose the intended were. '
              'In the quoted object-relative case (row 264), source chooses the intended are but FCC chooses is: '
              'this is a transfer failure. All four cases have correct source/FCC time-only labels. '
              'A successful single-factor example must not stand in for successful joint reuse.', '',
              '## Natural activation context and overlapping membership', '',
              'All 39 unique source members and 34 unique target members were measured on the existing 32,768-token '
              'FineWeb validation sample. This sample was previously used for SAE quality; it is descriptive natural '
              'activation evidence, not new task validation. Each top context is clipped within both its original '
              '128-token model context and its source document. All 73 members and top-three contexts remain in '
              'the raw natural-member file, including active counts and document/token identity.', '',
              'The six displayed target time members were ranked by RMS code change times predictive-vector norm '
              'on old temporal fitting data. They were not selected by confirmation accuracy or removal effect. '
              'All 32 time members generate the intact operation. A bracketed token marks the measured position; '
              'the fragment is not a proposed semantic label.', '',
              '| Member | Natural high-activation fragment | Active fraction | Clause/title code change | Clause/title removal time loss (nat) | Also number | Source |',
              '|---:|---|---:|---:|---:|---|---|']
    manifest = json.loads((OUT / 'figure_member_example_manifest.json').read_text())
    displayed = [r for r in manifest['displayed_values'] if 'member' in r]
    for r in displayed:
        lines.append(f"| {r['member']} | {r['natural_excerpt']} | {r['natural_active_fraction']:.3f} | "
                     f"{r['temporal_code_change']:+.2f} / {r['quoted_code_change']:+.2f} | "
                     f"{r['temporal_removal_time_shift_loss']:+.2f} / {r['quoted_removal_time_shift_loss']:+.2f} | "
                     f"{'yes' if r['number_shared'] else 'no'} | [document]({r['source_url']}) |")
    lines += ['', 'Positive removal loss means that deleting the predictive contrast term lowers past-versus-present '
              'log odds relative to intact FCC. Negative loss means the opposite. Feature 1635 contributes no change '
              'in these fixed examples despite activating on nearly half of natural validation positions; feature 1825 '
              'has opposite-signed removal effects across roles. Eight of the ten target number members are also time '
              'predictors. The relation is overlapping and factor-conditioned. Neither overlap, natural context, nor '
              'regression coefficients identify a uniquely temporal native feature.', '',
              'Deleting a predictive term changes the constructed FCC edit. It does not zero a native SAE activation '
              'in the original model. The equal-norm controls test an amplitude-only explanation; they do not match '
              'every distributional property or certify that the exchanged edit is on-manifold.', '',
              '## Sources', '',
              '- `R5_MECHANISM_SUMMARY.json`: complete frozen-case operations and baselines.',
              '- `figure_member_example_manifest.json`: exact plotted values and natural source URLs.',
              '- `runs/SEVEN_R5_member_natural_contexts_v1_20260907/metrics.raw.jsonl`: all 73 natural-member records.', '']
    (OUT / 'R5_FIXED_CASES.md').write_text('\n'.join(lines), encoding='utf-8')
    with (OUT / 'fixed_cases.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (OUT / 'fixed_cases_manifest.json').write_text(json.dumps(dict(
        cases=16, factor_rows=len(records), source=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selection='source1target2/block0/number0past0distractor0/allcues/bothsyntax/bothroles',
        natural_member_source='runs/SEVEN_R5_member_natural_contexts_v1_20260907/metrics.raw.jsonl'), indent=2)+'\n')
    print(json.dumps(dict(cases=len(ids), factor_rows=len(records), report=str(OUT/'R5_FIXED_CASES.md'))))


if __name__ == '__main__':
    main()
