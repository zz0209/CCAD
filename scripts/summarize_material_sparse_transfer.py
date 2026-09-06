"""Compare frozen sparse maps with controls measured on the identical panel."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    args = parser.parse_args()
    run, controls = args.run, args.controls
    out = run / 'MATERIAL_COMPARISON.json'
    if out.exists():
        raise FileExistsError(out)
    for root in (run, controls):
        assert json.loads((root / 'metrics.summary.json').read_text())['status'] == 'PASS'
        assert json.loads((root / 'contract_validation.json').read_text())['ok']
    new, old = [read_rows(root / 'metrics.raw.jsonl') for root in (run, controls)]
    src = [{(r['case_id'], r['component']): r for r in rows if r['method'] == 'source'} for rows in (new, old)]
    assert src[0] == src[1], 'Source rows must match exactly, including inputs, dose and all endpoints'
    assert (run / 'compiled_cases.json').read_bytes() == (controls / 'compiled_cases.json').read_bytes()
    rows = new + [r for r in old if r['method'] != 'source']
    assert len({(r['case_id'], r['component'], r['method']) for r in rows}) == len(rows)
    groups = defaultdict(list)
    target_rows = []
    max_error = 0.
    for r in rows:
        if r['method'] == 'source':
            continue
        method = r['method'].rsplit('_', 1)[0] if r['method'] != 'raw' else 'raw'
        target = int(r['method'].rsplit('_', 1)[1]) if method != 'raw' else None
        for scope, p in r['probability_endpoints'].items():
            anchor = src[0][r['case_id'], r['component']]['probability_endpoints'][scope]
            for k in ('source_to_baseline_kl', 'source_nll_deltas', 'positions', 'observed_next_token_ids'):
                assert p[k] == anchor[k]
            den = sum(p['source_to_baseline_kl'])
            energy = sum(v*v for v in p['source_nll_deltas'])
            values = [sum(p['source_to_candidate_kl']) / den if den > 1e-12*len(p['positions']) else None,
                      sum((s-c)**2 for s,c in zip(p['source_nll_deltas'], p['candidate_nll_deltas'])) / energy if energy > 1e-12*len(p['positions']) else None]
            for value, name in zip(values, ('normalized_kl_error', 'normalized_nll_delta_squared_error')):
                actual = p[name]
                assert (value is None and actual is None) or math.isclose(value, actual, rel_tol=1e-10, abs_tol=1e-12)
                if value is not None:
                    max_error = max(max_error, abs(value-actual))
            groups[r['case_id'], r['source_seed'], r['source_atom'], r['condition'], scope, r['component'], method].append(values)
            target_rows.append(dict(case_id=r['case_id'], target=target, scope=scope, component=r['component'], method=method, kl=values[0], nll=values[1]))
    cases = []
    for key, vals in groups.items():
        assert len(vals) == (1 if key[-1] == 'raw' else 4)
        cases.append(dict(zip(('case_id','source_seed','source_atom','condition','scope','component','method'), key),
                          **{name: median([v[i] for v in vals if v[i] is not None]) if any(v[i] is not None for v in vals) else None for i,name in enumerate(('kl','nll'))}))
    aggregates = []
    for scope, component, method in sorted({(r['scope'],r['component'],r['method']) for r in cases}):
        rr = [r for r in cases if (r['scope'],r['component'],r['method']) == (scope,component,method)]
        aggregates.append(dict(scope=scope,component=component,method=method,cases=len(rr),
                               **{m:median([r[m] for r in rr if r[m] is not None]) for m in ('kl','nll')}))
    inputs = [run/'metrics.raw.jsonl', controls/'metrics.raw.jsonl', run/'config.resolved.json', Path(__file__)]
    result = dict(scope='Development extension to previously exposed compact panel; frozen maps, no refit; not independent confirmation or duration causality.',
                  source_rows_exact=len(src[0]), source_cases_exact=True, recompute_max_abs_error=max_error,
                  new_forwards=174, reused_control_rows=len(old)-len(src[1]), cases=cases, aggregates=aggregates, target_rows=target_rows,
                  inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    out.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# 素材与稀疏读出的跨面板比较', '', result['scope'], '',
             '六case、三个source query、四个非自身target；先取case内target中位数，再取六case中位数。A/B为共线数学部分。', '',
             '|部分|方法|KL误差|NLL变化误差|','|---|---|---:|---:|']
    for r in aggregates:
        if r['scope'] == 'intervention_positions':
            lines.append(f"|{r['component']}|{r['method']}|{r['kl']:.6g}|{r['nll']:.6g}|")
    (run/'MATERIAL_COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('source_rows_exact','recompute_max_abs_error','aggregates')}))


if __name__ == '__main__':
    main()
