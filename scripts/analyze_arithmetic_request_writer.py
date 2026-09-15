"""Report the complete exposed request-writer pilot, without model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_arithmetic_response import answer

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    run = ROOT/args.run
    status = json.loads((run/'status.json').read_text())
    summary = json.loads((run/'metrics.summary.json').read_text())
    cfg = json.loads((run/'config.resolved.json').read_text())
    rows = [json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()]
    logging_correction = None
    if status['status'] != 'PASS':
        # v1 saved digit identity in component, while the shared uniqueness key
        # uses operation. Preserve its FAIL status and verify this exact defect.
        assert status['error'] is None
        assert {k for k,v in summary['checks'].items() if not v} == {'unique_results'}
        keys = [tuple((r['component'] if k=='operation' and r['kind']=='request_fit' else r.get(k))
                      for k in ['kind','task','row_id','mode','method','seed','target_seed','operation']) for r in rows]
        assert len(keys)==len(set(keys))
        logging_correction = 'Original FAIL retained: request_fit records store digit in component, not operation. All records unique when that explicit digit is included; generation rows are unchanged and independently unique.'
    lookup = {(r['seed'], r['method'], r['operation'], r['row_id']): r for r in rows if r['kind']=='source_patch'}
    assert len(lookup)==sum(r['kind']=='source_patch' for r in rows)
    methods = ['member', 'assignment', 'raw_readout', 'full_activation_readout'] + [
        f'learned_{k}_u{u}' for k in ['native', 'raw'] for u in cfg['request_writers']['updates']]
    nq = cfg['pairs_per_template']
    seed_pairs = cfg['request_writers']['seed_pairs']
    weights = np.random.default_rng(946016).multinomial(nq, np.full(nq, 1/nq), 10000)/nq
    fidelity, cells, replicates = [], [], {}
    for method in methods:
        counts = np.zeros((nq, 4))
        for q in range(nq):
            for form in range(2):
                rid = form*nq+q
                base = answer(lookup[0, 'no_edit', 'unit', rid])
                for s, t in seed_pairs:
                    for op in ['unit', 'tens']:
                        for part in [0, 1]:
                            src = answer(lookup[s, f'source_part{part}_bank0', op, rid])
                            dst = answer(lookup[t, f'{method}_part{part}_bank0', op, rid])
                            change = int(src != base)
                            counts[q, 2*change] += 1
                            counts[q, 2*change+1] += int(src==dst)
        def score(x):
            return .5*(x[..., 1]/x[..., 0]+x[..., 3]/x[..., 2])
        value = score(counts.sum(0))
        rep = score(weights@counts)
        replicates[method] = value, rep
        fidelity.append(dict(method=method, balanced=float(value*100), counts=counts.sum(0).tolist(),
                             interval=(100*np.nanquantile(rep, [.025,.975])).tolist()))
    for method in ['source']+methods:
        for suffix in ['full', 'part0_bank0', 'part1_bank0']:
            for op in ['unit','tens']:
                rr = [lookup[s if method=='source' else t, method+'_'+suffix, op, rid]
                      for s,t in seed_pairs for rid in range(nq*2)]
                cells.append(dict(method=method, query=suffix, operation=op, n=len(rr),
                                  **{name:100*float(np.mean([r[key] for r in rr])) for name,key in
                                     [('H','exact_hybrid'),('T','target_digit_success'),('P','preserve_digit_success')]}))
    contrasts = []
    for method in methods:
        if not method.startswith('learned_'):
            continue
        for other in ['member','raw_readout','full_activation_readout']:
            a, ar = replicates[method]; b, br = replicates[other]
            contrasts.append(dict(contrast=method+' minus '+other, points=float(100*(a-b)),
                                  interval=(100*np.nanquantile(ar-br,[.025,.975])).tolist()))
    diag = json.loads((run/'NATIVE_EXECUTION.json').read_text())['records']
    assert all(r['min_edited_code']>=-1e-6 and r['max_changed_members']<=64 for r in diag)
    fit = json.loads((run/'REQUEST_WRITER_FIT.json').read_text()) if (run/'REQUEST_WRITER_FIT.json').exists() else None
    out = dict(run=args.run, scope=cfg['scope'], run_status=status, logging_correction=logging_correction,
               fidelity=fidelity, cells=cells, contrasts=contrasts,
               fit=fit, diagnostic_calls=len(diag), min_code=min(r['min_edited_code'] for r in diag),
               max_members=max(r['max_changed_members'] for r in diag),
               raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),
               statistics='Question-cluster intervals; forms, parts, digits and specified dependent seed pairs retained. Development checkpoints, no independent-confirmation claim.')
    (ROOT/args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(fidelity=fidelity, full=[r for r in cells if r['query']=='full'], fit=fit)))


if __name__=='__main__':
    main()
