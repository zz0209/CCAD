"""Compare discovery-selected atom to saved distributed maps after exact replay."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read_rows(path): return [json.loads(s) for s in path.read_text().splitlines() if s]
def key(r): return tuple(r[k] for k in ('source_seed','source_atom','target_seed','condition','sequence','rank'))
def median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    run=args.run_dir;cfg=json.loads((run/'config.resolved.json').read_text())
    other=run.parent/cfg['comparison_run']
    for folder in (run,other):
        assert json.loads((folder/'status.json').read_text())['status']=='PASS'
    assert json.loads((run/'selection.json').read_text())==json.loads((other/'selection.json').read_text()),'Selection replay mismatch'
    atom_rows=read_rows(run/'metrics.raw.jsonl');saved=read_rows(other/'metrics.raw.jsonl')
    targets={key(r):r for r in saved if r['method']=='target'}
    assert len(targets)==len(atom_rows)==len({key(r) for r in atom_rows})
    matched=[]
    for a in atom_rows:
        d=targets[key(a)]
        for field in ('common_source_dose_scale','source_natural_hook_energy','recipient_masked_hook_norm','source_hook_fraction','intervention_positions','donor_sequence','donor_positions'):
            assert a[field]==d[field],f'Replay mismatch: {key(a)}, {field}'
        assert a['hook']['source_energy']==d['hook']['source_energy']
        for endpoint in a['endpoints']:
            for field in ('source_energy','source_rms'):
                assert a['endpoints'][endpoint][field]==d['endpoints'][endpoint][field],f'Source endpoint mismatch: {key(a)}, {endpoint}/{field}'
        for endpoint in ('next_state','centered_logits'):
            matched.append({**dict(zip(('source_seed','source_atom','target_seed','condition','sequence','rank'),key(a))),
                            'endpoint':endpoint,'distributed_error':d['endpoints'][endpoint]['normalized_error'],
                            'single_atom_error':a['endpoints'][endpoint]['normalized_error']})
    tables={}
    for name,fields in [('target',('condition','source_seed','source_atom','target_seed','rank','endpoint')),
                        ('query',('condition','source_seed','source_atom','rank','endpoint'))]:
        groups={}
        for row in matched: groups.setdefault(tuple(row[k] for k in fields),[]).append(row)
        table=[]
        for group,rows in sorted(groups.items()):
            distributed=median(r['distributed_error'] for r in rows);atomic=median(r['single_atom_error'] for r in rows)
            table.append({**dict(zip(fields,group)),'distributed_error':distributed,'single_atom_error':atomic,
                          'relative_error_reduction_vs_atom':1-distributed/atomic if distributed is not None and atomic is not None and atomic>0 else None,
                          'distributed_better':distributed<atomic if distributed is not None and atomic is not None else None,
                          'valid_pairs':sum(r['distributed_error'] is not None and r['single_atom_error'] is not None for r in rows),'requested_pairs':len(rows)})
        tables[name]=table
        with (run/f'atom_comparison_{name}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    summary={}
    for condition in ('positive','negative'):
        for endpoint in ('next_state','centered_logits'):
            group=[r for r in tables['query'] if r['condition']==condition and r['endpoint']==endpoint]
            valid=[r for r in group if r['distributed_better'] is not None]
            summary[f'{condition}/{endpoint}']={'requested_queries':len(group),'valid_queries':len(valid),
                'distributed_median':median(r['distributed_error'] for r in valid),
                'single_atom_median':median(r['single_atom_error'] for r in valid),
                'distributed_better_queries':sum(r['distributed_better'] for r in valid),
                'distributed_below_zero':sum(r['distributed_error']<1 for r in valid),
                'single_atom_below_zero':sum(r['single_atom_error']<1 for r in valid)}
    report={'source_replay':'EXACT for selection, hook scale/energy and all source endpoint energies/RMS',
            'single_atom_fit_spec':cfg['single_atom_fit'],
            'matched_rows':len(atom_rows),'summary':summary,'query_table':tables['query'],
            'source_artifacts':{str(path):digest(path) for path in [run/'metrics.raw.jsonl',other/'metrics.raw.jsonl',run/'single_atom_fits.json',run/'selection.json',run/'config.resolved.json']},
            'script_sha256':digest(Path(__file__)),
            'scope':'Development on same v8 documents; query medians over dependent targets/documents, no independent CI. Single-atom source-aligned map, not native deletion. Discovery selection, no target endpoint tuning. All8 queries retained including unsupported negative query.'}
    (run/'atom_comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
