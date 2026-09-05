"""Describe an explicitly development-selected source-only applicability range."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None
def pair_key(r): return tuple(r[k] for k in ('source_seed','source_atom','rank','condition','sequence'))
def selected(row,rule):
    return bool(row['supported'] and row['largest_atom_energy_share'] is not None and
                row['largest_atom_energy_share']<=rule['maximum_largest_source_atom_energy_share'] and
                row['natural_source_hook_fraction'] is not None and
                row['natural_source_hook_fraction']>=rule['minimum_natural_source_hook_fraction'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['analysis_id'];run.mkdir(exist_ok=False)
    (run/'config.resolved.json').write_text(json.dumps(cfg,indent=2)+'\n')
    provenance={str(args.config):digest(args.config),str(Path(__file__)):digest(Path(__file__))}
    selections=[];groups={};coverage=[]
    # Materialize and save source-only choices for all panels before opening endpoint rows.
    by_run={}
    for name in cfg['run_names']:
        folder=ROOT/'runs'/name;path=folder/'source_participation.raw.jsonl'
        source=read(path);provenance[str(path)]=digest(path)
        assert len(source)==len({pair_key(r) for r in source})
        by_run[name]={pair_key(r):selected(r,cfg['rule']) for r in source}
        selections.extend(dict(run=name,selected=by_run[name][pair_key(r)],**r) for r in source)
        for s,a,c in sorted({(r['source_seed'],r['source_atom'],r['condition']) for r in source}):
            rows=[r for r in source if (r['source_seed'],r['source_atom'],r['condition'])==(s,a,c)]
            coverage.append(dict(run=name,source_seed=s,source_atom=a,condition=c,requested_pairs=len(rows),
                                 supported_pairs=sum(r['supported'] for r in rows),selected_pairs=sum(by_run[name][pair_key(r)] for r in rows)))
    (run/'source_selection.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selections))
    for name in cfg['run_names']:
        folder=ROOT/'runs'/name;path=folder/'metrics.raw.jsonl';provenance[str(path)]=digest(path)
        for row in read(path):
            choice=by_run[name][pair_key(row)]
            for endpoint in cfg['endpoints']:
                for subset in ('all','selected' if choice else 'rejected'):
                    key=(name,subset,row['condition'],row['source_seed'],row['source_atom'],row['target_seed'],row['method'],endpoint)
                    groups.setdefault(key,[]).append(row['endpoints'][endpoint]['normalized_error'])
    targets=[];query_groups={}
    fields=('run','subset','condition','source_seed','source_atom','target_seed','method','endpoint')
    for key,values in sorted(groups.items()):
        record=dict(zip(fields,key));record.update(error=median(values),valid_pairs=sum(v is not None for v in values),requested_pairs=len(values));targets.append(record)
        qkey=key[:5]+key[6:];query_groups.setdefault(qkey,[]).append(record)
    queries=[]
    for key,records in sorted(query_groups.items()):
        errors=[r['error'] for r in records]
        queries.append(dict(zip(('run','subset','condition','source_seed','source_atom','method','endpoint'),key))|
                       {'error':median(errors),'valid_targets':sum(v is not None for v in errors),'min_target_error':min((v for v in errors if v is not None),default=None),'max_target_error':max((v for v in errors if v is not None),default=None)})
    for name,table in [('coverage',coverage),('target',targets),('query',queries)]:
        with (run/f'{name}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result={'scope':cfg['evidence_level'],'total_pairs':len(selections),'selected_pairs':sum(r['selected'] for r in selections),
            'inputs':provenance,'selected_target_method_query_rows':[r for r in queries if r['subset']=='selected' and r['method']=='target'],
            'limitations':'Previously exposed development documents and outcomes informed this rule. Source-only application does not make its development independent. Panels share seed models; no independent CI. Missing/rejected rows retained in separate outputs.'}
    (run/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
