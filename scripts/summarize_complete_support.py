"""All-row descriptive comparison for complete-support experiments."""
import argparse, json, hashlib, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    run=args.run;summary=json.loads((run/'metrics.summary.json').read_text());contract=json.loads((run/'contract_validation.json').read_text())
    assert summary['status']=='PASS' and contract['ok']
    groups=defaultdict(list);seen=set();examples=[]
    for line in (run/'metrics.raw.jsonl').read_text().splitlines():
        r=json.loads(line);identity=tuple(r[k] for k in ['source_seed','target_seed','factor','consumer','method','row_id'])
        assert identity not in seen;seen.add(identity)
        key=tuple(r[k] for k in ['factor','consumer','role','method']);groups[key].append(r)
        if r['source_seed']==1 and r['row_id'] in [0,256]:examples.append(r)
    assert len(seen)==summary['rows']
    records=[]
    for key,rows in sorted(groups.items()):
        seeds=sorted({r['source_seed'] for r in rows});means=[]
        for seed in seeds:
            ss=[r for r in rows if r['source_seed']==seed]
            means.append(dict(source_seed=seed,n=len(ss),kl=statistics.fmean(r['kl_reference'] for r in ss),source_agreement=statistics.fmean(r['source_label_agreement'] for r in ss)))
        records.append(dict(zip(['factor','consumer','role','method'],key),n=len(rows),kl=statistics.fmean(r['kl'] for r in means),source_seed_means=means,min_seed_kl=min(r['kl'] for r in means),max_seed_kl=max(r['kl'] for r in means)))
    lookup={(r['factor'],r['consumer'],r['role'],r['method']):r for r in records};comparisons=[]
    for r in records:
        if r['method']=='source':continue
        legacy=lookup[r['factor'],r['consumer'],r['role'],'legacy_contrast']
        for control in ['legacy_contrast','old_support_balanced','dense_complete','full_code_complete','raw_complete']:
            baseline=lookup[r['factor'],r['consumer'],r['role'],control]
            paired=list(zip(r['source_seed_means'],baseline['source_seed_means']))
            comparisons.append(dict(factor=r['factor'],consumer=r['consumer'],role=r['role'],method=r['method'],control=control,kl_difference=r['kl']-baseline['kl'],relative_reduction=1-r['kl']/baseline['kl'],directions_improved=sum(a['kl']<b['kl'] for a,b in paired)))
    result=dict(written_utc=datetime.now(timezone.utc).isoformat(),run=run.as_posix(),run_summary=summary,raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),rows=records,comparisons=comparisons,
        fit_diagnostics=json.loads((run/'fit_diagnostics.json').read_text()),geometry=json.loads((run/'geometry.json').read_text()),fixed_examples=examples,
        scope='All retained inputs and five cyclic directions, sharing five SAEs. Equal direction means and ranges are descriptive, not independent replicates or confidence intervals. Complete removal has no semantic expected label. Selection/fit use old temporal data; role-panel outcomes are retrospective development.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    for consumer in ['contrast','complete_removal']:
        print(consumer)
        for method in sorted({r['method'] for r in records if r['method']!='source'}):
            print(method, [round(lookup[f,consumer,role,method]['kl'],6) for f in ['number','time'] for role in ['temporal','quoted']])


if __name__=='__main__':main()
