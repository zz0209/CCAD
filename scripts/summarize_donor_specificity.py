"""Summarize the exposed-panel donor diagnostic, preserving seed dependence."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['kl', 'accuracy', 'teacher_agreement', 'abs_number_shift', 'abs_time_shift',
           'signed_number_shift', 'signed_time_shift', 'delta_norm']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); run = ROOT / args.run; out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run/'metrics.summary.json').read_text())
    if summary['status'] != 'PASS': raise ValueError('Only a complete run may be summarized')
    cfg = json.loads((run/'config.resolved.json').read_text())
    panel = json.loads((ROOT/cfg['material_run']/'panel.json').read_text())
    rows = panel['rows']; baseline = {}
    for line in (ROOT/cfg['material_run']/'metrics.raw.jsonl').open():
        row=json.loads(line)
        if row['kind']=='baseline': baseline[row['row_id']]=row
    teachers={}; groups=defaultdict(lambda:dict(n=0, **{k:0. for k in METRICS}))
    examples=[]; raw_count=0
    for line in (run/'metrics.raw.jsonl').open():
        row=json.loads(line); raw_count+=1; i=row['row_id']; p=rows[i]; base=baseline[i]
        tkey=row['source_seed'],row['factor'],i
        if row['method']=='source_teacher': teachers[tkey]=row
        teacher=teachers[tkey]
        dn=row['number_logodds']-base['number_logodds']; dtime=row['past_logodds']-base['past_logodds']
        values=dict(kl=0. if row['method']=='source_teacher' else row['kl_reference'],accuracy=float(row['correct']),
                    teacher_agreement=float(row['label']==teacher['label']),abs_number_shift=abs(dn),abs_time_shift=abs(dtime),
                    signed_number_shift=dn*(1 if rows[panel['pairs'][i]['number']]['number'] else -1),
                    signed_time_shift=dtime*(1 if rows[panel['pairs'][i]['time']]['past'] else -1),delta_norm=row['delta_norm'])
        role=p['cue_role']; cue='familiar_cue' if p['cue_id']<2 else 'new_cue'
        for part in ['all',role,role+'/'+cue,role+'/'+cue+'/'+p['template']]:
            key=part,row['factor'],row['method'],row['source_seed'],row.get('target_seed',0),p['block']
            groups[key]['n']+=1
            for metric,value in values.items(): groups[key][metric]+=value
        if row['source_seed']==1 and row.get('target_seed',2)==2 and p['block']==0 and p['number']==p['past']==p['distractor']==0:
            examples.append(dict(text=p['text'],cue_role=role,donor_text=rows[panel['pairs'][i][row['factor']]]['text'],baseline=base,**row))
    assert raw_count==summary['rows']
    fields=['partition','factor','method','source_seed','target_seed','block']
    blocks=[dict(zip(fields,key),**value) for key,value in groups.items()]
    def average(items):
        n=sum(x['n'] for x in items)
        return dict(n=n,**{k:sum(x[k] for x in items)/n for k in METRICS})
    by_edge=defaultdict(list); by_method=defaultdict(list)
    for row in blocks:
        by_edge[tuple(row[k] for k in fields[:-1])].append(row)
        by_method[tuple(row[k] for k in fields[:3])].append(row)
    edges=[dict(zip(fields[:-1],key),**average(items)) for key,items in by_edge.items()]
    pooled=[]
    for key,items in by_method.items():
        sources=defaultdict(list)
        for row in items: sources[row['source_seed']].append(row)
        seed_means=[dict(source_seed=s,**average(rr)) for s,rr in sorted(sources.items())]
        pooled.append(dict(zip(fields[:3],key),**average(items),source_seed_means=seed_means,
                           source_seed_kl_min=min(r['kl'] for r in seed_means),source_seed_kl_max=max(r['kl'] for r in seed_means)))
    edge_lookup={tuple(row[k] for k in fields[:-1]):row for row in edges}
    comparisons=[]
    for part,factor in sorted({(r['partition'],r['factor']) for r in edges}):
        intact=[r for r in edges if r['partition']==part and r['factor']==factor and r['method']=='fcc_group']
        for method in ['fcc_wrong_donor','fcc_wrong_donor_norm_matched']:
            paired=[]
            for row in intact:
                other=edge_lookup[part,factor,method,row['source_seed'],row['target_seed']]
                paired.append(dict(source_seed=row['source_seed'],target_seed=row['target_seed'],
                                   excess_kl=other['kl']-row['kl'],accuracy_loss=row['accuracy']-other['accuracy']))
            loso=[]
            for seed in range(1,6):
                retained=[x for x in paired if x['source_seed']!=seed and x['target_seed']!=seed]
                loso.append(dict(excluded_seed=seed,directions=len(retained),excess_kl=sum(x['excess_kl'] for x in retained)/len(retained)))
            comparisons.append(dict(partition=part,factor=factor,comparator=method,directions=len(paired),
                                    intact_lower_kl_directions=sum(x['excess_kl']>0 for x in paired),
                                    mean_excess_kl=sum(x['excess_kl'] for x in paired)/len(paired),
                                    mean_accuracy_loss=sum(x['accuracy_loss'] for x in paired)/len(paired),
                                    leave_one_seed_out=loso,paired_directions=paired))
    for name,records in [('donor_directions',edges),('donor_block_sums',blocks)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    result=dict(run=run.relative_to(ROOT).as_posix(),run_summary=summary,rows=pooled,comparisons=comparisons,
                fixed_examples=examples,checks=json.loads((run/'donor_checks.json').read_text()),
                geometry=json.loads((run/'donor_geometry.json').read_text()),
                raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),
                statistics='All512 exposed inputs, all20 shared-seed directions. Source means average all four targets. Leave-one-seed-out removes every direction incident on the held-out seed, leaving12; this is robustness, not a confidence interval or20 independent replicates. No outcomes used for support/map/donor selection. Source KL in plotted tables is definitionally zero; raw source rows retain full-donor KL. Other methods use source-teacher KL.')
    (out/'R7_DONOR_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for row in pooled:
        if row['partition'] in ['temporal','quoted'] and row['method']!='source_teacher':
            print(json.dumps({k:row[k] for k in ['partition','factor','method','kl','accuracy','abs_time_shift','delta_norm']}))


if __name__=='__main__': main()
