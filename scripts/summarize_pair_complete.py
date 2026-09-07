"""Summarize both consumers without selecting successful rows or seed pairs."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    run=args.run;out=args.out;out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((run/'metrics.summary.json').read_text())
    if summary['status']!='PASS':raise ValueError('Only a completed passing run can be summarized')
    contract=json.loads((run/'contract_validation.json').read_text())
    if not contract['ok'] and contract['errors']!=['audit opened without frozen candidate family']:
        raise ValueError('An unreviewed artifact defect needs diagnosis before summarization')
    keys=['factor','consumer','role','method'];group=defaultdict(list);seeds=defaultdict(list)
    panel=json.loads((ROOT/json.loads((run/'config.resolved.json').read_text())['material_run']/'panel.json').read_text())
    count=0;seen=set();examples=[]
    with (run/'metrics.raw.jsonl').open() as f:
        for line in f:
            row=json.loads(line);count+=1
            if row['role']!=panel['rows'][row['row_id']]['cue_role']:raise ValueError('Role metadata differs from panel')
            identity=tuple(row.get(k) for k in ['source_seed','target_seed','factor','consumer','method','row_id'])
            if identity in seen:raise ValueError('Duplicate outcome')
            seen.add(identity);key=tuple(row[k] for k in keys)
            small={k:row[k] for k in ['kl_reference','source_label_agreement','delta_norm','kl_to_baseline']}
            group[key].append(small);seeds[key+(row['source_seed'],)].append(small)
            if row['source_seed']==1 and row['row_id'] in [0,256]:examples.append(row)
    is_anchor=json.loads((run/'config.resolved.json').read_text()).get('pair_anchor_followup',False)
    assert count==summary['rows']==(51200 if is_anchor else 81920)
    def aggregate(rows):
        return dict(n=len(rows),kl=statistics.fmean(r['kl_reference'] for r in rows),
            source_agreement=statistics.fmean(r['source_label_agreement'] for r in rows),
            delta_norm=statistics.fmean(r['delta_norm'] for r in rows),
            kl_to_baseline=statistics.fmean(r['kl_to_baseline'] for r in rows))
    records=[]
    for key,rows in sorted(group.items()):
        sr=[dict(source_seed=s,**aggregate(seeds[key+(s,)])) for s in range(1,6)]
        records.append(dict(zip(keys,key),**aggregate(rows),source_seed_means=sr,
            min_seed_kl=min(r['kl'] for r in sr),max_seed_kl=max(r['kl'] for r in sr)))
    geometry=json.loads((run/'geometry.json').read_text())['rows'];geo=defaultdict(list)
    for row in geometry:geo[tuple(row[k] for k in keys)].append(row)
    grecords=[]
    for key,rows in sorted(geo.items()):
        sums={k:sum(r[k] for r in rows) for k in ['error_sse','reference_energy','candidate_energy']}
        grecords.append(dict(zip(keys,key),**sums,relative_sse=sums['error_sse']/sums['reference_energy']))
    diag=json.loads((run/'fit_diagnostics.json').read_text())['rows']
    result=dict(written_utc=datetime.now(timezone.utc).isoformat(),run=run.as_posix(),run_summary=summary,
        contract_validation=contract,contract_scope_note=('All generic artifact checks passed; previously exposed panel is explicitly development.' if contract['ok'] else 'Preserved metadata failure: exposed audit data used for explicitly nonfrozen retrospective development triggers the generic confirmation gate. All remaining contract checks and computational checks passed. No reclassification as independent confirmation and no run metadata rewritten.'),
        rows=records,geometry=grecords,fit_diagnostics=diag,fixed_examples=examples,
        raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),
        scope='All 512 exposed inputs, five cyclic directions sharing five seeds. Fits on old 384 temporal rows only. Retrospective development, not independent confirmation. KL references each same-source same-consumer intervention. No semantic accuracy for deletion. Seed ranges are descriptive, not confidence intervals.',
        consumer_input_requirement=('Recipient and reciprocal donor needed for removal; odd map frozen. Not an unpaired context-to-Dz function.' if is_anchor else 'Single-recipient codes predict complete-group removal; reciprocal pair codes predict contrasts.'))
    stem='R9_PAIR_ANCHOR' if is_anchor else 'R9_PAIR_COMPLETE'
    (out/(stem+'_SUMMARY.json')).write_text(json.dumps(result,indent=2)+'\n')
    lookup={tuple(r[k] for k in keys):r for r in records};comparisons=[]
    for factor in ['number','time']:
        for role in ['temporal','quoted']:
            for method in (['anchor_compact','anchor_full_code','anchor_raw'] if is_anchor else ['legacy_plus_mean','paired_complete','paired_balanced','full_code_complete','raw_complete','same_members_native']):
                item=dict(factor=factor,role=role,method=method)
                for consumer in ['contrast','complete_removal']:
                    old=lookup[factor,consumer,role,'legacy_contrast'];new=lookup[factor,consumer,role,method]
                    item[consumer+'_kl']=new['kl'];item[consumer+'_legacy_kl']=old['kl']
                    item[consumer+'_reduction_fraction']=1-new['kl']/old['kl']
                    item[consumer+'_improved_seed_count']=sum(n['kl']<o['kl'] for n,o in zip(new['source_seed_means'],old['source_seed_means']))
                comparisons.append(item)
    (out/('R9_ANCHOR_COMPARISONS.json' if is_anchor else 'R9_COMPARISONS.json')).write_text(json.dumps(dict(rows=comparisons),indent=2)+'\n')
    print(json.dumps(dict(rows=count,groups=len(records),comparisons=comparisons),indent=2))


if __name__=='__main__':main()
