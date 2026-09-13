"""Summarize the actual feature-choice consumer without treating edges as independent."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(run,out):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((run/'config.resolved.json').read_text())
    operation_task={op:cfg['source_tasks'][weights.index(1)] for op,weights in cfg['operations'].items()}
    raw=defaultdict(lambda:dict(n=0,errors=0,clean_correct=0,decrement=0.,edit_norm=0.))
    for line in (run/'metrics.raw.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        if r['kind']!='functional_reuse':continue
        key=tuple(r[k] for k in ['split','objective','seed','operation','method','allowance','task'])
        c=raw[key];c['n']+=1;c['errors']+=r['introduced_error'];c['clean_correct']+=r['clean_margin']>0
        c['decrement']+=r['clean_margin']-r['margin'];c['edit_norm']+=r['edit_norm']
    cells=[];grouped=defaultdict(dict)
    for key,c in raw.items():
        item=dict(zip(['split','objective','seed','operation','method','allowance','task'],key),**c)
        item.update(error_rate=c['errors']/c['n'],margin_decrement=c['decrement']/c['n'],mean_edit_norm=c['edit_norm']/c['n'])
        cells.append(item);grouped[key[:-1]][key[-1]]=item
    summary=[]
    for key,tasks in grouped.items():
        if set(tasks)!=set(cfg['source_tasks']):continue
        lo,hi=cfg['consumer_ranges'][key[0]]
        if any(r['n']!=hi-lo for r in tasks.values()):continue
        own=operation_task[key[3]];other=[t for t in tasks if t!=own]
        target=tasks[own]['error_rate'];collateral=statistics.mean(tasks[t]['error_rate'] for t in other)
        summary.append(dict(zip(['split','objective','seed','operation','method','allowance'],key),requested_error_rate=target,collateral_error_rate=collateral,selectivity=target-collateral,requested_margin_decrement=tasks[own]['margin_decrement'],pairs_per_task=hi-lo))
    selected=[]
    if (run/'SELECTION_FREEZE.json').exists():
        decisions=json.loads((run/'SELECTION_FREEZE.json').read_text())['decisions']
        lookup={(r['objective'],r['seed'],r['operation'],r['method'],r['allowance']):r for r in summary if r['split']=='evaluation'}
        for d in decisions:
            key=(d['objective'],d['source'],d['operation'],d['family'],d['chosen_allowance'])
            if key in lookup:selected.append(dict(**lookup[key],validation_pairs_per_task=d['validation_pairs_per_task'],charged_candidate_pair_interventions=d['charged_candidate_pair_interventions']))
    groups=defaultdict(list)
    for r in selected:groups[r['objective'],r['method'],r['validation_pairs_per_task']].append(r)
    aggregates=[]
    for key,rr in groups.items():
        aggregates.append(dict(zip(['objective','method','validation_pairs_per_task'],key),task_seed_cells=len(rr),selectivity=statistics.mean(r['selectivity'] for r in rr),requested_error_rate=statistics.mean(r['requested_error_rate'] for r in rr),collateral_error_rate=statistics.mean(r['collateral_error_rate'] for r in rr),mean_selected_allowance=statistics.mean(r['allowance'] for r in rr)))
    def save(name,rows):
        if rows:
            with (out/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for n,rows in [('cells.csv',cells),('operations.csv',summary),('selected.csv',selected),('aggregates.csv',aggregates)]:save(n,rows)
    result=dict(run=str(run),status=json.loads((run/'status.json').read_text())['status'],cells=cells,operations=summary,selected=selected,aggregates=aggregates,scope='Equal task/seed-cell descriptive summaries. Five cyclic shared seeds and original lexical pairs remain dependent. Selectivity measures requested introduced errors minus collateral introduced errors; it is a causal feature-choice outcome, not model quality or human annotation agreement.')
    if cfg.get('independent_consensus'):
        result['scope']='Equal means over three fixed functions and five independently initialized target SAEs, conditional on one fixed five-source bank and shared training material/model. Generated pairs are shared across methods and targets. Primary16-pair validation choices precede new evaluation. Structural/union requests have separate summaries.'
        if cfg.get('leave_target_out'):result['scope']='Equal means over three fixed functions and a five-SAE source/target network. Each target excludes its own source annotation; target effects remain dependent through shared source banks. Generated pairs and calibration choices are shared. Region/union requests have separate summaries.'
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('out');a=ap.parse_args()
    r=analyze(a.run,a.out)
    for row in r['aggregates']:print(json.dumps(row))
