"""Summarize frozen city outcomes with control and seed dependence retained."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import fmean
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SINGLES={'100','010','001'}


def summarize(run):
    summary=json.loads((run/'metrics.summary.json').read_text())
    contract=json.loads((run/'contract_validation.json').read_text())
    if summary['status']!='PASS' or not contract['ok']:
        raise ValueError('Only completed valid runs can enter the confirmation summary')
    cfg=json.loads((run/'config.resolved.json').read_text())
    freeze=json.loads((ROOT/cfg['freeze_path']).read_text())
    raw=run/'metrics.raw.jsonl';digest=hashlib.sha256()
    buckets=defaultdict(lambda:dict(cause_sum=0,iso_sum=0,cause_n=0,iso_n=0,kl=0.,norm=0.,n=0))
    fixed=json.loads((run/'fixed_example.json').read_text())['row']
    panel=json.loads((run/'panel.json').read_text())['rows']
    example_ids={}
    for i,p in enumerate(panel):
        if p['component']==fixed['component'] and p['entity']==fixed['entity']:
            example_ids.setdefault(p['task'],i)
    examples=[];n=0
    with raw.open('rb') as handle:
        for line in handle:
            digest.update(line);r=json.loads(line);n+=1
            if r['kind']!='semantic' or r['split']!='confirmation':
                raise ValueError('Confirmation output includes an unexpected row')
            key=(r['seed'],r['target_seed'],r['method'],r['operation'],r['component'])
            b=buckets[key];endpoint=r['endpoint'].lower()
            b[endpoint+'_sum']+=int(r['first_token_correct']);b[endpoint+'_n']+=1
            b['kl']+=0. if r['method']=='frozen_source' else r['kl_to_reference']
            b['norm']+=r['edit_norm'];b['n']+=1;b['control']=r['control']
            if r['row_id'] in example_ids.values() and r['operation'] in SINGLES|{'111'}:
                p=panel[r['row_id']]
                examples.append(dict(**r,text=p['text'],base_label=p['label'].strip(),donor_label=p['donor_label'].strip()))
    if digest.hexdigest()!=summary['metrics_raw_sha256'] or n!=summary['rows']:
        raise ValueError('Raw output identity/row count differs')
    by_cell=defaultdict(list)
    for (s,t,m,o,pair),b in buckets.items():
        cause=b['cause_sum']/b['cause_n'] if b['cause_n'] else None
        iso=b['iso_sum']/b['iso_n'] if b['iso_n'] else None
        score=fmean(v for v in [cause,iso] if v is not None)
        by_cell[s,t,m,o].append(dict(component=pair,cause=cause,iso=iso,score=score,
            source_kl=b['kl']/b['n'],mean_edit_norm=b['norm']/b['n'],rows=b['n'],control=b['control']))
    controls=[]
    for (s,t,m,o),pairs in by_cell.items():
        if len(pairs)!=freeze['confirmation_city_pairs']:
            raise ValueError('Missing city pairs in one method/control')
        row=dict(source_seed=s,target_seed=t,method=m,operation=o,control=pairs[0]['control'],per_pair=pairs)
        for metric in ['cause','iso','score','source_kl','mean_edit_norm']:
            vals=[p[metric] for p in pairs]
            row[metric]=fmean(vals) if all(v is not None for v in vals) else None
        controls.append(row)
    families=[]
    grouped=defaultdict(list)
    for r in controls:
        group='single' if r['operation'] in SINGLES else ('joint' if all(v in [0,1] for v in r['control']) else 'fractional')
        grouped[r['source_seed'],r['target_seed'],r['method'],group].append(r)
        if group!='fractional':grouped[r['source_seed'],r['target_seed'],r['method'],'binary'].append(r)
    for (s,t,m,group),cells in grouped.items():
        if group=='binary' and len(cells)!=len(cfg['binary_controls']):
            # The wrong-attribute control is deliberately defined only for
            # singleton requests, so it is not a seven-request comparator.
            continue
        pairs=[]
        for p in sorted({v['component'] for r in cells for v in r['per_pair']}):
            values=[next(v for v in r['per_pair'] if v['component']==p) for r in cells]
            pairs.append(dict(component=p,score=fmean(v['score'] for v in values),source_kl=fmean(v['source_kl'] for v in values)))
        families.append(dict(source_seed=s,target_seed=t,method=m,family=group,controls=len(cells),per_pair=pairs,
            score=fmean(r['score'] for r in cells),source_kl=fmean(r['source_kl'] for r in cells),
            cause=fmean(r['cause'] for r in cells if r['cause'] is not None),
            iso=fmean(r['iso'] for r in cells if r['iso'] is not None) if any(r['iso'] is not None for r in cells) else None))
    pooled=[]
    for m,group in sorted({(r['method'],r['family']) for r in families}):
        rr=[r for r in families if r['method']==m and r['family']==group]
        pooled.append(dict(method=m,family=group,directions=len(rr),
            **{k:fmean(r[k] for r in rr) if all(r[k] is not None for r in rr) else None for k in ['cause','iso','score','source_kl']}))
    comparisons=[]
    pairs=[('adaptive_native256','target_reencode'),('adaptive_native_reencode_count','target_reencode'),
        ('adaptive_native256','random_base_native256'),('adaptive_native512','target_reencode'),
        ('direct_shared_space','atom_pw_mcc'),('direct_shared_space','raw_supervised_family'),
        ('direct_shared_space','raw_supervised_singleton'),
        ('direct_shared_space','raw_shared_space'),('direct_shared_space','wrong_attribute_norm_matched')]
    for method,comparator in pairs:
        for group in ['single','binary','fractional']:
            left=[r for r in families if r['method']==method and r['family']==group]
            right=[r for r in families if r['method']==comparator and r['family']==group]
            if not left or not right:continue
            for metric in (['source_kl'] if group=='fractional' else ['score','source_kl']):
                pairids=sorted({v['component'] for r in left for v in r['per_pair']})
                differences=[];edge_values=[]
                for r in left:
                    ref=next(v for v in right if (v['source_seed'],v['target_seed'])==(r['source_seed'],r['target_seed']))
                    delta=[next(v[metric] for v in r['per_pair'] if v['component']==p)-next(v[metric] for v in ref['per_pair'] if v['component']==p) for p in pairids]
                    differences.append(delta);edge_values.append(dict(source_seed=r['source_seed'],target_seed=r['target_seed'],difference=fmean(delta)))
                matrix=np.asarray(differences).T
                rng=np.random.default_rng(180908);sample=rng.integers(0,len(matrix),(10000,len(matrix)))
                boots=matrix.mean(1)[sample].mean(1)
                deletions=[]
                for seed in range(1,6):
                    values=[r['difference'] for r in edge_values if r['source_seed']!=seed and r['target_seed']!=seed]
                    deletions.append(dict(omitted_seed=seed,remaining_directions=len(values),difference=fmean(values)))
                comparisons.append(dict(method=method,comparator=comparator,family=group,metric=metric,
                    difference=float(matrix.mean()),conditional_city_pair_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),
                    directions=edge_values,leave_incident_seed_out=deletions,
                    interpretation='Positive score difference favors method; negative KL difference favors method. Cities are resampled jointly across all five dependent edges.'))
    diagnostics=json.loads((run/'native_writer_diagnostics.json').read_text())['rows'];diag=[]
    for s,t,m in sorted({(r['source_seed'],r['target_seed'],r['method']) for r in diagnostics}):
        rr=[r for r in diagnostics if (r['source_seed'],r['target_seed'],r['method'])==(s,t,m)]
        diag.append(dict(source_seed=s,target_seed=t,method=m,
            candidate_members=fmean(v for r in rr for v in r.get('candidate_members',r['changed_members'])),
            changed_members=fmean(v for r in rr for v in r['changed_members']),
            relative_squared_writer_error=sum(v for r in rr for v in r.get('squared_error',r.get('squared_writer_error',[])))/sum(v for r in rr for v in r['desired_energy']),
            minimum_final_state=min(r['minimum_final_state'] for r in rr),
            max_relative_projected_gradient=max([v for r in rr for v in r.get('relative_projected_gradient',[])] or [0.]),
            converged_rows=sum(r.get('converged_rows',0) for r in rr),
            solver_rows=sum(r.get('rows',0) for r in rr),solve_seconds=sum(r.get('solve_seconds',0) for r in rr),
            shared_path_selection_seconds=sum(r.get('selection_seconds') or 0 for r in rr),
            timing_scope='One shared maximum-size path is repeated across adaptive prefixes; do not sum selection time across methods.'))
    source_replication=[];source_inputs=[]
    from summarize_semantic_sources import measured_cells
    for spec in freeze['sources']:
        sr=ROOT/spec['run'];sp=sr/'family_source_summary.json';rp=sr/'metrics.raw.jsonl';mp=sr/'metrics.summary.json'
        detail=next(v for v in json.loads(sp.read_text())['methods'] if v['method']==spec['family'])
        raw_source=rp.read_bytes();raw_summary=json.loads(mp.read_text())
        if hashlib.sha256(raw_source).hexdigest()!=raw_summary['metrics_raw_sha256']:
            raise ValueError('Source replica raw identity differs')
        held=measured_cells([json.loads(v) for v in raw_source.splitlines()],spec['family'])
        source_replication.append(dict(source_seed=spec['source_seed'],target_seed=spec['target_seed'],
            family=spec['family'],updates=detail['updates'],selected_step=detail['selected_step'],
            calibration_family=detail['selected']['family_score'],calibration_single=detail['selected']['single_score'],
            held_family=fmean(v['score'] for v in held),held_single=fmean(v['score'] for v in held[:3]),
            held_controls=held,calibration_trace=detail['trace']))
        source_inputs.extend(p.relative_to(ROOT).as_posix() for p in [sp,rp,mp])
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run_id=run.name,raw_rows=n,raw_sha256=digest.hexdigest(),
        freeze_path=cfg['freeze_path'],freeze_sha256=cfg['freeze_sha256'],controls=controls,families=families,pooled=pooled,
        comparisons=comparisons,native_diagnostics=diag,source_replication=source_replication,examples=examples,fixed_example=fixed,
        input_summary_paths=[(run/p).relative_to(ROOT).as_posix() for p in ['metrics.raw.jsonl','metrics.summary.json','config.resolved.json','native_writer_diagnostics.json','source_replay_witness.json','fixed_example.json','panel.json']]+[cfg['freeze_path']]+source_inputs,
        scope='New intervention cities from an already screened original-training population and templates, not official RAVEL test or new model. Seven binary controls are weighted equally; triple control has Cause only. Fractional categorical scores are descriptive. Conditional city-pair bootstrap preserves all seed edges; seed sensitivity removes both incident edges, not an independent-seed confidence interval.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    result=summarize(args.run.resolve());args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),pooled=result['pooled'])))


if __name__=='__main__':main()
