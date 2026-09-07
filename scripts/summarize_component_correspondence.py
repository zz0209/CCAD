"""Summarize all retained source-mask interventions, preserving seed dependence."""
import argparse,collections,datetime,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    run=args.run if args.run.is_absolute() else ROOT/args.run
    groups=collections.defaultdict(list);seedgroups=collections.defaultdict(list);cases=[];raw=run/'metrics.raw.jsonl';count=0;digest=hashlib.sha256()
    strata=collections.defaultdict(lambda:[0,0.]);interactions={}
    for line in raw.open('rb'):
        digest.update(line)
        row=json.loads(line);count+=1
        key=tuple(row[k] for k in ['factor','consumer','dose','role','mask','method'])
        value=dict(kl=row['kl_reference'],number_error=abs(row['number_logodds']-row['source_number_logodds']),time_error=abs(row['past_logodds']-row['source_past_logodds']),source_number_effect=abs(row['source_number_change']),source_time_effect=abs(row['source_past_change']),label_agreement=row['label']==row['source_label'])
        groups[key].append(value);seedgroups[key+(row['source_seed'],)].append(value)
        if row['source_seed']==3 and row['row_id'] in [0,256] and row['factor']=='time':cases.append(row)
        if row['dose']==1 and row['mask']!='whole':
            for kind,level in [('lexical_block',row['block']),('cue_family','new' if row['cue_id']>=2 else 'known')]:
                sk=tuple(row[k] for k in ['factor','consumer','role','method','source_seed'])+(kind,level)
                strata[sk][0]+=1;strata[sk][1]+=row['kl_reference']
        if row['dose']==1 and row['mask'] in ['whole','rank_first_half','rank_second_half']:
            ik=tuple(row[k] for k in ['factor','consumer','role','method','source_seed','row_id'])
            interactions.setdefault(ik,{})[row['mask']]=[row[k] for k in ['number_change','past_change','source_number_change','source_past_change']]
    avg=lambda rows,k:sum(r[k] for r in rows)/len(rows)
    rows=[]
    for key,values in groups.items():
        means=[dict(source_seed=seed,**{k:avg(seedgroups[key+(seed,)],k) for k in values[0]}) for seed in range(1,6)]
        rows.append(dict(zip(['factor','consumer','dose','role','mask','method'],key),n=len(values),**{k:avg(values,k) for k in values[0]},source_seed_means=means))
    pooled=collections.defaultdict(list)
    for row in rows:
        key=tuple(row[k] for k in ['factor','consumer','dose','role','method'])+('whole' if row['mask']=='whole' else 'components',)
        pooled[key].append(row)
    pooled_rows=[]
    for key,values in pooled.items():
        means=[dict(source_seed=seed,kl=sum(next(s['kl'] for s in r['source_seed_means'] if s['source_seed']==seed) for r in values)/len(values)) for seed in range(1,6)]
        pooled_rows.append(dict(zip(['factor','consumer','dose','role','method','mask_family'],key),masks=len(values),n=sum(r['n'] for r in values),kl=avg(values,'kl'),number_error=avg(values,'number_error'),time_error=avg(values,'time_error'),source_number_effect=avg(values,'source_number_effect'),source_time_effect=avg(values,'source_time_effect'),label_agreement=avg(values,'label_agreement'),source_seed_means=means))
    lookup={tuple(r[k] for k in ['factor','consumer','dose','role','mask_family','method']):r for r in pooled_rows};comparisons=[]
    cfg=json.loads((run/'config.resolved.json').read_text());seed_pairs=cfg['seed_pairs']
    for key,row in lookup.items():
        if key[-1] in ['source','aggregate_ols']:continue
        ref=lookup[key[:-1]+('aggregate_ols',)]
        byseed={r['source_seed']:r['kl'] for r in row['source_seed_means']};refseed={r['source_seed']:r['kl'] for r in ref['source_seed_means']}
        loo=[]
        for seed in range(1,6):
            keep=[source for source,target in seed_pairs if seed not in [source,target]]
            loo.append(dict(removed_seed=seed,retained_directions=len(keep),primary_minus_comparator=sum(refseed[s]-byseed[s] for s in keep)/len(keep)))
        comparisons.append(dict(factor=row['factor'],consumer=row['consumer'],dose=row['dose'],role=row['role'],mask_family=row['mask_family'],method=row['method'],reference='aggregate_ols',kl=row['kl'],reference_kl=ref['kl'],relative_reduction=1-row['kl']/ref['kl'],primary_relative_reduction=1-ref['kl']/row['kl'],directions_better=sum(a['kl']<b['kl'] for a,b in zip(row['source_seed_means'],ref['source_seed_means'])),primary_directions_better=sum(b['kl']<a['kl'] for a,b in zip(row['source_seed_means'],ref['source_seed_means'])),leave_incident_seed_out=loo))
    interaction_groups=collections.defaultdict(list)
    for key,values in interactions.items():
        assert len(values)==3
        v=[values['whole'][j]-values['rank_first_half'][j]-values['rank_second_half'][j] for j in range(4)]
        interaction_groups[key[:-1]].append(dict(number_interaction=v[0],time_interaction=v[1],source_number_interaction=v[2],source_time_interaction=v[3],absolute_source_number_interaction=abs(v[2]),absolute_source_time_interaction=abs(v[3]),number_error=abs(v[0]-v[2]),time_error=abs(v[1]-v[3])))
    interaction_rows=[dict(zip(['factor','consumer','role','method','source_seed'],key),n=len(values),**{k:avg(values,k) for k in values[0]}) for key,values in interaction_groups.items()]
    strata_rows=[dict(zip(['factor','consumer','role','method','source_seed','stratum','level'],key),n=v[0],kl=v[1]/v[0]) for key,v in strata.items()]
    payload=dict(written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(run.relative_to(ROOT)),raw_rows=count,raw_sha256=digest.hexdigest(),run_summary=json.loads((run/'metrics.summary.json').read_text()),rows=rows,pooled=pooled_rows,comparisons=comparisons,fixed_cases=cases,strata=strata_rows,interactions=interaction_rows,scope='Equal authored rows within each mask, then equal mask/seed means. Five cyclic directions share SAEs; no independent-seed confidence claim. All source errors/inactive cases retained. Component family is the six declared non-whole masks, not an exhaustive source-subset universe. Higher-dose component means contain only firsthalf and must not be treated as the full six-mask family.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+'\n')
    brief=[c for c in comparisons if c['method'] in ['component_half','component_singleton'] and c['mask_family']=='components' and c['dose']==1]
    print(json.dumps(dict(rows=count,summary=str(args.output),comparisons=brief)))


if __name__=='__main__':main()
