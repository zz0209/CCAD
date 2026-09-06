"""Add exact global-matching interventions to the complete existing request panel."""
import json
from pathlib import Path
from collections import defaultdict
from analyze_f4_rejected_panel import collect,median,csv_write,write,METRICS,SCOPES,ROOT
from build_f4_common_panel import figure
from hashlib import sha256

OUT=ROOT/'runs/F4_global_matching_expanded_v1_20260906'
NEW=['global_matching_geometric','global_matching_pair_calibrated']
METHODS=['target','single_atom_dynamic','raw']+NEW


def main():
    requests,flat,oldcases,oldsummary=collect();inputs=list(oldsummary['inputs']);oldraw={};new_forwards=0;wall=0
    def read(path,lines=False):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;data=p.read_bytes()
        inputs.append(dict(path=str(p),sha256=sha256(data).hexdigest()))
        return [json.loads(s) for s in data.splitlines()] if lines else json.loads(data)
    for path in {r['artifact'] for r in flat if r['artifact']}:
        panel='original' if '_original_' in path else 'expanded'
        for row in read(path,True):
            if row['method']=='target':oldraw[panel,row['source_seed'],row['source_atom'],row['condition'],row['target_seed']]=row
    newrows={}
    for panel in ['original','expanded']:
        folder=f'runs/F4_global_matching_{panel}_v1_20260906'
        assert read(folder+'/status.json')['status']=='PASS'
        summary=read(folder+'/metrics.summary.json');new_forwards+=summary['model_forwards'];wall+=summary['wall_seconds']
        cfg=read(folder+'/config.resolved.json');oldcfg=read(f'runs/F4_rejected_probability_{panel}_v1_20260906/config.resolved.json')
        for k in ['factors_sha256','surface_sha256','token_manifest_sha256','maximum_source_hook_fraction','probability_endpoints']:
            assert cfg[k]==oldcfg[k]
        for row in read(folder+'/metrics.raw.jsonl',True):
            old=oldraw[panel,row['source_seed'],row['source_atom'],row['condition'],row['target_seed']]
            for k in ['sequence','donor_sequence','intervention_positions','donor_positions','document_ids','donor_document_ids','common_source_dose_scale']:
                assert row[k]==old[k],k
            for scope in SCOPES:
                for k in ['source_nll_deltas','source_to_baseline_kl','positions','observed_next_token_ids']:
                    assert row['probability_endpoints'][scope][k]==old['probability_endpoints'][scope][k],k
            key=f"{panel}:{row['source_seed']}:{row['source_atom']}:{row['condition']}",row['target_seed'],row['method']
            assert key not in newrows;newrows[key]=(row,folder+'/metrics.raw.jsonl')
    for q in requests:
        for scope in SCOPES:
            for target in q['targets']:
                template=next(r for r in flat if r['request_id']==q['request_id'] and r['scope']==scope and r['target_seed']==target and r['method']=='target')
                for method in NEW:
                    item=newrows.get((q['request_id'],target,method));raw,path=item if item else ({},None)
                    assert bool(item)==bool(q['entry'])
                    p=raw.get('probability_endpoints',{}).get(scope,{})
                    row=dict(template,method=method,artifact=path,status=p.get('status','NO_DONOR'))
                    for key in METRICS+['candidate_error_kl_mean','nll_delta_rmse']:row[key]=p.get(key)
                    flat.append(row)
    cases=[]
    for q in requests:
        for scope in SCOPES:
            group=[r for r in flat if r['request_id']==q['request_id'] and r['scope']==scope]
            cases.append(dict(request_id=q['request_id'],scope=scope,status=q['status'],
                              medians={m:{k:median([r[k] for r in group if r['method']==m]) for k in METRICS+['candidate_error_kl_mean','nll_delta_rmse']} for m in METHODS}))
    summaries=[]
    for scope in SCOPES:
        for label,states in [('selected',{'SELECTED'}),('rejected',{'SOURCE_RULE_REJECTED'}),('all_matched',{'SELECTED','SOURCE_RULE_REJECTED'})]:
            group=[c for c in cases if c['scope']==scope and c['status'] in states]
            summary=dict(scope=scope,group=label,cases=len(group),methods={},fcc_wins={},fcc_comparisons={})
            for method in METHODS:
                summary['methods'][method]={k:median([c['medians'][method][k] for c in group]) for k in METRICS+['candidate_error_kl_mean','nll_delta_rmse']}
            for method in METHODS[1:]:
                summary['fcc_wins'][method]={};summary['fcc_comparisons'][method]={}
                for k in METRICS:
                    pairs=[(c['medians']['target'][k],c['medians'][method][k]) for c in group]
                    pairs=[p for p in pairs if None not in p]
                    summary['fcc_wins'][method][k]=sum(a<b for a,b in pairs);summary['fcc_comparisons'][method][k]=len(pairs)
            summaries.append(summary)
    preparation=read('runs/F4_global_matching_prepare_v1_20260906/metrics.raw.jsonl',True)
    pw=[r for r in preparation if r['kind']=='full_dictionary_assignment']
    summary=dict(inputs=inputs,new_model_forwards=new_forwards,new_consumer_wall_seconds=wall,summaries=summaries,
                 full_dictionary_pw_mcc={f"{r['source_seed']}-{r['target_seed']}":r['mean_absolute_cosine'] for r in pw},
                 evidence='Existing-document development; no target outcome fitting; dependencies and unequal target input support disclosed.')
    assert len(flat)==32*4*5*2
    write(OUT/'global_matching_comparison.json',summary);write(OUT/'global_matching_cases.json',cases)
    csv_write(OUT/'GLOBAL_MATCHING_ROWS.csv',flat)
    fig=figure(OUT,requests,flat,methods=['target']+NEW,labels=['FCC full','Global geometric','Global pair-calibrated'],
               subtitle='Existing-data development: all 32 requests / 24 matched cases; 264 new forwards for full-dictionary matching',
               include_rejected=True,footer_lines=[
                   'Global matching: all3072 features per seed; every source-group member mapped once. Common source basis and dose for all methods.',
                   'Pair calibration uses only fixed source-defined discovery rows; no target behavioral labels. Full FCC has greater target input support.',
                   'Small symbols: four dependent targets; large: case medians. Same log axes. Best dynamic atom and raw retained in the full CSV/table.',
                   'All24 matched requests measured,8 no-donor requests remain missing. Existing exposed documents; no independent-replicate CI.'],
               alt_text='Full FCC compared with geometric and paired-calibrated full-dictionary one-to-one group readouts on every matched source request; missing donors and adverse cases retained.')
    write(OUT/'global_matching_figure.json',fig)
    lines=['# Full-dictionary one-to-one group operations','',
           'Global3072-by3072 maximum-absolute decoder cosine assignments are computed once per unordered seed pair and inverted for reverse use. Each source group maps all its members. Geometric sign/norm scaling and independent per-pair linear calibration share the original source basis and common dose. Calibration uses only the exact original source-positive discovery rows/weights.','',
           '|Scope/group|Cases|FCC KL/NLL|Global geometric KL/NLL|Global pair-calibrated KL/NLL|FCC wins vs calibrated KL/NLL|',
           '|---|---:|---|---|---|---|']
    for s in summaries:
        pair=lambda m:' / '.join('NA' if s['methods'][m][k] is None else f"{s['methods'][m][k]:.6g}" for k in METRICS)
        wins=' / '.join(f"{s['fcc_wins'][NEW[1]][k]}/{s['fcc_comparisons'][NEW[1]][k]}" for k in METRICS)
        lines.append(f"|{s['scope']}/{s['group']}|{s['cases']}|{pair('target')}|{pair(NEW[0])}|{pair(NEW[1])}|{wins}|")
    lines+=['',f"Full-dictionary PW-MCC range over10 unordered pairs: {min(r['mean_absolute_cosine'] for r in pw):.6f}–{max(r['mean_absolute_cosine'] for r in pw):.6f}. These pair scores are not independent replicates or query-level function scores.",'',
            'Each statistic first takes the four dependent-target median within a request, then the median across requests. Geometric/pair-calibrated target support equals the source-group size; full FCC reads all target decoded contributions, so this is an alignment-family comparison, not equal target-support capacity. Both execute in the source basis, not native target deletion. The solver and scalar regressions are mature components, not claimed novel.','',
            'All32 requests retained;24matched measured and8no-donor missing. Old source-selected and rejected cases are separated; concentration and weak source effects remain confounded. Existing short-training configuration and exposed documents limit generalization. Raw and best dynamic atom, absolute errors and both scopes remain in GLOBAL_MATCHING_ROWS.csv/comparison JSON.','',
            'Actual source probability anchors, donors, positions and common dose replay exactly against old FCC rows. New candidate outcomes were evaluated only after assignment/calibration/coordinates were fixed. This is implementation/merge evidence, not independent scientific replication or new-document confirmation.','',
            'See global_matching_comparison.json for all input hashes, global_matching_cases.json for every case median, and global_matching_figure.json/figure_points.json for the full plot.']
    (OUT/'GLOBAL_MATCHING_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'global_matching_analysis_provenance.json',dict(scripts={str(p):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/analyze_f4_rejected_panel.py',ROOT/'scripts/build_f4_common_panel.py']},
          outputs={name:sha256((OUT/name).read_bytes()).hexdigest() for name in ['GLOBAL_MATCHING_FINDINGS.md','GLOBAL_MATCHING_ROWS.csv','global_matching_comparison.json','global_matching_cases.json','global_matching_figure.json','common_panel.png','figure_points.json']}))
    print(json.dumps({'forwards':new_forwards,'wall':wall,'primary':[s for s in summaries if s['scope']==SCOPES[0]],'figure':fig}))


if __name__=='__main__':main()
