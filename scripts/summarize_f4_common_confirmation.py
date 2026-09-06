"""Report every frozen fresh-document request, including rejection and missingness."""
import json
import math
from collections import Counter
from pathlib import Path
from build_f4_common_panel import ROOT, METRICS, SCOPES, med, csv_write, figure, write, sha256

OUT = ROOT / 'runs/F4_common_confirmation_apply_v1_20260906'
EXTRA = ['candidate_error_kl_mean', 'nll_delta_rmse']


def main():
    inputs=[]
    def read(path, lines=False):
        path=ROOT/path;inputs.append(dict(path=str(path),sha256=sha256(path)))
        text=path.read_text(encoding='utf-8')
        return [json.loads(l) for l in text.splitlines() if l.strip()] if lines else json.loads(text)
    cfg=read('configs/f4_common_confirmation_corpus_v1.json');frozen=cfg['frozen_scope'];methods=frozen['methods']
    rule=read(frozen['rule_config'])['rule'];requests=[];flat=[];cases=[];runs=[];recomputed=0;largest_error=0
    for panel in frozen['panels']:
        label=panel['label'];run=f'runs/F4_common_confirmation_{label}_v1_20260906'
        assert read(run+'/status.json')['status']=='PASS'
        assert read(run+'/contract_validation.json')['ok']
        config=read(run+'/config.resolved.json');summary=read(run+'/metrics.summary.json')
        assert config['methods']==methods and config['probability_endpoints']==frozen['probability_endpoints']
        assert config['maximum_source_hook_fraction']==frozen['maximum_source_hook_fraction']
        rows=read(run+'/metrics.raw.jsonl',True);runs.append(dict(run=run,**summary))
        index={(r['source_seed'],r['source_atom'],r['condition'],r['target_seed'],r['method']):r for r in rows}
        assert len(index)==len(rows)
        payload=read(config['case_replay']['path']);selection=read(run+'/selection.json')['queries']
        assert len(payload['choices'])==16 and not payload['prior_endpoint_exposure']
        for choice in payload['choices']:
            s,a,condition=choice['source_seed'],choice['source_atom'],choice['condition'];e=choice['entry'];scope=choice.get('source_scope') or {}
            chosen=bool(e and scope['selected']);failed=[]
            if e:
                if scope['natural_source_hook_fraction']<rule['minimum_natural_source_hook_fraction']:failed.append('weak')
                if scope['largest_atom_energy_share']>rule['maximum_largest_source_atom_energy_share']:failed.append('concentrated')
                assert chosen==(scope['supported'] and not failed)
            unit=next(u for u in selection if (u['source_seed'],u['source_atom'])==(s,a))
            status='SELECTED' if chosen else 'SOURCE_RULE_REJECTED' if e else choice['matching_status']
            request=dict(request_id=f'{label}:{s}:{a}:{condition}',panel=label,source_seed=s,source_atom=a,condition=condition,
                         targets=unit['targets'],selected=chosen,status=status,failed_source_rules=failed,entry=e,
                         natural_hook_fraction=scope.get('natural_source_hook_fraction'),largest_atom_energy_share=scope.get('largest_atom_energy_share'))
            requests.append(request)
            group=[r for key,r in index.items() if key[:3]==(s,a,condition)]
            assert len(group)==(20 if e else 0)
            for r in group:
                for key in ['sequence','donor_sequence','intervention_positions','donor_positions','document_ids','donor_document_ids']:assert r[key]==e[key]
                assert r['common_source_dose_scale']==group[0]['common_source_dose_scale']
                for scope_name in SCOPES:
                    p=r['probability_endpoints'][scope_name];source=group[0]['probability_endpoints'][scope_name]
                    for key in ['source_nll_deltas','source_to_baseline_kl','positions','observed_next_token_ids']:assert p[key]==source[key]
                    for metric in METRICS:
                        if p.get(metric) is None:continue
                        if metric=='normalized_kl_error':value=math.fsum(p['source_to_candidate_kl'])/math.fsum(p['source_to_baseline_kl'])
                        else:value=math.fsum((a-b)**2 for a,b in zip(p['candidate_nll_deltas'],p['source_nll_deltas']))/math.fsum(a*a for a in p['source_nll_deltas'])
                        largest_error=max(largest_error,abs(value-p[metric])/max(abs(value),1e-15));recomputed+=1
            for scope_name in SCOPES:
                values={m:{k:[] for k in METRICS+EXTRA} for m in methods}
                for method in methods:
                    for t in unit['targets']:
                        r=index.get((s,a,condition,t,method));p=r['probability_endpoints'][scope_name] if r else {}
                        row=dict(request_id=request['request_id'],source_seed=s,source_atom=a,condition=condition,target_seed=t,
                                 scope=scope_name,method=method,source_rule_status=status,status=p.get('status','NO_DONOR'),
                                 source_natural_hook_fraction=request['natural_hook_fraction'],largest_atom_energy_share=request['largest_atom_energy_share'],
                                 source_kl_mean=p.get('source_kl_mean'),source_nll_delta_rms=p.get('source_nll_delta_rms'),artifact=run+'/metrics.raw.jsonl' if r else None)
                        for metric in METRICS+EXTRA:row[metric]=p.get(metric);values[method][metric].append(p.get(metric))
                        flat.append(row)
                source=group[0]['probability_endpoints'][scope_name] if group else {}
                cases.append(dict(request_id=request['request_id'],scope=scope_name,source_rule_status=status,
                                  source_kl_mean=source.get('source_kl_mean'),source_nll_delta_rms=source.get('source_nll_delta_rms'),
                                  medians={m:{k:med(v) for k,v in vs.items()} for m,vs in values.items()}))
    assert len(requests)==32 and len(flat)==1280 and largest_error<1e-12
    summaries=[]
    for scope in SCOPES:
        for label,states in [('selected',{'SELECTED'}),('rejected',{'SOURCE_RULE_REJECTED'}),('all_matched',{'SELECTED','SOURCE_RULE_REJECTED'})]:
            group=[c for c in cases if c['scope']==scope and c['source_rule_status'] in states]
            reqs=[r for r in requests if r['status'] in states]
            documents={d for r in reqs for key in ['document_ids','donor_document_ids'] for d in r['entry'][key]}
            summary=dict(scope=scope,group=label,cases=len(group),methods={},fcc_wins={},valid_comparisons={},
                         query_count=len({(r['source_seed'],r['source_atom']) for r in reqs}),source_seeds=sorted({r['source_seed'] for r in reqs}),recipient_donor_document_count=len(documents),
                         source_kl_median=med([c['source_kl_mean'] for c in group]),source_nll_rms_median=med([c['source_nll_delta_rms'] for c in group]))
            for method in methods:summary['methods'][method]={k:med([c['medians'][method][k] for c in group]) for k in METRICS+EXTRA}
            for comparator in methods[1:]:
                summary['fcc_wins'][comparator]={};summary['valid_comparisons'][comparator]={}
                for metric in METRICS:
                    pairs=[(c['medians']['target'][metric],c['medians'][comparator][metric]) for c in group]
                    pairs=[p for p in pairs if None not in p]
                    summary['fcc_wins'][comparator][metric]=sum(a<b for a,b in pairs);summary['valid_comparisons'][comparator][metric]=len(pairs)
            summaries.append(summary)
    result=dict(coverage=dict(Counter(r['status'] for r in requests)),summaries=summaries,inputs=inputs,
                consumer_forwards=sum(r['model_forwards'] for r in runs),consumer_wall_seconds=sum(r['wall_seconds'] for r in runs),
                ratios_recomputed=recomputed,maximum_ratio_relative_difference=largest_error,
                evidence='Frozen methods/source rules in102fresh-document sampling pool; actual evaluated document/query counts separate; all32requests; dependent query/seed/document cases. Unequal target support disclosed, no new fitting.')
    write(OUT/'common_confirmation_summary.json',result);write(OUT/'common_confirmation_cases.json',cases);write(OUT/'common_confirmation_requests.json',requests)
    csv_write(OUT/'COMMON_CONFIRMATION_ROWS.csv',flat)
    fig=figure(OUT,requests,flat,methods=['target']+methods[-2:],labels=['FCC full','Global geometric','Global pair-calibrated'],include_rejected=True,
        subtitle=f"Frozen confirmation:102fresh documents / all32requests / {sum(r['entry'] is not None for r in requests)}matched cases; five methods retained",
        footer_lines=['Common source groups, basis, dose and fresh cases. Full FCC has greater target input support than32-member matching.',
                      'Small symbols: four dependent targets; large: case medians. Same log axes. Best dynamic atom/raw in complete tables.',
                      'Source rule fixed before sampling: weak=natural hook fraction<0.1; concentrated=largest atom energy share>0.5.',
                      'Missing donors and source rejections retained. Parameters fitted on old independent discovery only; no new target tuning.'],
        alt_text='Frozen fresh-document confirmation of FCC and full-dictionary group matching; all source-selected, rejected and missing cases shown.')
    write(OUT/'common_confirmation_figure.json',fig)
    lines=['# Frozen common-document functional confirmation','',
           f"102new documents in sampling pool,32requests, coverage {json.dumps(result['coverage'])}; {result['consumer_forwards']}consumer forwards and128separate asset forwards. Actual evaluated recipient/donor-document counts are separate in JSON;102is not the number of independent effect measurements. Original16queries/five source seeds, all five frozen methods and both scopes retained. No new fitting or target-dependent choice.",'',
           '|Scope/group|Cases|FCC KL/NLL|Raw KL/NLL|Dynamic atom KL/NLL|Geometric KL/NLL|Pair-calibrated KL/NLL|FCC wins vs paircal KL/NLL|',
           '|---|---:|---|---|---|---|---|---|']
    for s in summaries:
        def pair(m):return ' / '.join('NA' if s['methods'][m][k] is None else f"{s['methods'][m][k]:.6g}" for k in METRICS)
        wins=' / '.join(f"{s['fcc_wins'][methods[-1]][k]}/{s['valid_comparisons'][methods[-1]][k]}" for k in METRICS)
        lines.append('|'+f"{s['scope']}/{s['group']}|{s['cases']}|"+'|'.join(pair(m) for m in methods)+f'|{wins}|')
    lines+=['','Each statistic first takes four dependent-target medians, then request medians. Absolute errors/source denominators in CSV/JSON; source weakness and concentration are not independent mechanisms. Shared query/seed/document/reversed donor pairs are not independent observations. No significance inferred from win counts.','',
            'Full FCC reads all target contributions, matching reads32matched members. This confirms a frozen alignment-family operation comparison, not equal-capacity superiority, unique human concepts, native deletion, universal many-to-many necessity or training-mechanism identification. Source-selected, all matched and missing scopes must remain separate.','',
            f"Source/case/donor/dose anchors agree across every method. Independently recomputed{recomputed}ratios from stored raw vectors, max relative difference{largest_error:.6g}; this checks implementation arithmetic, not an independent scientific replicate."]
    (OUT/'COMMON_CONFIRMATION_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    names=['common_confirmation_summary.json','common_confirmation_cases.json','common_confirmation_requests.json','COMMON_CONFIRMATION_ROWS.csv','COMMON_CONFIRMATION_FINDINGS.md','common_confirmation_figure.json','common_panel.png','figure_points.json']
    write(OUT/'common_confirmation_analysis_provenance.json',dict(scripts={str(p):sha256(p) for p in [Path(__file__),ROOT/'scripts/build_f4_common_panel.py']},outputs={n:sha256(OUT/n) for n in names}))
    print(json.dumps({k:result[k] for k in ['coverage','summaries','consumer_forwards','consumer_wall_seconds','ratios_recomputed','maximum_ratio_relative_difference']}))


if __name__=='__main__':main()
