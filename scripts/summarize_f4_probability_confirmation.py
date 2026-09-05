"""Frozen-workflow confirmation: all requested cases, dependent-target summaries."""
import csv
import json
from collections import Counter
from pathlib import Path
from summarize_f4_probability import ROOT,METHODS,METRICS,key,median,fmt,jsonl,write,sha256


def main():
    output=ROOT/'runs/F4_probability_confirmation_expanded_v1_20260905'
    if (output/'probability_summary.json').exists():raise ValueError('Confirmation summary already exists')
    frozen=ROOT/'configs/f4_probability_confirmation_corpus_v1.json'
    freeze=json.loads(frozen.read_text());inputs=[];cases=[];table=[];coverage={};checks={};forwards=0;seconds=0
    for panel in ('original','expanded'):
        run=ROOT/'runs'/f'F4_probability_confirmation_{panel}_v1_20260905'
        prep=ROOT/'runs'/f'F4_probability_confirmation_source_{panel}_v1_20260905'
        cfg=json.loads((run/'config.resolved.json').read_text());m=json.loads((prep/'matching.json').read_text())
        assert not m['prior_endpoint_exposure'];assert cfg['frozen_corpus_config_sha256']==sha256(frozen)
        assert cfg['methods']==freeze['frozen_scope']['methods'] and cfg['probability_endpoints']==freeze['frozen_scope']['probability_endpoints']
        assert cfg['factors_sha256']==freeze['frozen_scope']['factors_sha256'] and cfg['ranks']==[1] and cfg['maximum_source_hook_fraction']==.1
        source=json.loads((run/'source_scope_selection.json').read_text())['rows']
        assert all(r['selected'] and r['supported'] for r in source)
        before=json.loads((prep/'source_scope_selection.json').read_text())['rows']
        all_selections=json.loads((run/'all_source_candidates.json').read_text())['queries']
        assert all_selections==json.loads((prep/'selection.json').read_text())['queries']
        expected_queries=next(p['fixed_queries'] for p in freeze['frozen_scope']['panels'] if p['label']==panel)
        assert [[u['source_seed'],u['source_atom']] for u in all_selections]==expected_queries
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        raw=jsonl(run/'metrics.raw.jsonl');details={key(r):r for r in jsonl(run/'case_details.jsonl')}
        summary=json.loads((run/'metrics.summary.json').read_text());forwards+=summary['model_forwards'];seconds+=summary['wall_seconds']
        used=set()
        for choice in m['choices']:
            s,a,c=choice['source_seed'],choice['source_atom'],choice['condition'];e=choice['entry']
            selected=bool(e and choice['source_scope']['selected']);group=[r for r in raw if (r['source_seed'],r['source_atom'],r['condition'])==(s,a,c)]
            assert len(group)==(16 if selected else 0)
            record=dict(panel=panel,source_seed=s,source_atom=a,condition=c,entry=e,selected=selected,
                matching_status=choice['matching_status'],source_scope=choice['source_scope'],scopes={},old_endpoints={})
            for r in group:
                used.add(key(r));assert r['endpoints']==details[key(r)]['endpoints']
                assert all(r[k]==e[k] for k in ('sequence','donor_sequence','document_ids','donor_document_ids','intervention_positions','donor_positions'))
                for p in details[key(r)]['positions']:assert p['source_reconstruction_residual']<1e-8 and (p['target_reconstruction_residual'] is None or p['target_reconstruction_residual']<1e-8)
                for scope,p in r['probability_endpoints'].items():
                    assert p['position_count']>0
                    table.append(dict(panel=panel,source_seed=s,source_atom=a,condition=c,target_seed=r['target_seed'],method=r['method'],scope=scope,position_count=p['position_count'],**{k:p[k] for k in METRICS}))
            if selected:
                for scope in ('intervention_positions','same_document_downstream'):
                    ref=group[0]['probability_endpoints'][scope]
                    assert all(r['probability_endpoints'][scope]['source_to_baseline_kl']==ref['source_to_baseline_kl'] and r['probability_endpoints'][scope]['source_nll_deltas']==ref['source_nll_deltas'] for r in group)
                    record['scopes'][scope]=dict(positions=ref['positions'],observed_next_token_ids=ref['observed_next_token_ids'],source_nll_deltas=ref['source_nll_deltas'],source_to_baseline_kl=ref['source_to_baseline_kl'],methods={})
                    for method in METHODS:
                        sub=[r['probability_endpoints'][scope] for r in group if r['method']==method]
                        record['scopes'][scope]['methods'][method]={k:median([r[k] for r in sub]) for k in METRICS}
                for method in METHODS:record['old_endpoints'][method]={ep:median([r['endpoints'][ep]['normalized_error'] for r in group if r['method']==method]) for ep in ('centered_logits','next_state')}
            cases.append(record)
        assert len(used)==len(raw)==len(details)
        coverage[panel]=dict(requests=len(m['choices']),original_candidates=len(before),original_selected=sum(r['selected'] for r in before),matched_supported=sum(r['entry'] is not None for r in m['choices']),selected=len(source),matching_status=dict(Counter(r['matching_status'] for r in m['choices'])))
        checks[panel]=dict(frozen_queries_methods_endpoints=True,source_selection_before_after_exact=True,method_rows=len(raw),all_signed_details_checked=True)
        for p in (frozen,prep/'matching.json',prep/'selection.json',run/'config.resolved.json',run/'metrics.raw.jsonl',run/'case_details.jsonl'):
            inputs.append(dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size))
    aggregates=[]
    for panel in ('original','expanded'):
        pc=[r for r in cases if r['panel']==panel and r['selected']]
        for scope in ('intervention_positions','same_document_downstream'):
            for method in METHODS:
                values=[r['scopes'][scope]['methods'][method] for r in pc];obs=[r for r in table if r['panel']==panel and r['scope']==scope and r['method']==method]
                def ok(r):return all(r[k] is not None and r[k]<1 for k in ('normalized_kl_error','normalized_nll_delta_squared_error'))
                aggregates.append(dict(panel=panel,scope=scope,method=method,cases=len(pc),medians={k:median([r[k] for r in values]) for k in METRICS},both_below_zero=sum(ok(r) for r in values),target_both_below_zero=sum(ok(r) for r in obs),target_rows=len(obs),full_wins={k:sum(r['scopes'][scope]['methods']['target'][k]<r['scopes'][scope]['methods'][method][k] for r in pc if r['scopes'][scope]['methods']['target'][k] is not None and r['scopes'][scope]['methods'][method][k] is not None) for k in ('normalized_kl_error','normalized_nll_delta_squared_error')}))
    result=dict(cases=cases,aggregates=aggregates,coverage=coverage,checks=checks,model_forwards=forwards,wall_seconds=seconds,inputs=inputs,generator_script_sha256=sha256(Path(__file__)),scope='Fresh-document confirmation of frozen whole source-only selection/class matching/probability workflow; shared seeds/documents/query conditions dependent; not semantic uniqueness. Original32requests retained, only source-selected pairs evaluated.')
    write(output/'probability_summary.json',result)
    with (output/'probability_rows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# 新文档概率确认：完整请求表','',result['scope'],'','|面板|范围|方法|有效case|KL误差|NLL变化平方误差|case双<1|target双<1|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:lines.append('|'+ '|'.join([r['panel'],r['scope'],r['method'],str(r['cases']),fmt(r['medians']['normalized_kl_error']),fmt(r['medians']['normalized_nll_delta_squared_error']),str(r['both_below_zero']),f"{r['target_both_below_zero']}/{r['target_rows']}"])+'|')
    lines+=['','|面板/source:atom/条件|状态|source KL均值|source NLL RMS|full KL/NLL|global KL/NLL|raw KL/NLL|top16 KL/NLL|','|---|---|---:|---:|---|---|---|---|']
    for r in cases:
        label=f"{r['panel']} {r['source_seed']}:{r['source_atom']} {r['condition']}"
        if not r['selected']:lines.append(f"|{label}|{r['matching_status'] if r['entry'] is None else 'SOURCE_NOT_SELECTED'}|NA|NA|NA|NA|NA|NA|");continue
        m=r['scopes']['intervention_positions']['methods'];s=m['target']
        lines.append('|'+ '|'.join([label,'SELECTED',fmt(s['source_kl_mean']),fmt(s['source_nll_delta_rms'])]+[fmt(m[n]['normalized_kl_error'])+'/'+fmt(m[n]['normalized_nll_delta_squared_error']) for n in METHODS])+'|')
    lines+=['','0=精确保留source干预，1=不干预；比率是误差而非正确率。先四target中位数再case中位数，不把共享seed方向当独立重复。所有逐位置source/candidate概率与NLL、原logits/state见raw；352范围方法行见CSV。']
    (output/'PROBABILITY_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(coverage=coverage,aggregates=aggregates,checks=checks,model_forwards=forwards,wall_seconds=seconds)))


if __name__=='__main__':main()
