"""Compare every old/new full/raw case without new forwards or exclusions."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from summarize_f4_probability import ROOT, key, median, fmt, jsonl, write, sha256

METRICS=('normalized_kl_error','normalized_nll_delta_squared_error')


def main():
    output=ROOT/'runs/F4_full_operation_expanded_dev_v1_20260905'
    if (output/'operation_comparison.json').exists():raise ValueError('Already summarized')
    table=[];aggregates=[];checks={};inputs=[];forwards=0;wall=0
    for panel in ('original','expanded'):
        new=ROOT/'runs'/f'F4_full_operation_{panel}_dev_v1_20260905'
        old=ROOT/'runs'/f'F4_probability_confirmation_{panel}_v1_20260905'
        oldrows={key(r):r for r in jsonl(old/'metrics.raw.jsonl')}
        newrows=jsonl(new/'metrics.raw.jsonl');details={key(r):r for r in jsonl(new/'case_details.jsonl')}
        assert len(newrows)==(24 if panel=='original' else 64)
        sourcechecks=0;fallback=0;recomputed=0;groups=defaultdict(list)
        for r in newrows:
            parent=oldrows[key(r)];assert r['endpoints']==details[key(r)]['endpoints']
            for field in ('sequence','donor_sequence','document_ids','donor_document_ids','intervention_positions','donor_positions','common_source_dose_scale','source_natural_hook_energy','source_hook_fraction'):
                assert r[field]==parent[field],field
            for scope,p in r['probability_endpoints'].items():
                pp=parent['probability_endpoints'][scope]
                for field in ('positions','observed_next_token_ids','source_to_baseline_kl','source_nll_deltas'):
                    assert p[field]==pp[field],field
                kb=sum(p['source_to_baseline_kl']);kc=sum(p['source_to_candidate_kl'])
                se=sum(x*x for x in p['source_nll_deltas'])
                ne=sum((s-c)**2 for s,c in zip(p['source_nll_deltas'],p['candidate_nll_deltas']))
                assert math.isclose(kc/kb,p[METRICS[0]],rel_tol=1e-10,abs_tol=1e-12)
                assert math.isclose(ne/se,p[METRICS[1]],rel_tol=1e-10,abs_tol=1e-12)
                recomputed+=2
                for version,value in (('old',pp),('operation',p)):
                    row=dict(panel=panel,source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],target_seed=r['target_seed'],method=r['method'],version=version,scope=scope,**{k:value[k] for k in METRICS})
                    table.append(row);groups[version,r['method'],scope,r['source_seed'],r['source_atom'],r['condition']].append(row)
            sourcechecks+=1
            if (r['source_seed'],r['source_atom'])==(5,2194):
                assert r['probability_endpoints']==parent['probability_endpoints'] and r['endpoints']==parent['endpoints'];fallback+=1
        cases={g:{k:median([r[k] for r in rows]) for k in METRICS} for g,rows in groups.items()}
        for version in ('old','operation'):
            for method in ('target','raw'):
                for scope in ('intervention_positions','same_document_downstream'):
                    values=[v for g,v in cases.items() if g[:3]==(version,method,scope)]
                    aggregates.append(dict(panel=panel,version=version,method=method,scope=scope,cases=len(values),**{k:median([r[k] for r in values]) for k in METRICS}))
        checks[panel]=dict(source_anchor_rows_exact=sourcechecks,fallback_rows_exact=fallback,probability_ratios_recomputed=recomputed,raw_detail_rows_exact=len(newrows))
        summary=json.loads((new/'metrics.summary.json').read_text());forwards+=summary['model_forwards'];wall+=summary['wall_seconds']
        for path in (new/'metrics.raw.jsonl',new/'case_details.jsonl',new/'config.resolved.json',old/'metrics.raw.jsonl'):
            inputs.append(dict(path=str(path),sha256=sha256(path)))
    result=dict(aggregates=aggregates,checks=checks,inputs=inputs,model_forwards=forwards,wall_seconds=wall,fit_records=28,newly_fitted_maps=24,parent_fallback_maps=4,generator_sha256=sha256(Path(__file__)),scope='All 11 fixed cases; exposed development, not new confirmation. Median within four dependent targets, then across cases. Rank1 output is not correspondence cardinality. Fallback query retained for both methods.')
    write(output/'operation_comparison.json',result)
    with (output/'OPERATION_ROWS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# 完整映射操作匹配：全部新旧结果','',result['scope'],'','|面板|范围|版本|方法|case|KL误差|NLL变化平方误差|','|---|---|---|---|---:|---:|---:|']
    for r in aggregates:lines.append('|'+ '|'.join([r['panel'],r['scope'],r['version'],r['method'],str(r['cases'])]+[fmt(r[k]) for k in METRICS])+'|')
    lines+=['','误差越低越好；0为精确保留source作用，1为不干预。完整逐case/target主辅端点见OPERATION_ROWS.csv。旧global/atom/UOT没有因本次改拟合重新运行，原结果保留。', '', '拟合记录28不等于成功新拟合28：实际六query24新映射，一query四映射原样回退；两方法一起回退且该案例未剔除。源组包含32 atom，rank1不等于one-to-many；当前仍不能据此宣称独立语义概念。', '', 'Checks: '+json.dumps(checks), '', f'Actual causal forwards={forwards}, wall_seconds={wall}; CPU fit times separate.']
    (output/'OPERATION_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
