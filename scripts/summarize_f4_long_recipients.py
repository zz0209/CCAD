"""Fixed-source recipient/complexity pilot; retain every frozen eligible case."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from summarize_f4_probability import ROOT, key, median, fmt, jsonl, write, sha256

METRICS=('normalized_kl_error','normalized_nll_delta_squared_error')
METHODS=('target','raw','readout_top16','short_single_atom','long128_target',
    'long128_top16','long128_single_atom','long32_target','long32_top16','long32_single_atom')


def main():
    output=ROOT/'runs/F4_long_recipient_fit_v1_20260905'
    if (output/'recipient_comparison.json').exists():raise ValueError('Already summarized')
    table=[];cases=[];aggregates=[];comparisons=[];checks={};inputs=[];forwards=0;wall=0
    for panel in ('original','expanded'):
        run=ROOT/'runs'/f'F4_long_recipient_{panel}_dev_v1_20260905'
        old=ROOT/'runs'/f'F4_probability_confirmation_{panel}_v1_20260905'
        summary=json.loads((run/'metrics.summary.json').read_text())
        assert summary['status']=='PASS'
        rows=jsonl(run/'metrics.raw.jsonl');oldrows={key(r):r for r in jsonl(old/'metrics.raw.jsonl')}
        assert len(rows)==(20 if panel=='original' else 80)
        sourcechecks=0;replay=0;recomputed=0;groups=defaultdict(list)
        for r in rows:
            anchor=dict(r,method='target');parent=oldrows[key(anchor)]
            for field in ('sequence','donor_sequence','document_ids','donor_document_ids','intervention_positions',
                          'donor_positions','common_source_dose_scale','source_natural_hook_energy','source_hook_fraction'):
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
                row=dict(panel=panel,source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],
                    target_seed=r['target_seed'],method=r['method'],scope=scope,
                    source_hook_fraction=r['source_hook_fraction'],candidate_hook_fraction=r['candidate_hook_fraction'],
                    **{k:p[k] for k in METRICS})
                table.append(row);groups[scope,r['source_seed'],r['source_atom'],r['condition'],r['method']].append(row)
            sourcechecks+=1
            if r['method'] in ('target','raw','readout_top16'):
                prior=oldrows[key(r)]
                assert r['probability_endpoints']==prior['probability_endpoints']
                assert r['endpoints']==prior['endpoints'] and r['hook']==prior['hook']
                replay+=1
        for (scope,s,a,condition,method),values in groups.items():
            cases.append(dict(panel=panel,scope=scope,source_seed=s,source_atom=a,condition=condition,method=method,
                targets=len(values),**{k:median([r[k] for r in values]) for k in METRICS}))
        checks[panel]=dict(source_anchor_rows_exact=sourcechecks,old_method_rows_exact=replay,probability_ratios_recomputed=recomputed)
        forwards+=summary['model_forwards'];wall+=summary['wall_seconds']
        for path in (run/'metrics.raw.jsonl',run/'config.resolved.json',old/'metrics.raw.jsonl'):
            inputs.append(dict(path=str(path),sha256=sha256(path)))
    for panel in ('original','expanded','combined_descriptive'):
        pc=[r for r in cases if panel=='combined_descriptive' or r['panel']==panel]
        for scope in ('intervention_positions','same_document_downstream'):
            for method in METHODS:
                values=[r for r in pc if r['scope']==scope and r['method']==method]
                aggregates.append(dict(panel=panel,scope=scope,method=method,cases=len(values),
                    **{k:median([r[k] for r in values]) for k in METRICS}))
            lookup={(r['panel'],r['source_seed'],r['source_atom'],r['condition'],r['method']):r for r in pc if r['scope']==scope}
            for candidate,baseline in [('long128_target','target'),('long32_target','target'),('long128_top16','readout_top16'),
                    ('long32_top16','readout_top16'),('long128_single_atom','short_single_atom'),('long32_single_atom','short_single_atom'),
                    ('target','short_single_atom'),('long128_target','long128_single_atom'),('long32_target','long32_single_atom'),
                    ('long128_top16','long128_target'),('long32_top16','long32_target')]:
                values=[r for r in pc if r['scope']==scope and r['method']==candidate]
                for metric in METRICS:
                    diff=[r[metric]-lookup[r['panel'],r['source_seed'],r['source_atom'],r['condition'],baseline][metric] for r in values]
                    comparisons.append(dict(panel=panel,scope=scope,candidate=candidate,baseline=baseline,metric=metric,
                        cases=len(diff),candidate_lower=sum(d<0 for d in diff),median_paired_difference=median(diff)))
    result=dict(aggregates=aggregates,cases=cases,comparisons=comparisons,checks=checks,inputs=inputs,
        model_forwards=forwards,wall_seconds=wall,fit_seconds=json.loads((output/'metrics.summary.json').read_text())['wall_seconds'],
        generator_sha256=sha256(Path(__file__)),
        scope='Exposed cross-configuration development. Fixed source and dose. 6 eligible cases / 4 source queries / 10 target-case pairs; 5 of original11 excluded before target encoding for long-training document overlap. Median across available dependent target seeds within each case, then across cases. Combined6 is supplementary descriptive; no independence or confirmation claim.')
    for name,values in [('RECIPIENT_ROWS.csv',table),('RECIPIENT_CASES.csv',cases)]:
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write(output/'recipient_comparison.json',result)
    lines=['# 长训练接收端：同一source作用与组成预算','',result['scope'],'',
        '误差越低越好；0为精确保留source作用，1为不干预。source dose相同，但candidate实际能量不强制相同，完整剂量见逐行CSV。',
        'short是旧k128；long128/long32均使用既有4194304token权重。训练流不同，不是嵌套学习曲线；两seed不能冒充五seed长配置主套件。',
        'full经decoded hook拟合；top16是固定conditional-energy排序截断；single是相同discovery上全3072列的最佳conditional-variation单atom。不是最优稀疏拟合。',
        '', '|面板|端点范围|方法|case|KL误差|NLL变化平方误差|','|---|---|---|---:|---:|---:|']
    for r in aggregates:lines.append('|'+ '|'.join([r['panel'],r['scope'],r['method'],str(r['cases'])]+[fmt(r[k]) for k in METRICS])+'|')
    lines+=['','## 逐案例主端点','', '|面板|source|条件|target数|方法|KL|NLL|','|---|---|---|---:|---|---:|---:|']
    for r in cases:
        if r['scope']=='intervention_positions':lines.append('|'+ '|'.join([r['panel'],f"{r['source_seed']}:{r['source_atom']}",r['condition'],str(r['targets']),r['method']]+[fmt(r[k]) for k in METRICS])+'|')
    lines+=['','Checks: '+json.dumps(checks),'',f'Actual LM forwards={forwards}, causal wall_seconds={wall}, preparation wall_seconds={result["fit_seconds"]}. No new training, downloads, or audit.','',
        'case_exclusions.json保留原选择及5项排除原因。旧短SAE的完整120文档结果不因长训练重叠而作废；本次不能把该完整语料称为长SAE未见。',
        '数学消费者见DERIVATION_PACKAGE.md §13：操作类、适用分布、donor差分、joint Jacobian及component-error Gram。代数边界不替代实际端点。']
    (output/'RECIPIENT_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(aggregates=aggregates,comparisons=comparisons,checks=checks,model_forwards=forwards,wall_seconds=wall)))


if __name__=='__main__':main()
