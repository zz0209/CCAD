"""Fixed-source recipient/complexity pilot; retain every frozen eligible case."""
import csv
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from summarize_f4_probability import ROOT, key, median, fmt, jsonl, write, sha256

METRICS=('normalized_kl_error','normalized_nll_delta_squared_error')
METHODS=('target','raw','readout_top16','short_single_atom','long128_target',
    'long128_top16','long128_single_atom','long32_target','long32_top16','long32_single_atom')


def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group();modes.add_argument('--confirmation-apply',type=Path);modes.add_argument('--five-seed-fit',type=Path);args=parser.parse_args()
    five=args.five_seed_fit is not None
    fresh=args.confirmation_apply is not None or five
    output=(args.five_seed_fit if five else args.confirmation_apply) if fresh else ROOT/'runs/F4_long_recipient_fit_v1_20260905'
    output=output if output.is_absolute() else ROOT/output
    if (output/'recipient_comparison.json').exists():raise ValueError('Already summarized')
    table=[];cases=[];aggregates=[];comparisons=[];checks={};inputs=[];forwards=0;wall=0
    for panel in ('original','expanded'):
        run=ROOT/'runs'/(f'F4_long_recipient_confirmation_{panel}_v1_20260905' if fresh else f'F4_long_recipient_{panel}_dev_v1_20260905')
        old=ROOT/'runs'/f'F4_probability_confirmation_{panel}_v1_20260905'
        summary=json.loads((run/'metrics.summary.json').read_text())
        assert summary['status']=='PASS'
        rows=jsonl(run/'metrics.raw.jsonl')
        assert len(rows)==(20 if panel=='original' else (60 if fresh else 80))
        if five:
            extension=ROOT/'runs'/f'F4_long_k128_newseeds_{panel}_v1_20260905'
            es=json.loads((extension/'metrics.summary.json').read_text());assert es['status']=='PASS'
            extra=jsonl(extension/'metrics.raw.jsonl');assert len(extra)==(14 if panel=='original' else 70)
            ec=json.loads((extension/'config.resolved.json').read_text());pc=json.loads((run/'config.resolved.json').read_text())
            for field in ('source_query_subset','probability_endpoints','source_scope','factors_sha256','ranks','maximum_source_hook_fraction'):
                assert ec[field]==pc[field],field
            assert ec['target_seed_subset']==[3,4,5] and ec['methods']==[m for m in METHODS if not m.startswith('long32')]
            prior={(r['source_seed'],r['source_atom'],r['condition'],r['sequence']):r for r in rows if r['method']=='target'}
            for r in extra:
                anchor=prior[r['source_seed'],r['source_atom'],r['condition'],r['sequence']]
                for field in ('sequence','donor_sequence','document_ids','donor_document_ids','intervention_positions','donor_positions','common_source_dose_scale','source_natural_hook_energy','source_hook_fraction'):assert r[field]==anchor[field],field
                for scope,p in r['probability_endpoints'].items():
                    for field in ('positions','observed_next_token_ids','source_to_baseline_kl','source_nll_deltas'):assert p[field]==anchor['probability_endpoints'][scope][field],field
            rows.extend(extra);assert len({key(r) for r in rows})==len(rows)
            forwards+=es['model_forwards'];wall+=es['wall_seconds']
            inputs.extend(dict(path=str(p),sha256=sha256(p)) for p in (extension/'metrics.raw.jsonl',extension/'config.resolved.json'))
        oldrows={key(r):r for r in (rows if fresh else jsonl(old/'metrics.raw.jsonl'))}
        if fresh:
            frozen_path=ROOT/'configs/f4_long_recipient_confirmation_corpus_v1.json'
            frozen=json.loads(frozen_path.read_text());rc=json.loads((run/'config.resolved.json').read_text())
            assert rc['frozen_corpus_config_sha256']==sha256(frozen_path)
            scope=frozen['frozen_scope']
            assert rc['methods']==scope['methods'] and rc['probability_endpoints']==scope['probability_endpoints']
            assert rc['target_seed_subset']==scope['target_seed_subset'] and rc['ranks']==scope['ranks']
            assert rc['maximum_source_hook_fraction']==scope['maximum_source_hook_fraction']
            assert rc['factors_sha256']==scope['factors_sha256']
            prep=ROOT/'runs'/f'F4_long_recipient_source_{panel}_v1_20260905'
            matched=json.loads((prep/'matching.json').read_text())
            assert not matched['prior_endpoint_exposure']
            assert json.loads((run/'all_source_candidates.json').read_text())['queries']==json.loads((prep/'selection.json').read_text())['queries']
            expected=next(p['fixed_queries'] for p in scope['panels'] if p['label']==panel)
            assert rc['source_query_subset']==expected
            for choice in matched['choices']:
                e=choice['entry'];active=bool(e and choice['source_scope']['selected'])
                subset=[r for r in rows if (r['source_seed'],r['source_atom'],r['condition'])==(choice['source_seed'],choice['source_atom'],choice['condition'])]
                expected_rows=10*len([t for t in scope['target_seed_subset'] if t!=choice['source_seed']])
                if five:expected_rows+=7*len([t for t in [3,4,5] if t!=choice['source_seed']])
                assert len(subset)==(expected_rows if active else 0)
                for r in subset:
                    for field in ('sequence','donor_sequence','document_ids','donor_document_ids','intervention_positions','donor_positions'):assert r[field]==e[field]
            inputs.extend(dict(path=str(p),sha256=sha256(p)) for p in (frozen_path,prep/'matching.json'))
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
                for num,den,metric in ((kc,kb,METRICS[0]),(ne,se,METRICS[1])):
                    if den<=1e-12*len(p['positions']):assert p[metric] is None
                    else:assert math.isclose(num/den,p[metric],rel_tol=1e-10,abs_tol=1e-12)
                recomputed+=2
                row=dict(panel=panel,source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],
                    target_seed=r['target_seed'],method=r['method'],scope=scope,
                    source_hook_fraction=r['source_hook_fraction'],candidate_hook_fraction=r['candidate_hook_fraction'],
                    **{k:p[k] for k in METRICS})
                table.append(row);groups[scope,r['source_seed'],r['source_atom'],r['condition'],r['method']].append(row)
            sourcechecks+=1
            if not fresh and r['method'] in ('target','raw','readout_top16'):
                prior=oldrows[key(r)]
                assert r['probability_endpoints']==prior['probability_endpoints']
                assert r['endpoints']==prior['endpoints'] and r['hook']==prior['hook']
                replay+=1
        for (scope,s,a,condition,method),values in groups.items():
            cases.append(dict(panel=panel,scope=scope,source_seed=s,source_atom=a,condition=condition,method=method,
                targets=len(values),**{k:median([r[k] for r in values]) for k in METRICS}))
        checks[panel]=dict(source_anchor_rows_exact=sourcechecks,source_anchor_reference='within_same_new_case' if fresh else 'old_run',old_method_rows_exact=replay,probability_ratios_recomputed=recomputed)
        forwards+=summary['model_forwards'];wall+=summary['wall_seconds']
        for path in ((run/'metrics.raw.jsonl',run/'config.resolved.json') if fresh else (run/'metrics.raw.jsonl',run/'config.resolved.json',old/'metrics.raw.jsonl')):
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
                    diff=[r[metric]-lookup[r['panel'],r['source_seed'],r['source_atom'],r['condition'],baseline][metric] for r in values if r[metric] is not None and lookup[r['panel'],r['source_seed'],r['source_atom'],r['condition'],baseline][metric] is not None]
                    comparisons.append(dict(panel=panel,scope=scope,candidate=candidate,baseline=baseline,metric=metric,
                        cases=len(diff),candidate_lower=sum(d<0 for d in diff),median_paired_difference=median(diff)))
    result=dict(aggregates=aggregates,cases=cases,comparisons=comparisons,checks=checks,inputs=inputs,
        model_forwards=forwards,wall_seconds=wall,fit_seconds=json.loads((output/'metrics.summary.json').read_text())['wall_seconds'],
        generator_sha256=sha256(Path(__file__)),
        scope='Exposed cross-configuration development. Fixed source and dose. 6 eligible cases / 4 source queries / 10 target-case pairs; 5 of original11 excluded before target encoding for long-training document overlap. Median across available dependent target seeds within each case, then across cases. Combined6 is supplementary descriptive; no independence or confirmation claim.')
    if fresh:
        result['scope']='Fresh-document confirmation of frozen cross-configuration compactness: 109 documents sampled after ID/text-hash exclusions; 4 fixed source queries and all8sign requests,6class-matched,5source-selected cases/3active queries/8target-case pairs. No refit or support changes. Median over dependent targets within case, then cases; combined5 is descriptive, not independent repetitions. Two long target seeds, not five-seed main suite. Original paired audit closed.'
        result['coverage']=[dict(panel=panel,source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],matched=r['entry'] is not None,selected=bool(r['entry'] and r['source_scope']['selected']),matching_status=r.get('matching_status')) for panel in ('original','expanded') for r in json.loads((output/f'{panel}_case_selection.json').read_text())['choices']]
    if five:
        result['scope']='Five long-k128 recipient seeds, fixed short-source interface. Same5selectedcases/3activequeries/20dependenttarget-casepairs,all8requests retained. Originaltargets1/2 were fresh-document confirmation; addedtargets3/4/5 are seed replication on now-exposed109documents. Common7methods cover4targets/case (exclude source index); long32secondary remains onlytargets1/2. Not a long-to-long five-seed FCC or renewed unseen-document claim.'
        newcases=[]
        for scope,s,a,condition,method in sorted({(r['scope'],r['source_seed'],r['source_atom'],r['condition'],r['method']) for r in table if r['target_seed']>=3}):
            rr=[r for r in table if r['target_seed']>=3 and (r['scope'],r['source_seed'],r['source_atom'],r['condition'],r['method'])==(scope,s,a,condition,method)]
            newcases.append(dict(scope=scope,source_seed=s,source_atom=a,condition=condition,method=method,targets=len(rr),**{k:median([r[k] for r in rr]) for k in METRICS}))
        result['new_seed_cases']=newcases
        result['new_seed_aggregates']=[dict(scope=scope,method=method,cases=len(rr),**{k:median([r[k] for r in rr]) for k in METRICS}) for scope in ('intervention_positions','same_document_downstream') for method in METHODS if not method.startswith('long32') for rr in [[r for r in newcases if r['scope']==scope and r['method']==method]]]
        result['new_seed_top16_comparisons']=[]
        for metric in METRICS:
            rr=[r for r in newcases if r['scope']=='intervention_positions' and r['method']=='long128_top16'];lookup={(r['source_seed'],r['source_atom'],r['condition']):r for r in newcases if r['scope']=='intervention_positions' and r['method']=='readout_top16'}
            result['new_seed_top16_comparisons'].append(dict(metric=metric,cases=len(rr),candidate_lower=sum(r[metric]<lookup[r['source_seed'],r['source_atom'],r['condition']][metric] for r in rr)))
    for name,values in [('RECIPIENT_ROWS.csv',table),('RECIPIENT_CASES.csv',cases)]:
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write(output/'recipient_comparison.json',result)
    lines=['# 长训练接收端：同一source作用与组成预算','',result['scope'],'',
        '误差越低越好；0为精确保留source作用，1为不干预。source dose相同，但candidate实际能量不强制相同，完整剂量见逐行CSV。',
        ('short是旧k128；long128已具五个同配置4194304token SAE，long32仍仅两seed。这里是固定short source到long recipients，尚非long-to-long五seed对应；短长训练流不同，不是嵌套学习曲线。' if five else 'short是旧k128；long128/long32均使用既有4194304token权重。训练流不同，不是嵌套学习曲线；两seed不能冒充五seed长配置主套件。'),
        'full经decoded hook拟合；top16是固定conditional-energy排序截断；single是相同discovery上全3072列的最佳conditional-variation单atom。不是最优稀疏拟合。',
        '', '|面板|端点范围|方法|case|KL误差|NLL变化平方误差|','|---|---|---|---:|---:|---:|']
    for r in aggregates:lines.append('|'+ '|'.join([r['panel'],r['scope'],r['method'],str(r['cases'])]+[fmt(r[k]) for k in METRICS])+'|')
    if five:
        lines+=['','## 新增target3/4/5单列（不借旧target1/2优势）','', '|范围|方法|case|KL|NLL|','|---|---|---:|---:|---:|']
        for r in result['new_seed_aggregates']:lines.append('|'+ '|'.join([r['scope'],r['method'],str(r['cases'])]+[fmt(r[k]) for k in METRICS])+'|')
    lines+=['','## 逐案例主端点','', '|面板|source|条件|target数|方法|KL|NLL|','|---|---|---|---:|---|---:|---:|']
    for r in cases:
        if r['scope']=='intervention_positions':lines.append('|'+ '|'.join([r['panel'],f"{r['source_seed']}:{r['source_atom']}",r['condition'],str(r['targets']),r['method']]+[fmt(r[k]) for k in METRICS])+'|')
    lines+=['','Checks: '+json.dumps(checks),'',f'Actual LM forwards={forwards}, causal wall_seconds={wall}, preparation wall_seconds={result["fit_seconds"]}. No new training, downloads, or audit.','',
        ('八请求覆盖详见recipient_comparison.json：三请求未评估，不能从分母中隐去或当作成功。旧开发与本次确认分别报告。' if fresh else 'case_exclusions.json保留原选择及5项排除原因。旧短SAE的完整120文档结果不因长训练重叠而作废；本次不能把该完整语料称为长SAE未见。'),
        '数学消费者见DERIVATION_PACKAGE.md §13：操作类、适用分布、donor差分、joint Jacobian及component-error Gram。代数边界不替代实际端点。']
    (output/'RECIPIENT_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(aggregates=aggregates,comparisons=comparisons,checks=checks,model_forwards=forwards,wall_seconds=wall)))


if __name__=='__main__':main()
