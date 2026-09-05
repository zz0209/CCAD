"""Describe all source-component cases and nonlinear NLL interactions."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from run_f4_source_reference_causal import ROOT,np,jsonl,write,sha256


def med(values):
    finite=[v for v in values if v is not None]
    return median(finite) if finite else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run if args.run.is_absolute() else ROOT/args.run
    if (run/'component_summary.json').exists():raise FileExistsError('Already summarized')
    summary=json.loads((run/'metrics.summary.json').read_text());assert summary['status']=='PASS'
    cfg=json.loads((run/'config.resolved.json').read_text());rows=jsonl(run/'metrics.raw.jsonl');lookup={(r['case_id'],r['method'],r['component']):r for r in rows}
    assert len(rows)==90 and len(lookup)==90 and len({r['case_id'] for r in rows})==5
    table=[];compositions=[];cases=[];ratios=0
    for r in rows:
        for scope,p in r['probability_endpoints'].items():
            ds=np.array(p['source_nll_deltas']);dc=np.array(p['candidate_nll_deltas']);kl=sum(p['source_to_baseline_kl']);kc=sum(p['source_to_candidate_kl']);se=float(ds@ds)
            expected=kc/kl if kl>1e-12*len(ds) else None
            assert (expected is None and p['normalized_kl_error'] is None) or np.isclose(expected,p['normalized_kl_error'],rtol=1e-10,atol=1e-12)
            expected=float(np.sum((ds-dc)**2))/se if se>1e-12*len(ds) else None
            assert (expected is None and p['normalized_nll_delta_squared_error'] is None) or np.isclose(expected,p['normalized_nll_delta_squared_error'],rtol=1e-10,atol=1e-12)
            ratios+=2
            source=lookup[r['case_id'],'source',r['component']]['probability_endpoints'][scope]
            assert p['source_nll_deltas']==source['source_nll_deltas'] and p['source_to_baseline_kl']==source['source_to_baseline_kl']
            table.append(dict(case_id=r['case_id'],source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],method=r['method'],component=r['component'],scope=scope,
                normalized_kl_error=p['normalized_kl_error'],normalized_nll_delta_squared_error=p['normalized_nll_delta_squared_error'],source_nll_delta_rms=p['source_nll_delta_rms'],
                candidate_nll_delta_mean=p['candidate_nll_delta_mean'],candidate_error_kl_mean=p['candidate_error_kl_mean'],nll_delta_rmse=p['nll_delta_rmse'],source_kl_mean=p['source_kl_mean'],
                source_hook_fraction=r['source_hook_fraction'],candidate_hook_fraction=r['candidate_hook_fraction'],common_family_scale=r['common_family_scale'],old_full_scale=r['old_full_scale']))
    metrics=('normalized_kl_error','normalized_nll_delta_squared_error')
    for case_id in range(5):
        allcase=[r for r in table if r['case_id']==case_id];first=allcase[0]
        for scope in ('intervention_positions','same_document_downstream'):
            for component in ('full','A','B'):
                for method in ('long','raw','source'):
                    rr=[r for r in allcase if r['scope']==scope and r['component']==component and (r['method'].startswith('long_') if method=='long' else r['method']==method)]
                    assert len(rr)==(4 if method=='long' else 1)
                    cases.append(dict(case_id=case_id,source_seed=first['source_seed'],source_atom=first['source_atom'],condition=first['condition'],scope=scope,component=component,method=method,targets=len(rr),
                        **{k:med([r[k] for r in rr]) for k in metrics},source_nll_delta_rms=rr[0]['source_nll_delta_rms'],source_kl_mean=rr[0]['source_kl_mean']))
            methods=sorted({r['method'] for r in allcase})
            source_parts={c:np.array(lookup[case_id,'source',c]['probability_endpoints'][scope]['source_nll_deltas']) for c in ('full','A','B')}
            source_interaction=source_parts['full']-source_parts['A']-source_parts['B'];full_energy=float(source_parts['full']@source_parts['full'])
            for method in methods:
                parts={c:np.array(lookup[case_id,method,c]['probability_endpoints'][scope]['candidate_nll_deltas']) for c in ('full','A','B')}
                interaction=parts['full']-parts['A']-parts['B'];error=interaction-source_interaction
                compositions.append(dict(case_id=case_id,method=method,scope=scope,source_interaction=source_interaction.tolist(),candidate_interaction=interaction.tolist(),
                    source_interaction_rms=float(np.sqrt(np.mean(source_interaction**2))),candidate_interaction_rms=float(np.sqrt(np.mean(interaction**2))),
                    interaction_error_over_full_source_energy=float(error@error)/full_energy if full_energy else None,
                    source_interaction_over_full_source_energy=float(source_interaction@source_interaction)/full_energy if full_energy else None))
    aggregates=[]
    for scope in ('intervention_positions','same_document_downstream'):
        for component in ('full','A','B'):
            for method in ('long','raw'):
                rr=[r for r in cases if (r['scope'],r['component'],r['method'])==(scope,component,method)]
                aggregates.append(dict(scope=scope,component=component,method=method,cases=len(rr),**{k:med([r[k] for r in rr]) for k in metrics}))
    paired=[]
    for component in ('full','A','B'):
        rr=[r for r in table if r['scope']=='intervention_positions' and r['component']==component and r['method'].startswith('long_')]
        raws={r['case_id']:r for r in table if r['scope']=='intervention_positions' and r['component']==component and r['method']=='raw'}
        for k in metrics:
            paired.append(dict(component=component,metric=k,target_cases=len(rr),non_null=sum(r[k] is not None for r in rr),
                below_no_intervention=sum(r[k] is not None and r[k]<1 for r in rr),lower_than_raw=sum(r[k] is not None and raws[r['case_id']][k] is not None and r[k]<raws[r['case_id']][k] for r in rr)))
    # Preserve and explain the preceding failed validator without repainting its run.
    previous=ROOT/'runs'/cfg['previous_run'];prevsummary=json.loads((previous/'metrics.summary.json').read_text());prevrows=jsonl(previous/'metrics.raw.jsonl')
    assert prevsummary['status']=='FAIL' and prevsummary['error'] is None
    assert [k for k,v in prevsummary['checks'].items() if not v]==['cached_hook_replay']
    prevnumeric=json.loads((previous/'numerical_checks.json').read_text())
    oldconsumer=ROOT/'runs/F4_long_k128_newseeds_expanded_v1_20260905'
    oldsummary=json.loads((oldconsumer/'metrics.summary.json').read_text());assert oldsummary['status']=='PASS'
    maximum=max(prevnumeric['cached_hook_replay']);prior=oldsummary['max_replay_relative']
    assert abs(maximum-prior)<1e-10 and maximum<=1e-4
    previous_evidence=dict(run=str(previous),original_status='FAIL retained',failure='new script accidentally tightened existing cache batch4/live batch1 tolerance from1e-4 to1e-5',
        max_replay_relative=maximum,prior_same_case_replay_relative=prior,existing_consumer_tolerance=1e-4,
        source_tail_zero_cases=sum(r['method']=='source' and r['component']=='B' and r['probability_endpoints']['intervention_positions']['source_nll_delta_rms']==0 for r in prevrows),
        false_tail_prediction_rms=[dict(case_id=r['case_id'],method=r['method'],nll_rms=r['probability_endpoints']['intervention_positions']['nll_delta_rmse']) for r in prevrows if r['component']=='B' and r['method']!='source'])
    inputs=[dict(path=str(p),sha256=sha256(p)) for p in (run/'metrics.raw.jsonl',run/'config.resolved.json',run/'component_fits.json',run/'component_coordinates.npz',previous/'metrics.raw.jsonl',previous/'metrics.summary.json',previous/'numerical_checks.json',oldconsumer/'metrics.summary.json')]
    result=dict(scope=cfg['scope'],cases=cases,aggregates=aggregates,paired=paired,compositions=compositions,previous_head_tail=previous_evidence,
        validation=dict(raw_probability_ratios_recomputed=ratios,source_anchors_consistent=True,all_20_dependent_target_cases_per_component=True),
        inputs=inputs,generator_sha256=sha256(Path(__file__)))
    write(run/'component_summary.json',result)
    for name,rr in [('COMPONENT_ROWS.csv',table),('COMPONENT_CASES.csv',cases),('COMPONENT_INTERACTIONS.csv',compositions)]:
        with (run/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    def fmt(v):return 'NA' if v is None else f'{v:.6g}'
    lines=['# Source组成部分：真实作用与跨seed预测','',cfg['scope'],'',
        '两部分由原discovery能量排名奇偶交替决定；不是语义标签或两个非共线机制。所有target在相同source组件参考、basis和family剂量下比较，raw得到同两输出拟合信息。',
        '组装时full系数为A+B并重放旧full；真实NLL交互另测，不拿向量可加冒充行为可加。五case/三query及共享seed不是独立重复；组设计看过本批source诊断，属于开发。',
        '', '## 主要端点：先按case内四target中位数，再跨五case中位数','',
        '|部分|方法|KL误差|NLL变化平方误差|','|---|---|---:|---:|']
    for r in aggregates:
        if r['scope']=='intervention_positions':lines.append(f"|{r['component']}|{r['method']}|{fmt(r[metrics[0]])}|{fmt(r[metrics[1]])}|")
    lines+=['','## 逐case主端点（全部保留）','','|Case / source / condition|部分|source NLL RMS|long KL|raw KL|long NLL|raw NLL|','|---|---|---:|---:|---:|---:|---:|']
    for cid in range(5):
        for component in ('full','A','B'):
            rr={r['method']:r for r in cases if r['case_id']==cid and r['scope']=='intervention_positions' and r['component']==component};l=rr['long'];raw=rr['raw']
            lines.append('|'+ '|'.join([f"{cid} / {l['source_seed']}:{l['source_atom']} / {l['condition']}",component,fmt(l['source_nll_delta_rms']),fmt(l[metrics[0]]),fmt(raw[metrics[0]]),fmt(l[metrics[1]]),fmt(raw[metrics[1]])])+'|')
    lines+=['','## 前版和数值状态说明','',
        '前版head/tail的source B在五case均无作用；两个query在discovery仅8/10成员有变差，inactive部分在独立mean中心化的state拟合中仍有常数目标，可能产生伪变化。不能把非零target B当source机制，不能将NA当零误差。',
        'v1原始状态FAIL保留：唯一未过的是新脚本误收紧到1e-5的cached hook检查。其实测1.1723889e-5与既有已通过同case1.1723891e-5一致；原consumer固定容差为1e-4。本版恢复原规则，未改v1任何raw/config/status。',
        '', '完整两范围、逐target、source能量/剂量及非线性交互见CSV和component_summary.json。ranges不是CI；无新数据确认、native编辑、独立语义或FCC胜raw声明。']
    (run/'COMPONENT_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(aggregates=aggregates,paired=paired,validation=result['validation'])),flush=True)


if __name__=='__main__':main()
