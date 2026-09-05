"""Shared-support component fidelity with every short/long/full/atom/raw control."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from run_f4_source_reference_causal import ROOT,np,write,jsonl,sha256

METRICS=('normalized_kl_error','normalized_nll_delta_squared_error')
METHODS=('short_full','short_shared16','long_full','long_shared16','short_single','long_single','raw')


def method_name(name):
    return name.rsplit('_',1)[0] if name.startswith(('long_','short_')) else name


def med(values):
    vv=[v for v in values if v is not None]
    return median(vv) if vv else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run if args.run.is_absolute() else ROOT/args.run
    if (run/'compact_component_summary.json').exists():raise FileExistsError('Already summarized')
    cfg=json.loads((run/'config.resolved.json').read_text());summary=json.loads((run/'metrics.summary.json').read_text())
    assert summary['status']=='PASS' and json.loads((run/'contract_validation.json').read_text())['ok']
    parent=ROOT/cfg['frozen_components']['path'];raw=run/'metrics.raw.jsonl';parent_raw=parent/'metrics.raw.jsonl'
    assert sha256(parent_raw)==cfg['frozen_components']['raw_sha256']
    rows=jsonl(raw);old=jsonl(parent_raw);assert len(rows)==315 and len(old)==90
    source={(r['case_id'],r['component']):r for r in rows if r['method']=='source'}
    oldsource={(r['case_id'],r['component']):r for r in old if r['method']=='source'}
    for key,r in source.items():
        p=oldsource[key]
        for field in ('common_family_scale','old_full_scale','sequence','donor_sequence','document_ids','donor_document_ids','intervention_positions','donor_positions'):assert r[field]==p[field]
        assert r['probability_endpoints']==p['probability_endpoints'],'Old controls require identical measured source anchors'
    for r in old:
        if r['method']!='source':
            r=dict(r);r['method']=r['method'].replace('long_','long_full_');rows.append(r)
    assert len(rows)==390
    lookup={(r['case_id'],r['method'],r['component']):r for r in rows};assert len(lookup)==390
    table=[];ratios=0
    for r in rows:
        for scope,p in r['probability_endpoints'].items():
            src=source[r['case_id'],r['component']]['probability_endpoints'][scope]
            assert p['source_nll_deltas']==src['source_nll_deltas'] and p['source_to_baseline_kl']==src['source_to_baseline_kl']
            ds=np.array(p['source_nll_deltas']);dc=np.array(p['candidate_nll_deltas']);se=float(ds@ds);kl=sum(p['source_to_baseline_kl']);kc=sum(p['source_to_candidate_kl'])
            for name,expected in zip(METRICS,(kc/kl if kl>1e-12*len(ds) else None,float(np.sum((ds-dc)**2))/se if se>1e-12*len(ds) else None)):
                actual=p[name];assert (expected is None and actual is None) or np.isclose(expected,actual,rtol=1e-10,atol=1e-12);ratios+=1
            method=method_name(r['method']);target=int(r['method'].rsplit('_',1)[1]) if method not in ('source','raw') else None
            table.append(dict(case_id=r['case_id'],source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],method=method,target_seed=target,
                component=r['component'],scope=scope,**{k:p[k] for k in METRICS},source_nll_delta_rms=p['source_nll_delta_rms'],source_kl_mean=p['source_kl_mean'],
                source_hook_fraction=r['source_hook_fraction'],candidate_hook_fraction=r['candidate_hook_fraction'],common_family_scale=r['common_family_scale']))
    cases=[];aggregates=[];comparisons=[];interactions=[]
    for cid in range(5):
        for scope in ('intervention_positions','same_document_downstream'):
            for component in ('full','A','B'):
                for method in METHODS:
                    rr=[r for r in table if (r['case_id'],r['scope'],r['component'],r['method'])==(cid,scope,component,method)]
                    assert len(rr)==(1 if method=='raw' else 4)
                    cases.append(dict(case_id=cid,scope=scope,component=component,method=method,source_seed=rr[0]['source_seed'],source_atom=rr[0]['source_atom'],condition=rr[0]['condition'],targets=len(rr),
                        **{k:med([r[k] for r in rr]) for k in METRICS}))
            sd={c:np.array(source[cid,c]['probability_endpoints'][scope]['source_nll_deltas']) for c in ('full','A','B')};si=sd['full']-sd['A']-sd['B'];energy=float(sd['full']@sd['full'])
            for name in sorted({r['method'] for r in rows if r['case_id']==cid}):
                pred={c:np.array(lookup[cid,name,c]['probability_endpoints'][scope]['candidate_nll_deltas']) for c in ('full','A','B')};pi=pred['full']-pred['A']-pred['B'];err=pi-si
                interactions.append(dict(case_id=cid,scope=scope,method=name,source_interaction_rms=float(np.sqrt(np.mean(si*si))),candidate_interaction_rms=float(np.sqrt(np.mean(pi*pi))),
                    error_over_full_source_energy=float(err@err)/energy if energy else None))
    for scope in ('intervention_positions','same_document_downstream'):
        for component in ('full','A','B'):
            for method in METHODS:
                rr=[r for r in cases if (r['scope'],r['component'],r['method'])==(scope,component,method)]
                aggregates.append(dict(scope=scope,component=component,method=method,cases=len(rr),**{k:med([r[k] for r in rr]) for k in METRICS}))
            for reference in ('short_shared16','long_single','long_full','raw'):
                rr=[r for r in table if (r['scope'],r['component'],r['method'])==(scope,component,'long_shared16')]
                ref={(r['case_id'],r['target_seed']):r for r in table if (r['scope'],r['component'],r['method'])==(scope,component,reference)}
                for metric in METRICS:
                    comparisons.append(dict(scope=scope,component=component,candidate='long_shared16',reference=reference,metric=metric,target_cases=len(rr),
                        candidate_lower=sum(r[metric] is not None and ref[r['case_id'],None if reference=='raw' else r['target_seed']][metric] is not None and r[metric]<ref[r['case_id'],None if reference=='raw' else r['target_seed']][metric] for r in rr),
                        below_no_intervention=sum(r[metric] is not None and r[metric]<1 for r in rr),non_null=sum(r[metric] is not None for r in rr)))
    fits=json.loads((run/'component_fits.json').read_text());supports=[r for r in fits if r['kind'].endswith('_shared16')];atoms=[r for r in fits if r['kind'].endswith('_single')]
    assert len(supports)==24 and all(len(r['support'])==len(set(r['support']))==16 and r['union_budget']==16 for r in supports) and len(atoms)==24
    inputs=[dict(path=str(p),sha256=sha256(p)) for p in (raw,run/'config.resolved.json',run/'component_fits.json',run/'component_coordinates.npz',parent_raw,parent/'component_fits.json')]
    result=dict(scope=cfg['scope'],cases=cases,aggregates=aggregates,comparisons=comparisons,interactions=interactions,shared_supports=supports,joint_atoms=atoms,inputs=inputs,
        checks=dict(parent_source_anchors_exact=True,all_controls_and_targets_retained=True,probability_ratios_recomputed=ratios,shared_union16=True),
        model_forwards=summary['model_forwards'],parent_control_forwards_reused_not_new=75,timings=summary['timings'],generator_sha256=sha256(Path(__file__)))
    write(run/'compact_component_summary.json',result)
    for name,rr in [('COMPACT_COMPONENT_ROWS.csv',table),('COMPACT_COMPONENT_CASES.csv',cases),('COMPACT_COMPONENT_INTERACTIONS.csv',interactions)]:
        with (run/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    def fmt(v):return 'NA' if v is None else f'{v:.6g}'
    lines=['# 一个共享16成员接口：两个source部分的作用复用','',cfg['scope'],'',
        'A、B使用完全相同的16个target成员，支持并集16；full是两列相加，不是为full另选16。joint单atom同样只用一个atom拟合两列。',
        '源分组、配对、basis、dose均冻结；长/短按同一规则。long/full及raw原75行直接复用，所有新source实测端点逐值相同后才合并；父run的旧负结果不改。',
        '下表先取同case四target中位，再跨五case中位；5case/3query/20target-case有依赖。短长训练流不同，不能归因为单独训练时长的严格学习曲线。',
        '', '## 主端点全部方法','', '|部分|方法|KL误差|NLL变化平方误差|','|---|---|---:|---:|']
    for r in aggregates:
        if r['scope']=='intervention_positions':lines.append(f"|{r['component']}|{r['method']}|{fmt(r[METRICS[0]])}|{fmt(r[METRICS[1]])}|")
    lines+=['','## 逐case共享16比较（主端点）','','|case/source/condition|部分|short KL|long KL|short NLL|long NLL|','|---|---|---:|---:|---:|---:|']
    for cid in range(5):
        for component in ('full','A','B'):
            rr={r['method']:r for r in cases if r['case_id']==cid and r['component']==component and r['scope']=='intervention_positions'};a=rr['short_shared16'];b=rr['long_shared16']
            lines.append('|'+ '|'.join([f"{cid}/{a['source_seed']}:{a['source_atom']}/{a['condition']}",component,fmt(a[METRICS[0]]),fmt(b[METRICS[0]]),fmt(a[METRICS[1]]),fmt(b[METRICS[1]])])+'|')
    lines+=['','该排序是忽略项间协方差的固定能量启发式，不是最优稀疏拟合。组成操作仍位于source rank1方向，不是target native或独立语义机制；原八请求的三项未评估保持缺失。',
        '全逐target、两端点范围、joint单atom、source剂量和非线性交互见CSV/JSON。开发正信号须冻结后在新输入确认。']
    (run/'COMPACT_COMPONENT_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(aggregates=[r for r in aggregates if r['scope']=='intervention_positions'],comparisons=[r for r in comparisons if r['scope']=='intervention_positions'],checks=result['checks'])),flush=True)


if __name__=='__main__':main()
