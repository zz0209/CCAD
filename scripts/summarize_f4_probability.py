"""Raw-backed probability comparison on all frozen source-selected matched cases."""
import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict
from run_f4_source_reference_causal import jsonl, write, sha256

ROOT=Path(__file__).resolve().parents[1]
METHODS=('target','global_rows','raw','readout_top16')
METRICS=('normalized_kl_error','normalized_nll_delta_squared_error','source_kl_mean','source_nll_delta_mean','candidate_nll_delta_mean','nll_delta_rmse','source_nll_delta_rms')
def key(r):return tuple(r[k] for k in ('source_seed','source_atom','condition','sequence','target_seed','method'))
def median(v):
    v=[x for x in v if x is not None]
    return statistics.median(v) if v else None
def fmt(v):return 'NA' if v is None else f'{v:.6g}'


def main():
    output=ROOT/'runs/F4_probability_expanded_v1_20260905'
    if (output/'probability_summary.json').exists():raise ValueError('Summary already exists')
    inputs=[];table=[];cases=[];forwards=0;seconds=0;checks={}
    for panel in ('original','expanded'):
        run=ROOT/'runs'/f'F4_probability_{panel}_v1_20260905'
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        summary=json.loads((run/'metrics.summary.json').read_text());forwards+=summary['model_forwards'];seconds+=summary['wall_seconds']
        matching=ROOT/'runs/F4_token_class_donor_preparation_v1_20260905'/f'{panel}_matching.json'
        choices=json.loads(matching.read_text())['choices']
        prior={kind:jsonl(ROOT/'runs'/f'F4_{kind}_{panel}_v1_20260905/metrics.raw.jsonl') for kind in ('cases','class_matched')}
        prior={kind:{key(r):r for r in rows} for kind,rows in prior.items()}
        rows=jsonl(run/'metrics.raw.jsonl');groups=defaultdict(list)
        for r in rows:groups[r['source_seed'],r['source_atom'],r['condition']].append(r)
        selected=[c for c in choices if c['entry'] and c['source_scope']['selected']]
        assert len(rows)==16*len(selected)
        for choice in selected:
            s,a,c=choice['source_seed'],choice['source_atom'],choice['condition'];group=groups[s,a,c];assert len(group)==16
            old=prior['cases' if choice['matching_status']=='UNCHANGED_REUSABLE' else 'class_matched']
            for r in group:
                previous=old[key(r)]
                for field in ('endpoints','hook','common_source_dose_scale','sequence','donor_sequence','intervention_positions','donor_positions'):assert r[field]==previous[field],field
                for scope,p in r['probability_endpoints'].items():
                    assert p['position_count']>0
                    assert p['normalized_kl_error'] is None or p['normalized_kl_error']>=0
                    table.append(dict(panel=panel,source_seed=s,source_atom=a,condition=c,target_seed=r['target_seed'],method=r['method'],scope=scope,position_count=p['position_count'],**{m:p[m] for m in METRICS}))
            record=dict(panel=panel,source_seed=s,source_atom=a,condition=c,entry=choice['entry'],scopes={})
            for scope in ('intervention_positions','same_document_downstream'):
                reference=group[0]['probability_endpoints'][scope]
                assert all(r['probability_endpoints'][scope]['source_to_baseline_kl']==reference['source_to_baseline_kl'] and r['probability_endpoints'][scope]['source_nll_deltas']==reference['source_nll_deltas'] for r in group)
                record['scopes'][scope]=dict(positions=reference['positions'],observed_next_token_ids=reference['observed_next_token_ids'],source_nll_deltas=reference['source_nll_deltas'],source_to_baseline_kl=reference['source_to_baseline_kl'],methods={})
                for method in METHODS:
                    sub=[r['probability_endpoints'][scope] for r in group if r['method']==method]
                    record['scopes'][scope]['methods'][method]={m:median([r[m] for r in sub]) for m in METRICS}
            cases.append(record)
        checks[panel]=dict(selected_cases=len(selected),method_rows=len(rows),all_previous_logit_state_hook_dose_exact=True,source_reference_identical=True)
        for p in (matching,run/'config.resolved.json',run/'metrics.raw.jsonl',run/'metrics.summary.json',*[ROOT/'runs'/f'F4_{kind}_{panel}_v1_20260905/metrics.raw.jsonl' for kind in ('cases','class_matched')]):
            inputs.append(dict(path=str(p),sha256=sha256(p)))
    aggregates=[]
    for panel in ('original','expanded'):
        pc=[r for r in cases if r['panel']==panel]
        for scope in ('intervention_positions','same_document_downstream'):
            for method in METHODS:
                values=[r['scopes'][scope]['methods'][method] for r in pc]
                aggregates.append(dict(panel=panel,scope=scope,method=method,cases=len(pc),medians={m:median([r[m] for r in values]) for m in METRICS},
                    valid_kl=sum(r['normalized_kl_error'] is not None for r in values),valid_nll=sum(r['normalized_nll_delta_squared_error'] is not None for r in values),
                    both_below_zero=sum(r['normalized_kl_error'] is not None and r['normalized_nll_delta_squared_error'] is not None and r['normalized_kl_error']<1 and r['normalized_nll_delta_squared_error']<1 for r in values)))
    result=dict(checks=checks,cases=cases,aggregates=aggregates,inputs=inputs,model_forwards=forwards,wall_seconds=seconds,generator_script_sha256=sha256(Path(__file__)),
        scope='Exposed-document development, all11frozen source-selected matched cases. Median over4dependent target seeds then cases; shared source/query/documents not independent. Probability fidelity is not semantic uniqueness. Joint-position intervention, not per-position causal attribution.')
    write(output/'probability_summary.json',result)
    with (output/'probability_rows.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# 固定配对的预测概率端点','',result['scope'],'',f'新增 {forwards} 前向 / {seconds:.6f} 秒（含数值端点和案例导出）。所有176方法行的旧logits/state/hook/dose逐值相等。','',
        'KL误差为 pooled KL(source||candidate) / pooled KL(source||baseline)；NLL误差为观测词loss变化向量相对source的平方误差。0=精确保留，1=不干预参照；不是正确率。正NLL delta表示观测词变差，负表示改善。数值以nats计。主端点只在原干预位置，辅助端点在相同文档下游；两者均排除EOS预测和无下一词的末尾。','',
        '## 聚合（先四target中位数，再case中位数）','','|面板|范围|方法|case|KL误差|NLL变化误差|双端点<1|','|---|---|---|---:|---:|---:|---:|']
    for r in aggregates:lines.append('|'+ '|'.join([r['panel'],r['scope'],r['method'],str(r['cases']),fmt(r['medians']['normalized_kl_error']),fmt(r['medians']['normalized_nll_delta_squared_error']),str(r['both_below_zero'])])+'|')
    lines+=['','## 全部11案例：主端点','','|面板/source:atom/条件|source KL均值|source NLL变化均值|source NLL变化RMS|full KL/NLL|global KL/NLL|raw KL/NLL|top16 KL/NLL|','|---|---:|---:|---:|---|---|---|---|']
    for r in cases:
        m=r['scopes']['intervention_positions']['methods'];s=m['target']
        lines.append('|'+ '|'.join([f"{r['panel']} {r['source_seed']}:{r['source_atom']} {r['condition']}",fmt(s['source_kl_mean']),fmt(s['source_nll_delta_mean']),fmt(s['source_nll_delta_rms'])]+[fmt(m[n]['normalized_kl_error'])+'/'+fmt(m[n]['normalized_nll_delta_squared_error']) for n in METHODS])+'|')
    lines+=['','逐target/范围见 probability_rows.csv；逐位置source KL/NLL及完整冻结entry、来源hash见 probability_summary.json。未评估的21/32请求继续保留在上游matching，不作为概率成功。原8/扩展8只是原query面板结构，本次实际为原1/扩展5个不同query。11正负配对与共享seed有依赖，不能据此生成独立重复显著性。']
    (output/'PROBABILITY_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(checks=checks,aggregates=aggregates,model_forwards=forwards,wall_seconds=seconds)))


if __name__=='__main__':main()
