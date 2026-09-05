"""Build an auditable descriptive casebook from completed preselected replays."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict
from run_f4_source_reference_causal import sha256, jsonl, write

ROOT=Path(__file__).resolve().parents[1]
METHODS=('target','global_rows','raw','readout_top16')


def key(r): return tuple(r[k] for k in ('source_seed','source_atom','condition','sequence','target_seed','method'))
def esc(s): return str(s).replace('|','\\|').replace('\n',' ↵ ').replace('\r','')
def med(values): return statistics.median(values)
def f(v): return f'{v:.6g}'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--original',required=True);parser.add_argument('--expanded',required=True);args=parser.parse_args()
    output=ROOT/'runs'/args.expanded
    if (output/'CASEBOOK.md').exists(): raise ValueError('Casebook already exists')
    table=[];cases=[];inputs=[];checks={};details_count=0;residuals=[];forwards=0;seconds=0
    lines=['# 固定 source-only 案例包：真实文本、signed贡献与反事实作用','',
           '原/扩展两面板共32个请求条件，31个有支持文本对、1个缺失。固定选择先取原顺序首个source-selected有支持输入，否则首个有支持输入。全部四target和四方法保留。旧确认文档已暴露：这是解释性重放，不是新确认；并非人工标注或唯一概念证明。','',
           '误差沿用全序列source-normalized平方误差，1为zero。下列主文本展示每对source贡献绝对值最大的位置（同值按原位置排序），并列出全部干预位置；不依据target效果选位置。完整各位置及所有非零signed atom项在case_details.jsonl。上下文只给当前位置之前12token内且不跨文档boundary的prefix。显示token之后的观测token仅用于描述，不参与选择。','',
           '有符号贡献是被减去的hook分量；输出Δ是“干预后−原输出”的centered logit变化，和原误差计算的removal effect符号相反，但两侧同时反号不改变误差。正/负条件是source激活选择条件，不是词义极性。raw没有SAE atom分解，不能伪造。','',
           '每个logit明细来自该recipient全部已选位置的联合干预；后面位置可能受前面干预影响，不能将某位置的logit变化独立归因于该位置atom项。source能量位置占比是hook空间描述，不是下游效应归因。','']
    for panel,name in [('original',args.original),('expanded',args.expanded)]:
        run=ROOT/'runs'/name;old=ROOT/'runs'/f'F4_fit_scope_confirmation_{panel}_v1_20260905'
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        base={key(r):r for r in jsonl(old/'metrics.raw.jsonl')};raw=jsonl(run/'metrics.raw.jsonl');details=jsonl(run/'case_details.jsonl')
        summary=json.loads((run/'metrics.summary.json').read_text());forwards+=summary['model_forwards'];seconds+=summary['wall_seconds']
        assert len(raw)==len(details) and len({key(r) for r in details})==len(details)
        rawindex={key(r):r for r in raw};groups=defaultdict(list)
        for row in details:
            k=key(row);r=rawindex[k];prior=base[k]
            assert r['endpoints']==prior['endpoints'] and row['endpoints']==r['endpoints']
            assert r['hook']==prior['hook'] and r['common_source_dose_scale']==prior['common_source_dose_scale']
            for p in row['positions']:
                residuals.append(p['source_reconstruction_residual'])
                if p['target_reconstruction_residual'] is not None:residuals.append(p['target_reconstruction_residual'])
                for side in ('source_atoms','target_atoms'):
                    atom=p[side]
                    if atom is not None:
                        assert abs(sum(v['signed_contribution'] for v in atom['terms'])-atom['total'])<1e-8
            groups[k[:4]].append(row)
            table.append(dict(panel=panel,**{n:row[n] for n in ('source_seed','source_atom','condition','sequence','target_seed','method')},
                              logits_error=r['endpoints']['centered_logits']['normalized_error'],state_error=r['endpoints']['next_state']['normalized_error']))
        choices=json.loads((run/'case_selection.json').read_text())['choices']
        for choice in choices:
            title=f"{panel} s{choice['source_seed']}:{choice['source_atom']} {choice['condition']}"
            lines+=['## '+title,'']
            if choice['entry'] is None:
                lines+=['无支持recipient/donor；保留缺失，不赋予成功或零误差。',''];cases.append(dict(panel=panel,**choice,methods=None));continue
            e=choice['entry'];k=(choice['source_seed'],choice['source_atom'],choice['condition'],e['sequence']);rows=groups[k]
            assert len(rows)==16 and {r['method'] for r in rows}==set(METHODS)
            first=rows[0];position_index=max(range(len(first['positions'])),key=lambda i:abs(first['positions'][i]['source_atoms']['total']))
            p=first['positions'][position_index];sc=p['source_atoms'];rctx=p['recipient'];dctx=p['donor']
            case=dict(panel=panel,source_seed=k[0],source_atom=k[1],condition=k[2],sequence=k[3],selected=choice['source_scope']['selected'],
                      recipient=rctx,donor=dctx,source_atoms=sc,source_logits=p['logits'],methods={},targets=[],
                      positions=[dict(recipient=v['recipient'],donor=v['donor'],source_total=v['source_atoms']['total'],source_nonzero_atoms=v['source_atoms']['nonzero_atoms']) for v in first['positions']])
            lines += [f"范围：{'selected' if case['selected'] else 'fallback'}；recipient seq{e['sequence']} / donor seq{e['donor_sequence']}；共{len(first['positions'])}位置。",'',
                      f"Recipient：{esc(rctx['prefix'])} **[{esc(rctx['token'])}]**（pos{rctx['position']}；下一观测token `{esc(rctx['next_observed_token'])}`）",'',
                      f"Donor：{esc(dctx['prefix'])} **[{esc(dctx['token'])}]**（pos{dctx['position']}）",'',
                      f"Source标量和={f(sc['total'])}；正项和={f(sc['positive_sum'])}，负项和={f(sc['negative_sum'])}，非零atom={sc['nonzero_atoms']}。绝对贡献最大三项："+', '.join(f"{x['atom']}:{f(x['signed_contribution'])}" for x in sc['terms'][:3])+'.','',
                      '| 方法 | logits误差（4target中位数） | state误差 |','|---|---:|---:|']
            for method in METHODS:
                sub=[r for r in rows if r['method']==method];vals=[med([r['endpoints'][ep]['normalized_error'] for r in sub]) for ep in ('centered_logits','next_state')]
                case['methods'][method]=vals;lines.append(f'| {method} | {f(vals[0])} | {f(vals[1])} |')
            lines+=['','Source干预在该位置使下列next-token logits增加/减少（仅展示极值，不代表语义选择性）：','',
                    '增加：'+', '.join(f"`{esc(x['token'])}` {f(x['source_delta'])}" for x in p['logits']['source_top_increased'][:3]),'',
                    '减少：'+', '.join(f"`{esc(x['token'])}` {f(x['source_delta'])}" for x in p['logits']['source_top_decreased'][:3]),'',
                    '| target | full 标量和/非零atom | global 标量和/非零atom | full最大两项（signed） |','|---|---:|---:|---|']
            for t in sorted({r['target_seed'] for r in rows}):
                full=next(r for r in rows if r['target_seed']==t and r['method']=='target')['positions'][position_index]['target_atoms']
                glob=next(r for r in rows if r['target_seed']==t and r['method']=='global_rows')['positions'][position_index]['target_atoms']
                case['targets'].append(dict(target_seed=t,full=full,global_rows=glob))
                lines.append(f"| {t} | {f(full['total'])}/{full['nonzero_atoms']} | {f(glob['total'])}/{glob['nonzero_atoms']} | "+', '.join(f"{x['atom']}:{f(x['signed_contribution'])}" for x in full['terms'][:2])+' |')
            lines+=['','| 所有位置 | Recipient prefix[token] | Donor prefix[token] | source标量 | 非零atom |','|---|---|---|---:|---:|']
            for v in case['positions']:
                rc=v['recipient'];dc=v['donor']
                lines.append(f"| {rc['position']} | {esc(rc['prefix'])}[{esc(rc['token'])}] | {esc(dc['prefix'])}[{esc(dc['token'])}] | {f(v['source_total'])} | {v['source_nonzero_atoms']} |")
            lines+=['','以上atom项仅给出实际坐标分解，不证明atom必要性；source-aligned读出不等于native删除。',''];cases.append(case)
        checks[panel]=dict(endpoint_and_hook_exact_rows=len(raw),case_detail_rows=len(details),supported_cases=len(groups),missing_conditions=sum(c['entry'] is None for c in choices))
        details_count+=len(details)
        for path in [run/'case_selection.json',run/'case_details.jsonl',run/'metrics.raw.jsonl',run/'config.resolved.json',old/'metrics.raw.jsonl']:
            inputs.append(dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size))
    result=dict(checks=checks,exact_replayed_rows=details_count,max_signed_reconstruction_residual=max(residuals),model_forwards=forwards,wall_seconds=seconds,
                inputs=inputs,generator_script_sha256=sha256(Path(__file__)),cases=cases,
                scope='Descriptive exposed-document replay; source-only case choice does not make prior target exposure disappear. No statistical independence or human semantics claim.')
    write(output/'case_summary.json',result)
    with (output/'case_table.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines+=['## 重放核对','',f'{details_count}方法行的完整endpoint、hook指标与原确认行逐值相等；所有非零atom项重建检查通过，最大residual={f(max(residuals))}。共{forwards}前向/{f(seconds)}秒（含导出），不新增训练或拟合。这不是独立科学复核。','',
            'case_table.csv保留逐target/方法效应；case_summary.json保留输入hash、结构化案例与检查；原/扩展run的case_details.jsonl保留每位置完整非零项和四方法token变化。语义结论必须另写为AI工作假说，不能把本案例包更名为独立标签集。']
    (output/'CASEBOOK.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','inputs')},indent=2))


if __name__=='__main__': main()
