"""Full development extension: preserve small-panel comparison and all controls."""
import csv
import json
from pathlib import Path
import numpy as np
from run_f4_agreement_source import ROOT
from run_r011s1_raw_hook_asset import entry,write_json as write
from ccad.artifacts import sha256


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s]


def main():
    run=ROOT/'runs/F4_task_paired_relations_full_dev_v1_20260905'
    out=run/'EXTENSION_COMPARISON.md'
    if out.exists():raise FileExistsError(out)
    small=ROOT/'runs/F4_task_paired_relations_causal_v1_20260905'
    rows=read(run/'metrics.raw.jsonl');before=read(small/'metrics.raw.jsonl')
    keys=lambda r:(r['method'],r.get('axis'),r['id'])
    lookup={keys(r):{k:v for k,v in r.items() if k!='run_id'} for r in rows}
    assert all(lookup[keys(r)]=={k:v for k,v in r.items() if k!='run_id'} for r in before)
    pilot_ids={r['id'] for r in before if 'target_seed' in r}
    source={(r['id'],r['axis']):np.array(r['margin_loss']) for r in rows if r['method']=='source_native_group'}
    table=[]
    for name in ('pilot8','extension56','all64'):
        select=lambda r:name=='all64' or ((r['id'] in pilot_ids)==(name=='pilot8'))
        for method in sorted({r['method'] for r in rows if 'target_seed' in r}):
            for axis in ('subject','attractor'):
                rr=[r for r in rows if r['method']==method and r['axis']==axis and select(r)]
                y=np.array([r['margin_loss'] for r in rr]);s=np.array([source[r['id'],axis] for r in rr])
                errors=np.sum((y-s)**2,axis=0)/np.sum(s*s,axis=0)
                by_template={t:np.mean([r['margin_loss'][0] for r in rr if r['template']==t]) for t in {r['template'] for r in rr}}
                table.append(dict(subset=name,method=method,axis=axis,n=len(rr),templates=len(by_template),
                    positive_template_means=int(sum(v>0 for v in by_template.values())),mean=float(y[:,0].mean()),
                    primary_error=float(errors[0]),past_error=float(errors[1]),mean_abs_primary=float(np.abs(y[:,0]).mean()),
                    mean_abs_tense=float(np.abs(y[:,2]).mean()),mean_hook_fraction=float(np.mean([r['hook_fraction'] for r in rr]))))
    with (run/'EXTENSION_TABLE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write(run/'extension_comparison.json',dict(rows=table,small_panel_exact_rows_replayed=len(before),
        evidence=[entry(p,'CCAD observed results','evidence') for p in (run/'metrics.raw.jsonl',small/'metrics.raw.jsonl')],
        generator_script_sha256=sha256(Path(__file__)),scope='Fixed-method development extension, no independent-seed p-values'))
    lines=['# 新paired对应：完整开发扩展','',
        '主要结果：任务邻近配对恢复跨seed FCC的部分作用。小面板中低于raw的两个主指标结果在完整64条上没有维持：raw主误差仍最低。Code FCC四方向均好于decoded FCC，且4/4方向past误差低于raw；code输入容量更大。不能写成全面超过raw或核心解释性贡献已成立。','',
        '|方法|完整64主误差|新增56主误差|完整64 past误差|完整64主语mean|完整64主语dose|','|---|---:|---:|---:|---:|---:|']
    index={(r['subset'],r['method'],r['axis']):r for r in table}
    for r in table:
        if r['subset']=='all64' and r['axis']=='subject':
            extra=index['extension56',r['method'],'subject']
            lines.append(f"|{r['method']}|{r['primary_error']:.6f}|{extra['primary_error']:.6f}|{r['past_error']:.6f}|{r['mean']:.6f}|{r['mean_hook_fraction']:.6f}|")
    lines+=['','所有误差均为Σ(candidate−source作用)^2/Σ(source作用)^2；零操作为1。新增56仍是开发扩展，不是预留确认数据。四target共享source，64条分属16模板，不当独立seed重复。',
        '', '连带作用：code FCC的attractor时态变化较小，但其实际剂量也小于raw。raw只在subject差分拟合，attractor是分布外操作轴。当前不能把小幅变化直接当作选择性优势；下一小对照把raw向量逐输入缩放到各code FCC相同范数，保持方向和其他条件不变，直接检验剂量解释。',
        '', '已有作用、全局/邻域负结果、投影路线和自然文本确认全部保留。此处FCC是在source weighted-decoder span中执行的有符号code贡献transport，不等于target native64删除；native64仍弱，不能偷换操作类别。',
        '', '资产：128新词汇化模板/512discovery输入；source-only规则含任务条件监督。原任务development/reserved全部名词排除。原reserved64仅文本生成，未编码/前向；新paired非discovery分组也未编码。',
        '', f'复核：小面板{len(before)}行逐值重放；完整面板3136行margin和2688预测/共同scale重算；非FCC编译数组在两种fit逐值一致。仅验证身份/数值，不是独立科学审查。',
        '', '下一步：先在同开发输入做一次raw匹配code FCC范数的窄对照（不重拟合、不扫阈值）；据完整方法结果冻结确认端点/剂量操作，再用原reserved新词汇/介词确认有效范围，随后发展有符号组成部分消融/组合预测。不能只凭有效transport宣称native或唯一可读概念。','']
    out.write_text('\n'.join(lines),encoding='utf-8');print(out)
    print(json.dumps([r for r in table if r['subset']=='extension56' and r['axis']=='subject' and (r['method'].startswith('fcc_codes') or r['method']=='raw_fitted')]))


if __name__=='__main__':main()
