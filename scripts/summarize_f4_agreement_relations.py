"""Replay task-contrast and cross-seed component observations; keep all methods."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
from run_f4_agreement_source import ROOT,margins,swap_indices,capped
from run_r011s1_raw_hook_asset import entry,write_json as write
from ccad.artifacts import sha256


def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit',default='F4_agreement_relations_fit_v1_20260905');parser.add_argument('--run',default='F4_agreement_relations_causal_v1_20260905');parser.add_argument('--source',default='F4_agreement_task_contrast_v1_20260905');args=parser.parse_args()
    source=ROOT/'runs'/args.source;fit=ROOT/'runs'/args.fit;run=ROOT/'runs'/args.run
    if (run/'relation_summary.json').exists():raise FileExistsError('Immutable summary output exists')
    allrows=rows(run/'metrics.raw.jsonl');old=rows(source/'metrics.raw.jsonl');base={r['id']:r for r in allrows if r['method']=='baseline'}
    anchors={(r['method'],r.get('axis'),r['id']):{k:v for k,v in r.items() if k!='run_id'} for r in old}
    for r in allrows:
        np.testing.assert_allclose(margins(np.array([r['logprobs']]),[r])[0],r['margins'],rtol=0,atol=1e-12)
        if r['method']!='baseline':np.testing.assert_allclose(np.array(base[r['id']]['margins'])-r['margins'],r['margin_loss'],rtol=0,atol=1e-12)
        if 'target_seed' not in r:assert anchors[r['method'],r.get('axis'),r['id']]=={k:v for k,v in r.items() if k!='run_id'}
    predicted=json.loads((run/'predictions_before_target_forward.json').read_text())['rows']
    pred={(r['method'],r['axis'],r['sample_id']):np.array(r['predicted_margin_loss']) for r in predicted}
    taskcfg=json.loads((run/'config.resolved.json').read_text());tasks=json.loads((run/'tokenized_development.json').read_text())['rows']
    native_reference=taskcfg.get('compiled_reference_method')=='source_native_group'
    panel=np.array([i for i,r in enumerate(tasks) if r['template'] in taskcfg['panel_templates']]);ids={r['id']:i for i,r in enumerate(tasks)}
    cache=np.load(ROOT/taskcfg['replay_activations_path'],allow_pickle=False);factors=np.load(fit/'relation_factors.npz',allow_pickle=False)
    bank=np.load(fit/'compiled_natural_deltas.npz',allow_pickle=False);b=factors['basis'];source_y=np.load(source/'source_task_direction.npz',allow_pickle=False)['source_contributions']
    table=[];supports=[]
    for method in sorted({r['method'] for r in allrows if 'target_seed' in r}):
        for axis in ('subject','attractor'):
            rr=[r for r in allrows if r['method']==method and r['axis']==axis];assert len(rr)==8
            observed=np.array([r['margin_loss'] for r in rr]);expected=np.array([pred[method,axis,r['id']] for r in rr])
            denom=np.sum(expected**2,axis=0);error=np.sum((observed-expected)**2,axis=0)
            donor=swap_indices(tasks,axis);ref=bank[f'source_native_reference_{axis}'] if native_reference else ((source_y-source_y[donor])@b)[:,None]*b
            _,scale=capped(ref,cache['hidden'],taskcfg['maximum_source_hook_fraction'])
            natural=bank[f'{method}_{axis}'];delta=natural*scale[:,None]
            for r in rr:
                i=ids[r['id']];np.testing.assert_allclose(r['dose_scale'],scale[i],rtol=0,atol=1e-12)
                np.testing.assert_allclose(r['hook_fraction'],np.linalg.norm(delta[i])/np.linalg.norm(cache['hidden'][i]),rtol=1e-10,atol=1e-12)
            table.append(dict(method=method,axis=axis,target_seed=rr[0]['target_seed'],n=8,
                source_mean=float(expected[:,0].mean()),observed_mean=float(observed[:,0].mean()),
                primary_normalized_error=float(error[0]/denom[0]),primary_rmse=float(np.sqrt(error[0]/8)),
                past_normalized_error=float(error[1]/denom[1]),mean_abs_tense_shift=float(np.abs(observed[:,2]).mean()),
                source_mean_abs_tense_shift=float(np.abs(expected[:,2]).mean()),
                mean_hook_fraction=float(np.mean([r['hook_fraction'] for r in rr])),
                sign_agreement=int(np.sum(np.sign(observed[:,0])==np.sign(expected[:,0])))))
    for target in (1,3,4,5):
        for label in ('native','single','random'):
            support=factors[f'{label}_ids_{target}'];z=cache[f'codes_{target}']
            for axis in ('subject','attractor'):
                donor=swap_indices(tasks,axis);dz=z-z[donor];x=dz[:,support]
                supports.append(dict(target_seed=target,method=label,axis=axis,support_size=len(support),
                    varying_supports_any_64=int(np.sum(np.any(x!=0,axis=0))),varying_supports_any_panel=int(np.sum(np.any(x[panel]!=0,axis=0))),
                    mean_varying_count_panel=float(np.mean(np.sum(x[panel]!=0,axis=1))),
                    selected_code_delta_energy_fraction_panel=float(np.sum(x[panel]**2)/max(np.sum(dz[panel]**2),1e-30))))
    with (run/'METHOD_TABLE.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    source_table=json.loads((source/'metrics.summary.json').read_text())['method_summary']
    historical=rows(ROOT/'runs/F4_agreement_source_remaining_v1_20260905/metrics.raw.jsonl')
    for method,atom in [('raw_hook_swap',None),('source_query',1655)]:
        for axis in ('subject','attractor'):
            rr=[r for r in historical if r['method']==method and r['axis']==axis and r.get('source_atom')==atom]
            if method=='source_query':rr=[r for r in rr if r['source_seed']==2]
            assert len(rr)==64
            source_table.append(dict(method=method,source_atom=atom,axis=axis,mean_margin_loss=np.mean([r['margin_loss'] for r in rr],axis=0).tolist(),
                mean_hook_fraction=float(np.mean([r['hook_fraction'] for r in rr])),historical_diagnostic=True))
    evidence=[entry(p,'CCAD observed artifact','evidence') for p in [source/'metrics.raw.jsonl',run/'metrics.raw.jsonl',fit/'metrics.raw.jsonl',fit/'relation_factors.npz',run/'predictions_before_target_forward.json',ROOT/'runs/F4_agreement_source_remaining_v1_20260905/metrics.raw.jsonl']]
    summary=dict(source_table=source_table,method_table=table,support_diagnostics=supports,inputs=evidence,
        checks=dict(source_anchor_rows_replayed=len(old),prediction_rows_compared=len(pred),raw_margin_rows_recomputed=len(allrows),all_candidate_source_scales_replayed=True),
        statistics='2 lexicalized templates;8 dependent number conditions;4targets share same source. No independent-seed p-values or unseen-input claim.',
        generator_script_sha256=sha256(Path(__file__)))
    write(run/'relation_summary.json',summary)
    lines=['# 任务方向与跨seed组成操作：开发结果','',
        '本轮新source方向有明确任务作用；小面板上FCC transport保留部分作用但弱于raw。当前全局discovery选出的native64/单项没有保留该作用，不能宣称核心解释性贡献已经成立。所有旧结果保留。','',
        '## Source方向（64开发输入，16词汇化模板）','',
        '|操作|主语margin loss均值|实际mean hook fraction|','|---|---:|---:|']
    for r in source_table:
        if r['axis']=='subject':lines.append(f"|{r['method']} {r.get('source_atom') or ''}|{r['mean_margin_loss'][0]:.6f}|{r['mean_hook_fraction']:.6f}|")
    lines+=['','新方向16/16模板均值正、51/64个体正；干扰交换绝对主语作用.091547，不能称完全选择性。新方向在本64条上构建，不是验证集；各操作剂量不同，20倍旧query均值差不是等预算提升。','',
        '## 跨seed实际作用（固定首尾2模板，8输入）','',
        '主要误差=Σ(candidate margin loss−source margin loss)^2 / Σ(source margin loss)^2；零操作为1。下表是subject轴，另一轴和辅助端点全部见METHOD_TABLE.csv。参数由512独立paired discovery状态的256全局差分拟合，未使用target任务code/标签/行为拟合。','',
        '|操作|平均作用|相对source平方误差|RMSE(nats)|实际mean hook fraction|','|---|---:|---:|---:|---:|']
    for r in table:
        if r['axis']=='subject':lines.append(f"|{r['method']}|{r['observed_mean']:.6f}|{r['primary_normalized_error']:.6f}|{r['primary_rmse']:.6f}|{r['mean_hook_fraction']:.6f}|")
    lines+=['','所有候选共用source投影的scale，不单独缩放到获胜剂量，也未声称等能量比较。raw只是同source basis的拟合transport，不是全hook能力上界。native为有符号加权donor差分，非native删除或纯非负插值；本次clip未必被触发，见fit raw。','',
        '## 支持使用诊断','',
        '|target|native64在64条subject差分中变化的成员|8条面板平均变化成员|面板所占code差分能量|','|---|---:|---:|---:|']
    for r in supports:
        if r['method']=='native' and r['axis']=='subject':lines.append(f"|{r['target_seed']}|{r['varying_supports_any_64']}|{r['mean_varying_count_panel']:.2f}|{r['selected_code_delta_energy_fraction_panel']:.6f}|")
    lines+=['','这些是code坐标诊断，不是hook作用份额或跨seed不变量。支持低参与和全局语料→任务消费者失配是待区分解释，不能仅凭本表归因TopK或排除真实关系。','',
        '## 接续决策','',
        '保留本次source任务方向，优先让旧的source-conditioned/操作匹配拟合路线服务任务消费者：从独立discovery按source任务状态构造邻域/差分，与现有全局差分共用样本数、正则和source参考，比较full FCC/raw及native支持的作用。最终query与邻域规则只用source信息；target端点只作开发反馈。若可用paired任务邻域不足，明确报告支持范围，再做一次门前候选→原k128执行的有针对性分支，不扩全query/多维阈值网格。','',
        '本轮的272项target预测在target前向前保存，但复制的是同开发输入已经测过的source响应，尚不是新输入的语义组合预测。reserved64未tokenize/编码/前向；无audit、重训或环境改动。五同配置SAE资产被使用，不把共享source的4方向当4独立重复。','',
        '复核：320source锚点逐值重放，592逐行margin重算，272项预测/共同scale核对。该复核只验证计算与身份，不是独立科学评审。','']
    if args.run!='F4_agreement_relations_causal_v1_20260905' and not native_reference:
        previous=json.loads((ROOT/'runs/F4_agreement_relations_causal_v1_20260905/relation_summary.json').read_text())
        before={(r['method'],r['axis']):r for r in previous['method_table']}
        comparison=[dict(method=r['method'],axis=r['axis'],old_error=before[r['method'],r['axis']]['primary_normalized_error'],new_error=r['primary_normalized_error'],old_mean=before[r['method'],r['axis']]['observed_mean'],new_mean=r['observed_mean']) for r in table]
        write(run/'global_comparison.json',dict(rows=comparison,old_summary_sha256=sha256(ROOT/'runs/F4_agreement_relations_causal_v1_20260905/relation_summary.json')))
        lines=['# Source邻域配对：与原全局拟合的直接比较','','固定原source方向、512拟合状态/256差分、ridge、native64、8个开发输入和共同source剂量缩放。新增4096行source-only候选检索，未匹配原全局方法的检索预算；所有方法共享新配对。target任务code/行为未参与选样或拟合。邻域不保证语法反事实。','','|方法|全局误差|邻域误差|全局平均作用|邻域平均作用|','|---|---:|---:|---:|---:|']
        for r in comparison:
            if r['axis']=='subject':lines.append(f"|{r['method']}|{r['old_error']:.6f}|{r['new_error']:.6f}|{r['old_mean']:.6f}|{r['new_mean']:.6f}|")
        lines+=['','主要误差=Σ(candidate−source margin loss)^2/Σ(source margin loss)^2；零操作为1。全部两轴、辅助端点和实际剂量见METHOD_TABLE.csv；支持参与情况见relation_summary.json。不同方法不是等能量操作。','','320source锚点逐值重放，592原始margin和272预测及共同scale重算；仅为计算重放，不是独立科学审查。预测复制同开发输入source响应，不是新输入泛化。四方向共享source，8条共享两词汇模板，不计算伪独立p值。reserved/audit仍未使用。','']
    if native_reference:
        lines=['# 冻结source原生教师：跨seed开发对照','','Source ids/g冻结，teacher是实际native donor差分，不投影回原b。独立discovery512行/256差分拟合；FCC与raw共用rank64 source decoder span/ridge。target native64/single/random使用实际vector teacher。不是与旧rank1结果的等复杂度比较。','','|方法|主语平均作用|相对source平方误差|past误差|主语dose|','|---|---:|---:|---:|---:|']
        for r in table:
            if r['axis']=='subject':lines.append(f"|{r['method']}|{r['observed_mean']:.6f}|{r['primary_normalized_error']:.6f}|{r['past_normalized_error']:.6f}|{r['mean_hook_fraction']:.6f}|")
        lines+=['','误差=Σ(candidate−source margin loss)^2/Σ(source margin loss)^2；零操作为1。两axis与时态连带影响、剂量、native支持参与见METHOD_TABLE.csv和relation_summary.json。所有候选共用source-native cap的scale，不是候选各自等剂量放大。','','参数拟合未使用target任务code/梯度/标签/端点；source primary梯度曾用于source组开发，现不重选。四target共享一个source，8条输入来自两个开发词汇模板，不作独立seed统计。事前target预测使用同开发输入已观测source效应，不是未见输入预测。reserved/audit未使用。','','448source锚点逐值重放，720原始margin与272预测/实际共同scale核对；仅为计算复核，不是独立科学审查。原投影/global/邻域结果原样保留。','']
    (run/'RESULTS_FOR_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(checks=summary['checks'],rows=len(allrows),methods=len(table)//2,report=str(run/'RESULTS_FOR_REVIEW.md'),supports=[r for r in supports if r['method']=='native' and r['axis']=='subject']),indent=2))


if __name__=='__main__':main()
