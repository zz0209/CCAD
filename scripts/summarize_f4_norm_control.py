"""Matched-norm outcomes and template-level uncertainty, no seed pseudoreplication."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from run_f4_agreement_source import ROOT,margins
from run_r011s1_raw_hook_asset import entry,write_json as write
from ccad.artifacts import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='F4_task_norm_matched_causal_v1_20260905');parser.add_argument('--parent',default='F4_task_paired_relations_full_dev_v1_20260905');args=parser.parse_args()
    run=ROOT/'runs'/args.run;out=run/'NORM_COMPARISON.md'
    if out.exists():raise FileExistsError(out)
    rr=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()]
    cfg=json.loads((run/'config.resolved.json').read_text());reserved=cfg['split']=='reserved'
    allrows={(r['method'],r.get('axis'),r['id']):r for r in rr};base={r['id']:r for r in rr if r['method']=='baseline'}
    for r in rr:
        np.testing.assert_allclose(margins(np.array([r['logprobs']]),[r])[0],r['margins'],rtol=0,atol=1e-12)
        if r['method']!='baseline':np.testing.assert_allclose(np.array(base[r['id']]['margins'])-r['margins'],r['margin_loss'],rtol=0,atol=1e-12)
    exact=0
    if not reserved:
        old=[json.loads(x) for x in (ROOT/'runs'/args.parent/'metrics.raw.jsonl').read_text().splitlines()]
        for r in old:
            key=r['method'],r.get('axis'),r['id']
            if key in allrows:
                assert {k:v for k,v in r.items() if k!='run_id'}=={k:v for k,v in allrows[key].items() if k!='run_id'};exact+=1
    observations=[];comparison=[];rawpoints=[];rng=np.random.default_rng(9051916)
    for seed in (1,3,4,5):
        axis_data={}
        for axis in ('subject','attractor'):
            a=sorted([r for r in rr if r['method']==f'fcc_codes_target{seed}' and r['axis']==axis],key=lambda r:r['id'])
            b=[allrows[f'raw_matched_target{seed}',axis,r['id']] for r in a]
            source=[allrows['source_native_group',axis,r['id']] for r in a]
            norms=np.max(np.abs(np.array([r['hook_fraction'] for r in a])-np.array([r['hook_fraction'] for r in b])))
            if norms>1e-12:raise ValueError('Actual applied doses differ')
            s=np.array([r['margin_loss'] for r in source]);aa=np.array([r['margin_loss'] for r in a]);bb=np.array([r['margin_loss'] for r in b])
            axis_data[axis]=(a,aa,bb,s)
            for label,values in [('codeFCC',aa),('matchedraw',bb)]:
                observations.append(dict(target_seed=seed,axis=axis,method=label,n=len(a),templates=len({r['template'] for r in a}),
                    primary_error=float(np.sum((values[:,0]-s[:,0])**2)/np.sum(s[:,0]**2)),
                    past_error=float(np.sum((values[:,1]-s[:,1])**2)/np.sum(s[:,1]**2)),
                    mean_primary=float(values[:,0].mean()),mean_abs_primary=float(np.abs(values[:,0]).mean()),mean_abs_tense=float(np.abs(values[:,2]).mean()),
                    mean_hook_fraction=float(np.mean([r['hook_fraction'] for r in a])),maximum_pair_dose_difference=float(norms)))
        a,aa,bb,s=axis_data['subject'];templates=sorted({r['template'] for r in a});index={t:[i for i,r in enumerate(a) if r['template']==t] for t in templates}
        for endpoint,j in [('primary',0),('past',1)]:
            denom=np.array([np.sum(s[index[t],j]**2) for t in templates]);delta=np.array([np.sum((bb[index[t],j]-s[index[t],j])**2-(aa[index[t],j]-s[index[t],j])**2) for t in templates])
            draw=rng.integers(0,len(templates),(5000,len(templates)));boot=delta[draw].sum(axis=1)/np.maximum(denom[draw].sum(axis=1),1e-30)
            comparison.append(dict(target_seed=seed,endpoint=endpoint,raw_minus_fcc_error=float(delta.sum()/denom.sum()),
                bootstrap_95_pointwise=np.quantile(boot,[.025,.975]).tolist(),better_templates=int(np.sum(delta>0)),templates=len(templates)))
            for t,d,n in zip(templates,delta,denom):rawpoints.append(dict(target_seed=seed,endpoint=endpoint,template=t,error_difference_numerator=float(d),source_energy_denominator=float(n)))
    for name,data in [('NORM_TABLE.csv',observations),('TEMPLATE_EFFECTS.csv',rawpoints)]:
        with (run/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(run/'norm_comparison.json',dict(rows=observations,comparisons=comparison,replayed_parent_rows=exact,raw_margins_recomputed=len(rr),
        bootstrap=dict(draws=5000,seed=9051916,unit='whole lexical template; four number conditions kept together',interval='95percent percentile pointwise, not simultaneous',shared_source_directions_not_independent=True),
        evidence=[entry(run/'metrics.raw.jsonl','CCAD actual forward','raw')],generator_script_sha256=sha256(Path(__file__)),numpy_version=np.__version__))
    lines=['# Code FCC与等范数raw：'+('预留词汇/介词确认' if reserved else '开发对照'),'',
        '同输入、同source参考、同hook位置；逐输入/axis实际向量范数相等。Matched raw借用了FCC的范数，只检验方向差异，不是独立训练的新基线。source教师与映射全部冻结；每个方向用同64输入，四方向共享source，不能当四个独立实验。','',
        '|target|FCC主误差|matched raw主误差|FCC past误差|matched raw past误差|FCC干扰时态abs|matched raw干扰时态abs|','|---|---:|---:|---:|---:|---:|---:|']
    ix={(r['target_seed'],r['axis'],r['method']):r for r in observations}
    for seed in (1,3,4,5):
        a=ix[seed,'subject','codeFCC'];b=ix[seed,'subject','matchedraw'];c=ix[seed,'attractor','codeFCC'];d=ix[seed,'attractor','matchedraw']
        lines.append(f"|{seed}|{a['primary_error']:.6f}|{b['primary_error']:.6f}|{a['past_error']:.6f}|{b['past_error']:.6f}|{c['mean_abs_tense']:.6f}|{d['mean_abs_tense']:.6f}|")
    lines+=['','误差=Σ(candidate−source作用)^2/Σ(source作用)^2，零操作为1。干扰主语abs及全部剂量见NORM_TABLE.csv，不省略不利端点。原始未缩放raw保留在METHOD_TABLE.csv或metrics.raw.jsonl；它在开发主指标占优，但在本次reserved不占优，必须按各自数据集分别报告。','',
        '|target|端点|matched raw误差−FCC误差|模板bootstrap 95% pointwise区间|改善模板数|','|---|---|---:|---|---:|']
    for r in comparison:lines.append(f"|{r['target_seed']}|{r['endpoint']}|{r['raw_minus_fcc_error']:.6f}|[{r['bootstrap_95_pointwise'][0]:.6f},{r['bootstrap_95_pointwise'][1]:.6f}]|{r['better_templates']}/{r['templates']}|")
    lines+=['','区间按16完整模板重采样5000次，seed9051916；不是方向间独立置信区间、同时区间或普遍性证明。原始模板分子/分母保留，无剔除。',
        '', '本对照检验source-aligned signed transport，不是target native删除、唯一概念或人类语义标签证明。主语任务配对本身有source定义的监督；past未用于source梯度拟合，attractor是不同操作轴。',
        '', ('这是方法/端点/操作点冻结后的新词汇和介词输入；结构族仍相同，不能称新句法族或跨模型泛化。' if reserved else '全部64输入仍为开发。下一步在不改这些参数/操作的条件下确认原reserved新词汇/介词范围。'),
        '',f'数值检查：{len(rr)}行原始margin重算，{exact}行开发父结果逐值重放；实际每输入配对dose差<=1e-12。','']
    out.write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
