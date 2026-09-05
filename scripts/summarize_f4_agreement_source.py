"""Combine source-only SVA screens, preserving the entire 40-query attempt pool."""
import csv
import json
from pathlib import Path
import numpy as np
from run_f4_agreement_source import ROOT,margins,swap_indices
from run_r011s1_raw_hook_asset import entry,write_json as write
from ccad.artifacts import sha256


def read_rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def main():
    first=ROOT/'runs/F4_agreement_source_v2_20260905'
    second=ROOT/'runs/F4_agreement_source_remaining_v1_20260905'
    output=second/'agreement_summary.json'
    if output.exists():raise FileExistsError(output)
    all_rows=[];inputs=[];anchors={};total_forwards=0;total_seconds=0.;numeric_seconds=0.
    for run in (first,second):
        assert json.loads((run/'status.json').read_text())['status']=='PASS'
        summary=json.loads((run/'metrics.summary.json').read_text())
        assert sha256(run/'metrics.raw.jsonl')==summary['metrics_raw_sha256']
        total_forwards+=summary['model_forwards'];total_seconds+=summary['wall_seconds'];numeric_seconds+=summary['numeric_seconds']
        rows=read_rows(run/'metrics.raw.jsonl');base={r['id']:r for r in rows if r['method']=='baseline'}
        for r in rows:
            np.testing.assert_allclose(margins(np.array([r['logprobs']]),[r])[0],r['margins'],rtol=0,atol=1e-12)
            if r['method']!='baseline':np.testing.assert_allclose(np.array(base[r['id']]['margins'])-r['margins'],r['margin_loss'],rtol=0,atol=1e-12)
            if r['method']!='source_query':
                key=(r['method'],r.get('axis'),r['id']);filtered={k:v for k,v in r.items() if k!='run_id'}
                if run==first:anchors[key]=filtered;all_rows.append(r)
                else:assert anchors[key]==filtered
            else:all_rows.append(r)
        for name in ('metrics.raw.jsonl','config.resolved.json','tokenized_development.json','padding_checks.json'):
            inputs.append(entry(run/name,'CCAD task source screen',name))
    queries=sorted({(r['source_seed'],r['source_atom']) for r in all_rows if r['method']=='source_query'})
    assert len(queries)==40 and len(all_rows)==5312 and len(anchors)==192
    rankings=[]
    for s,a in queries:
        axes={axis:[r for r in all_rows if r['method']=='source_query' and r['source_seed']==s and r['source_atom']==a and r['axis']==axis] for axis in ('subject','attractor')}
        assert all(len(v)==64 for v in axes.values())
        subject=np.array([r['margin_loss'][0] for r in axes['subject']]);attractor=np.array([r['margin_loss'][0] for r in axes['attractor']])
        templates=sorted({r['template'] for r in axes['subject']})
        template_means=[np.mean([r['margin_loss'][0] for r in axes['subject'] if r['template']==t]) for t in templates]
        rankings.append(dict(source_seed=s,source_atom=a,subject_mean_loss=float(subject.mean()),subject_median_loss=float(np.median(subject)),
            subject_mean_abs_loss=float(np.abs(subject).mean()),attractor_mean_abs_loss=float(np.abs(attractor).mean()),
            positive_inputs=int((subject>0).sum()),nonzero_input_deltas=sum(r['natural_norm']>0 for r in axes['subject']),
            positive_template_means=int((np.array(template_means)>0).sum()),templates=16,
            mean_hook_fraction=float(np.mean([r['hook_fraction'] for r in axes['subject']])),
            maximum_subject_loss=float(subject.max()),minimum_subject_loss=float(subject.min())))
    rankings.sort(key=lambda r:(-r['subject_mean_loss'],r['source_atom'],r['source_seed']))
    base=[r for r in all_rows if r['method']=='baseline']
    controls={axis:dict(mean_loss=float(np.mean([r['margin_loss'][0] for r in all_rows if r['method']=='raw_hook_swap' and r['axis']==axis])),
                       mean_abs_loss=float(np.mean([abs(r['margin_loss'][0]) for r in all_rows if r['method']=='raw_hook_swap' and r['axis']==axis]))) for axis in ('subject','attractor')}
    with (second/'ALL_SOURCE_QUERIES.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rankings[0]));writer.writeheader();writer.writerows(rankings)
    lines=['# All 40 source-only SVA query attempts','','Positive subject loss = move away from recipient-correct verb after subject swap. Attractor absolute loss is a separate sensitivity diagnostic, not an equal-energy control. No target endpoints or reserved inputs used.','','| Source query | Subject mean loss (nats) | Subject median | Attractor mean absolute loss | Positive templates /16 | Nonzero input deltas /64 |','|---|---:|---:|---:|---:|---:|']
    for r in rankings:lines.append(f"| {r['source_seed']}:{r['source_atom']} | {r['subject_mean_loss']:.6f} | {r['subject_median_loss']:.6f} | {r['attractor_mean_abs_loss']:.6f} | {r['positive_template_means']} | {r['nonzero_input_deltas']} |")
    (second/'ALL_SOURCE_QUERIES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=dict(inputs=inputs,source_ranking=rankings,shortlist_by_frozen_rule=rankings[:2],controls=controls,
        baseline_correct=sum(r['margins'][0]>0 for r in base),baseline_total=64,unique_rows=len(all_rows),exact_replayed_control_rows=192,
        source_observations=5120,completed_forwards=total_forwards,completed_wall_seconds=total_seconds,completed_numeric_seconds=numeric_seconds,
        failed_preflight_forwards=18,failed_preflight_wall_seconds=json.loads((ROOT/'runs/F4_agreement_source_v1_20260905/metrics.summary.json').read_text())['wall_seconds'],
        reserved_forwarded=False,target_endpoints_used=False,scope='Source-only synthetic task development. Ranking is not proof of strong/selective correspondence.',
        generator_script_path=Path(__file__).relative_to(ROOT).as_posix(),generator_script_sha256=sha256(Path(__file__)))
    write(output,report)
    print(json.dumps(dict(first5=rankings[:5],controls=controls,forwards=total_forwards,seconds=total_seconds,numeric_seconds=numeric_seconds),indent=2))


if __name__=='__main__':main()
