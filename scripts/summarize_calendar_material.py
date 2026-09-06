"""Descriptive aggregation of the complete authored calendar material panel."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_r011s1_raw_hook_asset import ROOT, write_json as write, entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    run = ROOT/'runs'/args.run
    out = ROOT/'artifacts'/'calendar_material_20260906'
    out.mkdir(exist_ok=True)
    load = lambda name:json.loads((run/name).read_text())
    if load('status.json')['status']!='PASS':
        raise ValueError('Run has not passed its operational checks')
    base = load('baseline_results.json')['rows']
    rows = [json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
    baseline = []
    for template in dict.fromkeys(r['template'] for r in base):
        a = [r for r in base if r['template']==template]
        baseline.append(dict(template=template, family=a[0]['family'], relation=a[0]['relation'], n=len(a),
            family_correct=sum(r['family_correct'] for r in a), full_correct=sum(r['full_correct'] for r in a),
            mean_family_mass=float(np.mean([r['family_mass'] for r in a])),
            mean_expected_probability=float(np.mean([r['expected_probability'] for r in a]))))
    keys = [('all',None)]+[('template',r['template']) for r in baseline]
    operations = []
    for scope, template in keys:
        groups = defaultdict(list)
        for r in rows:
            if template is None or r['template']==template:
                groups[r['method'],r['seed'],r['layer'],r['mode']].append(r)
        for (method,seed,layer,mode),a in groups.items():
            v = dict(scope=scope,template=template,method=method,seed=seed,layer=layer,mode=mode,n=len(a),
                donor_family_correct=sum(r['family_donor_correct'] for r in a),
                donor_full_correct=sum(r['full_donor_correct'] for r in a))
            for field in ['donor_margin_change','intervention_kl','donor_forward_kl','reference_kl',
                          'noop_reference_kl','reference_margin_change','vector_relative_squared_error']:
                vals = [r[field] for r in a if r[field] is not None]
                if vals:
                    v['median_'+field] = float(np.median(vals))
            if method=='sae_delta':
                denominator = sum(r['noop_reference_kl'] for r in a)
                v['aggregate_relative_reference_kl'] = sum(r['reference_kl'] for r in a)/denominator if denominator>0 else None
                v['aggregate_relative_margin_mse'] = sum((r['donor_margin_change']-r['reference_margin_change'])**2 for r in a)/sum(r['reference_margin_change']**2 for r in a)
                v['count_reference_kl_better_than_noop'] = sum(r['reference_kl']<r['noop_reference_kl'] for r in a)
            operations.append(v)
    # Diagnostic example fixed by source prompt order: January -> February,
    # the first pair in the month-next-plain template, without selecting results.
    example = [r for r in rows if r['template']=='month_next_plain' and r['recipient_value']=='January' and r['offset']==1]
    old = ROOT/'runs'/'F4_calendar_material_v1_20260906'
    oldbase = json.loads((old/'baseline_results.json').read_text())['rows']
    precision = dict(old_status=json.loads((old/'status.json').read_text())['status'],
        old_checks=json.loads((old/'operation_checks.json').read_text()),
        new_checks=load('operation_checks.json'),
        family_predictions_unchanged=all(a['family_prediction']==b['family_prediction'] for a,b in zip(base,oldbase)),
        full_predictions_unchanged=all(a['full_prediction']==b['full_prediction'] for a,b in zip(base,oldbase)),
        maximum_candidate_probability_difference=max(abs(x-y) for a,b in zip(base,oldbase) for x,y in zip(a['candidate_probabilities'],b['candidate_probabilities'])))
    prompts = load('prompts_and_pairs.json')['rows']
    raw = np.load(run/'raw_hooks.npz')['layer5']
    slots = np.array([r['slot'] for r in prompts])
    h = raw[np.arange(len(prompts)),slots]
    material_contrasts = []
    for seed in range(1,6):
        ar = np.load(run/f'codes_seed{seed}.npz')
        z = ar['codes'][np.arange(len(prompts)),slots].astype(float)
        y = z @ ar['decoder'].astype(float)
        for t in baseline:
            ix = [r['id'] for r in prompts if r['template']==t['template']]
            hc, yc = h[ix]-h[ix].mean(0), y[ix]-y[ix].mean(0)
            material_contrasts.append(dict(seed=seed,template=t['template'],n=len(ix),
                variable_centered_fve=float(1-np.sum((yc-hc)**2)/np.sum(hc**2)),
                raw_variation_energy=float(np.sum(hc**2)),
                definition='value-slot vectors centered within this template; decoder bias cancels',
                precision='saved float32 SAE codes/decoder recomputed in float64'))
    summary = dict(run=args.run,baseline=baseline,operations=operations,example=example,precision=precision,
        cost=load('metrics.summary.json'),environment=load('environment.json'),material=load('material_scores.json'),
        material_contrasts=material_contrasts,
        scope='descriptive dependent panel, no confidence intervals; all templates retained; no FCC fit')
    write(out/'summary.json',summary)
    paths = [run/n for n in ['config.resolved.json','metrics.summary.json','metrics.raw.jsonl',
             'baseline_results.json','operation_checks.json','prompts_and_pairs.json','material_scores.json',
             'environment.json','inputs.json','code_hashes.json']]
    write(out/'source_manifest.json',dict(files=[entry(x,'CCAD authored material experiment','source') for x in paths],
          analysis_script=entry(Path(__file__),'CCAD','analysis')))
    print(json.dumps(dict(baseline=baseline, all_operations=[v for v in operations if v['scope']=='all'],
         month_next=[v for v in operations if v['template']=='month_next_plain'],precision=precision),indent=2))


if __name__=='__main__':
    main()
