"""Align every frozen request to actual same-input controls; no new model/data reads."""
from __future__ import annotations
import csv
import json
import math
import platform
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ccad.artifacts import sha256, validate_run_directory
from run_r011s1_raw_hook_asset import aggregate, entry, write_json as write
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

METHODS = ['target', 'single_atom_dynamic', 'raw', 'global_rows', 'uot_default',
           'uot_discovery_tuned', 'wrong_query_matched_energy',
           'native_top16_difference_matched', 'readout_top16', 'readout_tail16',
           'readout_random16', 'global_hungarian_local_operation', 'decoder_span_operation']
METRICS = ['normalized_kl_error', 'normalized_nll_delta_squared_error']
SCOPES = ['intervention_positions', 'same_document_downstream']

def identity(r):
    return tuple(r[k] for k in ('source_seed','source_atom','condition','sequence',
                                'donor_sequence','target_seed','method'))

def source_anchor(r, scope):
    p = r['probability_endpoints'][scope]
    return {k:p[k] for k in ('source_nll_deltas','source_to_baseline_kl','positions','observed_next_token_ids')}

def med(values):
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None

def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def figure(run, requests, rows, *, methods=None, labels=None, subtitle=None, footer_lines=None,
           include_rejected=False, alt_text=None):
    image = Image.new('RGB',(2400,2200),'white'); d = ImageDraw.Draw(image)
    points=[]
    def text(x,y,s,size=25,anchor=None,bold=False):
        d.text((x,y),s,font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size),fill='#202020',anchor=anchor)
    colors=['#0072B2','#D55E00','#111111']; methods=methods or ['target','single_atom_dynamic','raw']
    vals=[r[m] for r in rows if r['method'] in methods for m in METRICS if r[m] is not None]
    assert vals and all(v>0 and math.isfinite(v) for v in vals)
    low=min(-3,math.floor(math.log10(min(vals)))); high=max(1,math.ceil(math.log10(max(vals))))
    def marker(x,y,k,small=False):
        r=3 if small else 7; color=colors[k]
        if k==0:d.ellipse((x-r,y-r,x+r,y+r),outline=color,width=1 if small else 3)
        elif k==1:d.rectangle((x-r,y-r,x+r,y+r),outline=color,width=1 if small else 3)
        else:d.polygon([(x,y-r-1),(x-r-1,y+r),(x+r+1,y+r)],outline=color,width=1 if small else 3)
    text(1200,30,'Which source-defined operations survive a change of SAE seed?',43,'ma',True)
    text(1200,89,subtitle or 'Existing-data development panel: all 32 requests / 16 queries; 11 selected cases; no new LM forwards',28,'ma')
    for k,label in enumerate(labels or ['FCC full','Best dynamic single atom','Raw hook']):
        xx=700+k*470;marker(xx,150,k);text(xx+20,150,label,28,'lm')
    text(35,207,'Panel / seed:atom / sign',26,bold=True)
    text(420,207,'Source rule status',26,bold=True)
    for j,title in enumerate(['KL error / source effect','NLL-change squared error / source effect']):
        left=880+j*740;right=left+640
        text((left+right)/2,205,title,26,'ma',True)
        for exponent in range(low,high+1):
            x=left+(exponent-low)/(high-low)*(right-left)
            d.line((x,263,x,1940),fill='#bbbbbb' if exponent==0 else '#eeeeee',width=2 if exponent==0 else 1)
            text(x,248,f'1e{exponent}',22,'mm')
        for i,q in enumerate(requests):
            y=292+i*51
            if j==0:
                text(35,y,f"{q['panel'][0].upper()}  {q['source_seed']}:{q['source_atom']}  {'+' if q['condition']=='positive' else '-'}",26,'lm')
                label=q['status'].replace('SOURCE_RULE_REJECTED','source rule: '+','.join(q['failed_source_rules']))
                text(420,y,label,22,'lm')
            group=[r for r in rows if r['request_id']==q['request_id'] and r['scope']==SCOPES[0]]
            if not q['selected'] and not (include_rejected and q['entry']):
                text((left+right)/2,y,'not measured (not zero)',23,'mm');continue
            for k,method in enumerate(methods):
                values=[r[METRICS[j]] for r in group if r['method']==method]
                for idx,v in enumerate(values):
                    if v is None:continue
                    x=left+(math.log10(v)-low)/(high-low)*(right-left); yy=y-14+idx*3+k*5
                    marker(x,yy,k,True)
                    points.append(dict(request_id=q['request_id'],metric=METRICS[j],method=method,kind='dependent_target',target_index=idx,value=v))
                v=med(values); x=left+(math.log10(v)-low)/(high-low)*(right-left)
                marker(x,y+7+k*4,k)
                points.append(dict(request_id=q['request_id'],metric=METRICS[j],method=method,kind='four_target_median',value=v))
    footers=footer_lines or [
        'O = original, E = expanded. Rule labels: weak = natural hook fraction < 0.1; concentrated = largest atom energy share > 0.5.',
        'Small symbols: four dependent targets; large symbols: their median. Log axes identical. 0 = exact source effect; 1 = no intervention.',
        'Best atom is a discovery-selected source-aligned scalar readout, NOT PW-MCC. Global Hungarian local execution and span operation: unmeasured.',
        'Selected-only effects do not estimate performance on all 32 requests. Same queries/documents/seeds recur; no independent-replicate CI.']
    for i,line in enumerate(footers):text(35,1975+44*i,line,24)
    image.save(run/'common_panel.png',dpi=(300,300))
    write(run/'figure_points.json',points)
    return dict(size_pixels=list(image.size),mode=image.mode,nominal_dpi=300,log10_limits=[low,high],points=len(points),
                publisher='internal provisional manuscript; submission requirements not asserted',
                alt=alt_text or 'All 32 source-only requests are retained. Eight have no class-compatible donor and thirteen fail the source rule. The eleven measured cases show FCC, best atom and raw errors on the identical interventions; their relative ordering varies. Missing is not zero.')

def main():
    run=ROOT/'runs/F4_common_request_panel_v2_20260906';run.mkdir(exist_ok=False)
    start=time.perf_counter(); inputs=[]; requests=[]; rows=[]; checks={}; error=None
    cfg=dict(run_id=run.name,purpose='C2 common denominator and same-operation atom/FCC/function comparison',
             parents=['F4_probability_confirmation','F4_component_probability'],panels=['original','expanded'],
             methods=METHODS,scopes=SCOPES,model_forwards=0,audit_opened=False,
             budget='Small existing JSON only; <60s CPU, no bulk arrays/model/lease',
             aggregation='per request four dependent target medians; no pooled significance',
             evidence_level='exposed_data_descriptive_reanalysis')
    write(run/'config.resolved.json',cfg)
    code=[]
    for rel in ['scripts/build_f4_common_panel.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.common.panel.v1',run_id=run.name,run_parent='F4',purpose=cfg['purpose'],milestone='M4',
         evidence_level=cfg['evidence_level'],started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),
         config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,
         audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='inherited independent mean, no arrays read',
         threshold_source_split='inherited frozen source-only rule; no thresholds selected',statistics_unit=cfg['aggregation'],
         device='CPU',seeds=[1,2,3,4,5],resource_lease='none',resource_lease_reason=cfg['budget']))
    write(run/'status.json',dict(status='RUNNING'))
    def read(rel,lines=False):
        p=ROOT/rel;inputs.append(entry(p,'existing development artifact','read_for_common_panel'))
        return [json.loads(x) for x in p.read_text().splitlines()] if lines else json.loads(p.read_text())
    try:
        freeze=read('configs/f4_probability_confirmation_corpus_v1.json')
        rule=read('configs/f4_source_applicability_dev_v1.json')['rule']
        for panel in cfg['panels']:
            prep=f'runs/F4_probability_confirmation_source_{panel}_v1_20260905'
            old=f'runs/F4_probability_confirmation_{panel}_v1_20260905'
            new=f'runs/F4_component_probability_{panel}_v1_20260905'
            choices=read(prep+'/matching.json')['choices'];selection=read(prep+'/selection.json')['queries']
            assert [[q['source_seed'],q['source_atom']] for q in selection]==next(p['fixed_queries'] for p in freeze['frozen_scope']['panels'] if p['label']==panel)
            a=read(old+'/metrics.raw.jsonl',True); b=read(new+'/metrics.raw.jsonl',True)
            ca=read(old+'/config.resolved.json');cb=read(new+'/config.resolved.json')
            for key in ['factors_sha256','token_manifest_sha256','maximum_source_hook_fraction','probability_endpoints']:
                assert ca[key]==cb[key],key
            index={identity(r):r for r in a}
            for r in b:
                if identity(r) in index:
                    for key in ['hook','endpoints','probability_endpoints','common_source_dose_scale']:
                        assert index[identity(r)][key]==r[key],key
                index[identity(r)]=r
            assert len(index)==len({identity(r) for r in a+b})
            for c in choices:
                s,atom,cond=c['source_seed'],c['source_atom'],c['condition'];e=c['entry'];scope=c.get('source_scope') or {}
                chosen=bool(e and scope['selected']);failed=[]
                if e:
                    if scope['natural_source_hook_fraction']<rule['minimum_natural_source_hook_fraction']:failed.append('weak')
                    if scope['largest_atom_energy_share']>rule['maximum_largest_source_atom_energy_share']:failed.append('concentrated')
                    assert chosen==(scope['supported'] and not failed)
                q=next(q for q in selection if (q['source_seed'],q['source_atom'])==(s,atom))
                status='SELECTED' if chosen else 'SOURCE_RULE_REJECTED' if e else c['matching_status']
                req=dict(request_id=f'{panel}:{s}:{atom}:{cond}',panel=panel,source_seed=s,source_atom=atom,condition=cond,
                         stratum=q['stratum'],targets=q['targets'],selected=chosen,status=status,failed_source_rules=failed,
                         natural_hook_fraction=scope.get('natural_source_hook_fraction'),largest_atom_energy_share=scope.get('largest_atom_energy_share'),
                         entry=e,matching_status=c['matching_status'],matching_path=prep+'/matching.json')
                requests.append(req)
                group=[r for r in index.values() if (r['source_seed'],r['source_atom'],r['condition'])==(s,atom,cond)]
                assert len(group)==(44 if chosen else 0)
                for r in group:
                    for key in ['sequence','donor_sequence','intervention_positions','donor_positions','document_ids','donor_document_ids']:
                        assert r[key]==e[key],key
                    for sc in SCOPES:assert source_anchor(r,sc)==source_anchor(group[0],sc)
                    assert r['common_source_dose_scale']==group[0]['common_source_dose_scale']
                for t in q['targets']:
                    for method in METHODS:
                        found=[r for r in group if r['target_seed']==t and r['method']==method]; assert len(found)<=1
                        r=found[0] if found else None
                        for sc in SCOPES:
                            p=r['probability_endpoints'][sc] if r else {}
                            rows.append(dict(request_id=req['request_id'],panel=panel,source_seed=s,source_atom=atom,condition=cond,
                                target_seed=t,method=method,scope=sc,status='MEASURED' if r else 'BASELINE_NOT_RUN' if chosen else status,
                                sequence=e['sequence'] if e else None,donor_sequence=e['donor_sequence'] if e else None,
                                intervention_positions=e['intervention_positions'] if e else [],donor_positions=e['donor_positions'] if e else [],
                                source_kl_mean=p.get('source_kl_mean'),source_nll_delta_rms=p.get('source_nll_delta_rms'),
                                normalized_kl_error=p.get(METRICS[0]),normalized_nll_delta_squared_error=p.get(METRICS[1]),
                                hook_error=r['hook']['normalized_error'] if r else None,
                                artifact=(new if method not in ['global_rows','raw'] else old)+'/metrics.raw.jsonl' if r else None))
            checks[panel]=dict(requests=len(choices),unique_method_rows=len(index),same_input_source_anchor_and_dose=True)
        assert len(requests)==32 and len({r['request_id'] for r in requests})==32
        assert len(rows)==32*4*len(METHODS)*2
        comparisons=[]
        for q in requests:
            if not q['selected']:continue
            p=[r for r in rows if r['request_id']==q['request_id'] and r['scope']==SCOPES[0]]
            vals={method:{metric:med([r[metric] for r in p if r['method']==method]) for metric in METRICS} for method in METHODS}
            comparisons.append(dict(request_id=q['request_id'],medians=vals,
                fcc_better_atom={metric:vals['target'][metric]<vals['single_atom_dynamic'][metric] for metric in METRICS},
                relative_error_reduction_vs_atom={metric:1-vals['target'][metric]/vals['single_atom_dynamic'][metric] for metric in METRICS}))
        summary=dict(coverage=dict(Counter(q['status'] for q in requests)),checks=checks,requests=32,queries=16,
             selected_queries=len({(q['source_seed'],q['source_atom']) for q in requests if q['selected']}),
             source_seeds_all=sorted({q['source_seed'] for q in requests}),source_seeds_selected=sorted({q['source_seed'] for q in requests if q['selected']}),
             comparisons=comparisons,model_forwards=0,rows=len(rows),measured_rows=sum(r['status']=='MEASURED' for r in rows),
             fcc_wins_over_atom={metric:sum(c['fcc_better_atom'][metric] for c in comparisons) for metric in METRICS},
             limitations=['global PW-MCC and local execution differ; latter absent','decoder span operation absent','selected-only outcomes do not identify all-request performance','reanalysis is development, not new confirmation'])
        csv_write(run/'REQUESTS.csv',requests);csv_write(run/'COMMON_METHOD_ROWS.csv',rows)
        write(run/'requests.json',requests)
        (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
        summary['metrics_raw_sha256']=sha256(run/'metrics.raw.jsonl')
        summary['figure']=figure(run,requests,rows)
        summary['wall_seconds']=time.perf_counter()-start
        write(run/'metrics.summary.json',summary)
        lines=['# Common request panel: descriptive development evidence','',
               'All 32 requests retained; all values joined by exact query, recipient, donor, positions, source anchors and common dose. No new model forwards. No sealed audit read.','',
               '|Request|Status|FCC KL/NLL|Best dynamic atom KL/NLL|Raw KL/NLL|','|---|---|---|---|---|']
        for q in requests:
            c=next((c for c in comparisons if c['request_id']==q['request_id']),None)
            fields=[' / '.join(f'{c["medians"][m][k]:.6g}' for k in METRICS) if c else 'NA (not measured)' for m in ['target','single_atom_dynamic','raw']]
            lines.append('|'+ '|'.join([q['request_id'],q['status']]+fields)+'|')
        lines+=['','Best dynamic atom is a source-aligned scalar regression selected on discovery conditional variation; NOT a global one-to-one matching or native ablation.',
                'Each number is the median of four dependent targets. Direction/condition counts are not independent replicates. Other nine controls or explicit baseline gaps are in COMMON_METHOD_ROWS.csv.',
                'Read/aggregate stage complete. Selection of the next experiment must be recorded separately, after inspecting this panel.']
        (run/'COMMON_PANEL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    except Exception:
        error=traceback.format_exc()
    write(run/'inputs.json',dict(inputs=inputs))
    write(run/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),pillow=pillow_version,
          cuda='not_applicable',gpu='not_applicable',pytorch='not_imported',transformers='not_imported',sae='not_imported'))
    (run/'stderr.log').write_text(error or '')
    (run/'stdout.log').write_text(json.dumps(dict(requests=len(requests),rows=len(rows),model_forwards=0,error=error))+'\n')
    write(run/'status.json',dict(status='FAIL' if error else 'PASS',error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    result=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=result.ok,errors=list(result.errors)))
    print(json.dumps(dict(run=run.name,error=error,contract_ok=result.ok,checks=checks)))
    return 1 if error or not result.ok else 0

if __name__=='__main__':raise SystemExit(main())
