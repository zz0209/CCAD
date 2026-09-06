"""Complete the existing request denominator with fixed source-rejected outcomes."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/F4_rejected_probability_expanded_v1_20260906'
METHODS=['target','single_atom_dynamic','raw']
METRICS=['normalized_kl_error','normalized_nll_delta_squared_error']
SCOPES=['intervention_positions','same_document_downstream']


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def collect():
    inputs=[]
    def read(rel,lines=False):
        path=ROOT/rel;data=path.read_bytes()
        inputs.append(dict(path=rel,sha256=hashlib.sha256(data).hexdigest()))
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)
    requests=read('runs/F4_common_request_panel_v2_20260906/requests.json')
    if isinstance(requests,dict):requests=requests['requests']
    records={};checks={};forwards=0;wall=0
    for panel in ['original','expanded']:
        old=f'runs/F4_component_probability_{panel}_v1_20260905'
        rawold=f'runs/F4_probability_confirmation_{panel}_v1_20260905'
        new=f'runs/F4_rejected_probability_{panel}_v1_20260906'
        configs=[read(r+'/config.resolved.json') for r in [old,rawold,new]]
        fields=['factors_sha256','surface_sha256','token_manifest_sha256','maximum_source_hook_fraction','probability_endpoints']
        for field in fields:assert all(c[field]==configs[0][field] for c in configs),field
        assert read(new+'/status.json')['status']=='PASS'
        summary=read(new+'/metrics.summary.json');forwards+=summary['model_forwards'];wall+=summary['wall_seconds']
        scope_rows=read(new+'/source_scope_selection.json')['rows']
        assert len(scope_rows)==configs[-1]['expected_evaluated_cases'] and all(not r['selected'] for r in scope_rows)
        for run,methods in [(old,['target','single_atom_dynamic']),(rawold,['raw']),(new,METHODS)]:
            for row in read(run+'/metrics.raw.jsonl',True):
                if row['method'] not in methods:continue
                key=(panel,row['source_seed'],row['source_atom'],row['condition'],row['target_seed'],row['method'])
                assert key not in records,key
                records[key]=(row,run+'/metrics.raw.jsonl')
        checks[panel]='same coefficients/data/endpoints/dose; rejected labels false; no duplicate rows'
    flat=[];cases=[]
    for q in requests:
        prefix=(q['panel'],q['source_seed'],q['source_atom'],q['condition'])
        group=[r for key,(r,_) in records.items() if key[:4]==prefix]
        measured=bool(q['entry'])
        assert len(group)==(12 if measured else 0)
        if measured:
            for r in group:
                for field in ['sequence','donor_sequence','intervention_positions','donor_positions','document_ids','donor_document_ids']:
                    assert r[field]==q['entry'][field],(prefix,field)
                assert r['common_source_dose_scale']==group[0]['common_source_dose_scale']
                for scope in SCOPES:
                    for field in ['positions','source_nll_deltas','source_to_baseline_kl','observed_next_token_ids']:
                        assert r['probability_endpoints'][scope][field]==group[0]['probability_endpoints'][scope][field]
        for scope in SCOPES:
            values={}
            for method in METHODS:
                values[method]={k:[] for k in METRICS+['candidate_error_kl_mean','nll_delta_rmse']}
                for target in q['targets']:
                    item=records.get(prefix+(target,method))
                    r,path=item if item else ({},None)
                    p=r.get('probability_endpoints',{}).get(scope,{})
                    entry=dict(request_id=q['request_id'],source_seed=q['source_seed'],source_atom=q['source_atom'],
                               condition=q['condition'],target_seed=target,source_rule_status=q['status'],scope=scope,method=method,
                               status=p.get('status','NO_DONOR'),source_natural_hook_fraction=q['natural_hook_fraction'],
                               largest_atom_energy_share=q['largest_atom_energy_share'],source_kl_mean=p.get('source_kl_mean'),
                               source_nll_delta_rms=p.get('source_nll_delta_rms'),artifact=path)
                    for metric in values[method]:
                        entry[metric]=p.get(metric);values[method][metric].append(p.get(metric))
                    flat.append(entry)
            source=group[0]['probability_endpoints'][scope] if measured else {}
            cases.append(dict(request_id=q['request_id'],scope=scope,source_rule_status=q['status'],measured=measured,
                              source_natural_hook_fraction=q['natural_hook_fraction'],largest_atom_energy_share=q['largest_atom_energy_share'],
                              source_kl_mean=source.get('source_kl_mean'),source_nll_delta_rms=source.get('source_nll_delta_rms'),
                              medians={m:{k:median(v) for k,v in stats.items()} for m,stats in values.items()}))
    summaries=[]
    for scope in SCOPES:
        for label,allowed in [('source_selected',{'SELECTED'}),('source_rejected',{'SOURCE_RULE_REJECTED'}),
                              ('all_matched',{'SELECTED','SOURCE_RULE_REJECTED'})]:
            group=[c for c in cases if c['scope']==scope and c['source_rule_status'] in allowed]
            result=dict(scope=scope,group=label,cases=len(group),methods={},wins={},valid_comparisons={},
                        source_kl_median=median([c['source_kl_mean'] for c in group]),
                        source_nll_rms_median=median([c['source_nll_delta_rms'] for c in group]))
            for method in METHODS:
                result['methods'][method]={metric:median([c['medians'][method][metric] for c in group])
                                           for metric in METRICS+['candidate_error_kl_mean','nll_delta_rmse']}
            for comparator in METHODS[1:]:
                result['wins'][comparator]={};result['valid_comparisons'][comparator]={}
                for metric in METRICS:
                    pairs=[(c['medians']['target'][metric],c['medians'][comparator][metric]) for c in group]
                    pairs=[p for p in pairs if None not in p]
                    result['wins'][comparator][metric]=sum(a<b for a,b in pairs)
                    result['valid_comparisons'][comparator][metric]=len(pairs)
            summaries.append(result)
    return requests,flat,cases,dict(checks=checks,coverage=dict(Counter(q['status'] for q in requests)),
          new_model_forwards=forwards,new_consumer_wall_seconds=wall,summaries=summaries,inputs=inputs,
          unit='request-level medians over four dependent targets; repeated query, donor pairs and seeds are not independent',
          interpretation='Exposed-data development boundary; concentrated source contributions and weak source effects are confounded. No global one-to-one or new confirmation claim.')


def plot(requests,cases,flat):
    from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
    image=Image.new('RGB',(2300,2140),'white');draw=ImageDraw.Draw(image);points=[]
    def text(x,y,s,size=23,bold=False,anchor=None):
        font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
        draw.text((x,y),s,font=font,fill='#222222',anchor=anchor)
    ratios=[]
    for c in cases:
        if c['scope']!=SCOPES[0]:continue
        for metric in METRICS:
            a,b=c['medians']['target'][metric],c['medians']['single_atom_dynamic'][metric]
            if a is not None and b is not None and a>0 and b>0:ratios.append(a/b)
    for q in requests:
        for target in q['targets']:
            rr=[r for r in flat if r['request_id']==q['request_id'] and r['scope']==SCOPES[0] and r['target_seed']==target]
            for metric in METRICS:
                a=next(r[metric] for r in rr if r['method']=='target');b=next(r[metric] for r in rr if r['method']=='single_atom_dynamic')
                if a is not None and b is not None and a>0 and b>0:ratios.append(a/b)
    low=min(-1,math.floor(math.log10(min(ratios))));high=max(1,math.ceil(math.log10(max(ratios))))
    text(1150,28,'Where does distributed correspondence improve on a single atom?',36,True,'ma')
    text(1150,82,'All 32 requests: 11 previously source-selected + 13 newly measured source-rejected + 8 without donor',24,False,'ma')
    text(35,139,'Request / source seed:atom',23,True);text(385,139,'Source rule',23,True)
    text(600,139,'Largest share',23,True);text(770,139,'Source NLL RMS',23,True)
    for j,title in enumerate(['FCC / atom KL error','FCC / atom NLL-change error']):
        left=1080+j*590;right=left+520
        text((left+right)/2,139,title,25,True,'ma')
        for exp in range(low,high+1):
            x=left+(exp-low)/(high-low)*(right-left)
            draw.line((x,214,x,1930),fill='#999999' if exp==0 else '#e5e5e5',width=2 if exp==0 else 1)
            text(x,190,f'1e{exp}',20,False,'mm')
        metric=METRICS[j]
        for i,q in enumerate(requests):
            y=242+i*52;c=next(c for c in cases if c['request_id']==q['request_id'] and c['scope']==SCOPES[0])
            color='#0072B2' if q['selected'] else '#D55E00'
            if j==0:
                text(35,y,q['request_id'].replace('original:','O ').replace('expanded:','E ').replace(':positive',' +').replace(':negative',' -'),23,False,'lm')
                text(385,y,'selected' if q['selected'] else 'rejected' if q['entry'] else 'no donor',23,False,'lm')
                text(670,y,f"{q['largest_atom_energy_share']:.3f}" if q['entry'] else 'NA',22,False,'mm')
                text(870,y,f"{c['source_nll_delta_rms']:.3g}" if c['source_nll_delta_rms'] is not None else 'NA',22,False,'mm')
            if not c['measured']:
                text((left+right)/2,y,'not measured',22,False,'mm');continue
            values=[]
            for target in q['targets']:
                rr=[r for r in flat if r['request_id']==q['request_id'] and r['scope']==SCOPES[0] and r['target_seed']==target]
                a=next(r[metric] for r in rr if r['method']=='target');b=next(r[metric] for r in rr if r['method']=='single_atom_dynamic')
                values.append((target,a,b))
            values.append(('ratio_of_medians',c['medians']['target'][metric],c['medians']['single_atom_dynamic'][metric]))
            for k,(target,a,b) in enumerate(values):
                yy=y-12+k*5
                if a is None or b is None or b==0 or a==0:
                    text(right,yy,'NA' if a is None or b is None else '0/0' if a==b else 'inf' if b==0 else '0',14,False,'rm');continue
                ratio=a/b;x=left+(math.log10(ratio)-low)/(high-low)*(right-left);radius=6 if k==4 else 3
                box=(x-radius,yy-radius,x+radius,yy+radius)
                if q['selected']:draw.ellipse(box,outline=color,width=2 if k==4 else 1)
                else:draw.rectangle(box,outline=color,width=2 if k==4 else 1)
                points.append(dict(request_id=q['request_id'],metric=metric,target=target,ratio=ratio))
    text(35,1960,'Left of 1: FCC lower error. Right of 1: best dynamic atom lower error. Log10 axes share the full observed range.',24)
    text(35,2004,'Blue circles: source-selected. Orange squares: source-rejected. Small marks: four dependent targets; large: ratio of case medians.',23)
    text(35,2048,'All rejected cases have concentrated source contribution; 12/13 also have weak natural effects. Descriptive development, no independent-replicate CI.',22)
    text(35,2087,'195 new LM forwards complete fixed matched requests; eight no-donor requests remain missing. Best dynamic atom is not global PW-MCC.',23)
    image.save(OUT/'source_rule_boundary.png',dpi=(300,300))
    return dict(points=points,log10_limits=[low,high],size_pixels=list(image.size),pillow_version=pillow_version,
                medium='provisional manuscript, publisher requirements not asserted',
                transformation='paired-target ratios plus ratio of four-target medians; no smoothing; zero/undefined ratios explicitly labeled')


def main():
    requests,flat,cases,summary=collect()
    assert len(flat)==32*4*3*2
    write(OUT/'source_rule_boundary.json',summary);write(OUT/'source_rule_cases.json',cases)
    csv_write(OUT/'SOURCE_RULE_BOUNDARY_ROWS.csv',flat)
    figure=plot(requests,cases,flat);write(OUT/'source_rule_figure.json',figure)
    report=['# Source-rule boundary on the complete matched request panel','',
            'Existing-data development: 195 new model forwards fill the13 previously unmeasured matched requests. The original11 selected requests retain their exact old results. All8 no-donor requests remain missing. No new fit or query/donor selection.','',
            '|Scope / group|Cases|FCC KL / NLL error|Dynamic atom KL / NLL error|Raw KL / NLL error|FCC wins vs atom KL / NLL|',
            '|---|---:|---|---|---|---|']
    for s in summary['summaries']:
        pair=lambda m:' / '.join('NA' if s['methods'][m][k] is None else f"{s['methods'][m][k]:.6g}" for k in METRICS)
        wins=' / '.join(f"{s['wins']['single_atom_dynamic'][k]}/{s['valid_comparisons']['single_atom_dynamic'][k]}" for k in METRICS)
        report.append(f"|{s['scope']} / {s['group']}|{s['cases']}|{pair('target')}|{pair('single_atom_dynamic')}|{pair('raw')}|{wins}|")
    report+=['','Values are medians across request-level four-target medians. Wins compare the same request-level medians, not pooled directions. Shared seeds, queries and sometimes reversed donor pairs are dependent. Both absolute candidate errors and source effect sizes are retained in the CSV; null ratios follow the original frozen denominator rule.','',
             'All13 rejected requests are concentrated (largest source atom share above0.5);12 are also weak under the original natural hook fraction rule. This panel cannot separately attribute an ordering change to concentration rather than signal strength. It does not establish causal training mechanisms, global one-to-one superiority or new-document confirmation.','',
             'The plot includes all requests, both error endpoints and each dependent target. No winning directions were dropped. Original source selection is metadata, not changed to true for the new requests.','',
             'Validation: exact case/donor/position identity, source endpoint anchors and common dose checked across methods; fit/data/endpoint configuration equality verified across old and new runs; new source-selection labels remain false. These checks establish the merge scope, not an independent replication.','',
             'Provenance: source_rule_boundary.json lists all raw/config hashes; source_rule_figure.json contains every plotted point and transformation.']
    (OUT/'SOURCE_RULE_BOUNDARY.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    write(OUT/'source_rule_analysis_provenance.json',dict(script='scripts/analyze_f4_rejected_panel.py',
          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_manifest='source_rule_boundary.json',
          outputs={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in ['SOURCE_RULE_BOUNDARY.md','SOURCE_RULE_BOUNDARY_ROWS.csv','source_rule_boundary.png','source_rule_boundary.json','source_rule_cases.json','source_rule_figure.json']}))
    print(json.dumps({'coverage':summary['coverage'],'new_forwards':summary['new_model_forwards'],'new_wall_seconds':summary['new_consumer_wall_seconds'],'figure_points':len(figure['points']),'primary':[s for s in summary['summaries'] if s['scope']==SCOPES[0]]}))


if __name__=='__main__':main()
