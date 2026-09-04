"""Query-level descriptive tables and an SVG plot from saved causal feedback."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

def target_table(rows):
    """Keep target-seed heterogeneity visible; no pooled independence claim."""
    fields=['condition','source_seed','source_atom','target_seed','rank','endpoint','method']
    groups={}
    for row in rows:
        for endpoint in ('next_state','centered_logits'):
            value=row['endpoints'].get(endpoint)
            if value is None or value['normalized_error'] is None:continue
            key=tuple(row[k] if k!='endpoint' else endpoint for k in fields)
            groups.setdefault(key,[]).append(value)
    return [dict(zip(fields,key),median_error=statistics.median(v['normalized_error'] for v in values),median_source_rms=statistics.median(v['source_rms'] for v in values),observations=len(values)) for key,values in sorted(groups.items())]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--no-plot',action='store_true');args=p.parse_args()
    run=args.run_dir;raw=run/'metrics.raw.jsonl'
    rows=[json.loads(s) for s in raw.read_text().splitlines() if s]
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    groups={}
    for row in rows:
        for endpoint in ('next_state','centered_logits'):
            x=row['endpoints'].get(endpoint)
            if x is None or x['normalized_error'] is None:continue
            key=(row['condition'],row['source_seed'],row['source_atom'],row['rank'],endpoint,row['method'])
            groups.setdefault(key,[]).append(x)
    table=[dict(zip(['condition','source_seed','source_atom','rank','endpoint','method'],key),median_error=statistics.median(x['normalized_error'] for x in values),median_source_rms=statistics.median(x['source_rms'] for x in values),observations=len(values)) for key,values in sorted(groups.items())]
    with (run/'query_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    per_target=target_table(rows)
    with (run/'target_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_target[0]));writer.writeheader();writer.writerows(per_target)
    summaries={}
    available_methods=sorted({r['method'] for r in rows})
    for condition in ('positive','negative'):
      for rank in sorted({r['rank'] for r in rows}):
        for endpoint in ('next_state','centered_logits'):
            g=[r for r in table if r['condition']==condition and r['rank']==rank and r['endpoint']==endpoint]
            by={(r['source_seed'],r['source_atom'],r['method']):r for r in g}
            queries=sorted({(r['source_seed'],r['source_atom']) for r in g})
            summaries[f'{condition}/r{rank}/{endpoint}']={'query_count':len(queries),'median_error_across_query_medians':{method:statistics.median(r['median_error'] for r in g if r['method']==method) for method in available_methods},'target_better_than_raw_queries':sum(by[s,a,'target']['median_error']<by[s,a,'raw']['median_error'] for s,a in queries),'target_better_than_wrong_matched_queries':sum(by[s,a,'target']['median_error']<by[s,a,'wrong_query_matched_energy']['median_error'] for s,a in queries),'median_source_rms':statistics.median(by[s,a,'target']['median_source_rms'] for s,a in queries)}
            if 'source_mean_only' in available_methods:
                summaries[f'{condition}/r{rank}/{endpoint}']['target_better_than_source_mean_only_queries']=sum(by[s,a,'target']['median_error']<by[s,a,'source_mean_only']['median_error'] for s,a in queries)
    report={'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'summary_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'aggregation':'Within each condition, median over targets and document sequences within query, then median across queries. Descriptive; shared seeds and documents are dependent. See observations column.','normalization':'sum((source_effect-candidate_effect)^2)/sum(source_effect^2), across all sequence positions and endpoint dimensions. Logits centered across vocabulary.','denominators':{'raw_rows':len(rows),'missing_normalized_endpoint_errors':sum(v['normalized_error'] is None for r in rows for v in r['endpoints'].values()),'empty_intervention_masks':sum(not r.get('intervention_positions',[]) for r in rows),'observations_per_query_method_endpoint':sorted({r['observations'] for r in table})},'summary':summaries}
    requested={(r['condition'],r['source_seed'],r['source_atom']) for r in rows}
    observed={(r['condition'],r['source_seed'],r['source_atom']) for r in table}
    report['denominators'].update(requested_query_count=len({(r['source_seed'],r['source_atom']) for r in rows}),unsupported_query_conditions=sorted(requested-observed),zero_source_hook_rows=sum(r['hook']['source_energy']==0 for r in rows))
    (run/'query_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.no_plot:
        print(json.dumps(report,indent=2));return
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="655" viewBox="0 0 1100 655" role="img" aria-labelledby="title desc">','<title id="title">Development source-reference causal feedback</title>','<desc id="desc">Eight query medians on a common log10 error scale. Target relations outperform energy-matched wrong queries in all eight, while raw maps are competitive. Positive condition only; negative condition is in the accompanying CSV.</desc>','<rect width="1100" height="655" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#233044} .small{font-size:12px}</style>','<text x="45" y="35" font-size="23">Signed transport preserves source-local effects</text>','<text x="45" y="60" font-size="14">Development · positive condition · content-position interventions · rank 1 · query medians</text>']
    methods=['target','raw','wrong_query_matched_energy'];colors=['#157b80','#b26917','#8e4f99']
    def marker(x,y,k):
        if k==0:return f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[k]}"/>'
        if k==1:return f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="{colors[k]}"/>'
        return f'<path d="M{x},{y-5}l5,9h-10z" fill="{colors[k]}"/>'
    for j,endpoint in enumerate(('next_state','centered_logits')):
        x0=100+j*525;y0=100;w=440;h=380
        svg.append(f'<text x="{x0}" y="88" font-size="17">{endpoint.replace("_"," ")}</text>')
        g=[r for r in table if r['condition']=='positive' and r['rank']==1 and r['endpoint']==endpoint]
        queries=sorted({(r['source_seed'],r['source_atom']) for r in g});by={(r['source_seed'],r['source_atom'],r['method']):r for r in g}
        def y(v):return y0+h*(1-(math.log10(max(v,1e-5))+5)/6)
        for tick in (1e-5,1e-4,1e-3,.01,.1,1,10):
            yy=y(tick);svg.append(f'<path d="M{x0},{yy}h{w}" stroke="#e1e6ec"/><text x="{x0-8}" y="{yy+4}" text-anchor="end" class="small">{tick:g}</text>')
        for qi,(s,a) in enumerate(queries):
            xx=x0+(qi+.5)*w/len(queries)
            svg.append(f'<text transform="translate({xx},495) rotate(45)" class="small">s{s}: {a}</text>')
            for k,method in enumerate(methods):
                val=by[s,a,method]['median_error']
                assert 1e-5 <= val <= 10, 'Expand the common log limits explicitly; do not clip values.'
                svg.append(marker(xx+(k-1)*8,y(val),k))
    for i,(label,color) in enumerate(zip(['Target relation','Raw map','Wrong query, matched energy'],colors)):
        x=80+i*310;svg.append(marker(x,563,i)+f'<text x="{x+12}" y="568" font-size="14">{label}</text>')
    n_targets=sorted({len({r['target_seed'] for r in rows if (r['source_seed'],r['source_atom'])==q}) for q in {(r['source_seed'],r['source_atom']) for r in rows}})
    n_docs=sorted({len({r['sequence'] for r in rows if r['condition']=='positive' and (r['source_seed'],r['source_atom'])==q}) for q in {(r['source_seed'],r['source_atom']) for r in rows}})
    svg.append(f'<text x="45" y="596" class="small">Y: squared effect error / squared source effect, log10 axis; lower is better. No zero/missing values in these points.</text><text x="45" y="615" class="small">Eight source-selected queries; targets/query: {n_targets}; positive sequences/query: {n_docs}. Shared seeds/documents are dependent.</text><text x="45" y="634" class="small">Descriptive development result, no CI or audit claim. Full positive/negative tables: query_summary.csv.</text></svg>')
    (run/'causal_feedback.svg').write_text('\n'.join(svg),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
