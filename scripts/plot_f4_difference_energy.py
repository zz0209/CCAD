"""Plot all query medians against source-only difference energy, no smoothing."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--include-atoms',action='store_true');args=p.parse_args()
    run=args.run_dir;raw=run/'metrics.raw.jsonl'
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    cfg=json.loads((run/'config.resolved.json').read_text())
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    assert all(r['mean_terms_cancelled_in_intervention'] for r in rows)
    methods=['target','raw','wrong_query_matched_energy'];colors=['#157b80','#b26917','#8e4f99']
    labels=['Target relation','Raw map','Wrong query, matched']
    if args.include_atoms:
        methods.extend(['single_atom_level','single_atom_dynamic']);colors.extend(['#0072B2','#000000']);labels.extend(['Atom: second moment','Atom: dynamic'])
    if not set(methods).issubset({r['method'] for r in rows}): raise ValueError('Requested plot methods missing')
    points=[];missing_groups=[]
    for condition in ('positive','negative'):
        for s,a in sorted({(r['source_seed'],r['source_atom']) for r in rows}):
            selected=[r for r in rows if r['condition']==condition and (r['source_seed'],r['source_atom'])==(s,a) and r['rank']==1]
            source=[r for r in selected if r['method']=='target' and r['endpoints']['centered_logits']['normalized_error'] is not None]
            if not source:
                missing_groups.append(f'{condition} s{s}:{a}')
                continue
            for method in methods:
                valid=[r for r in selected if r['method']==method and r['endpoints']['centered_logits']['normalized_error'] is not None]
                target_medians=[statistics.median(r['endpoints']['centered_logits']['normalized_error'] for r in valid if r['target_seed']==t) for t in sorted({r['target_seed'] for r in valid})]
                points.append(dict(condition=condition,query=f's{s}:{a}',method=method,source_energy=statistics.median(r['hook']['source_energy'] for r in source),error=statistics.median(r['endpoints']['centered_logits']['normalized_error'] for r in valid),target_median_min=min(target_medians),target_median_max=max(target_medians),observations=len(valid),missing=sum(r['method']==method for r in selected)-len(valid)))
    with (run/'difference_energy_points.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    assert all(p['source_energy']>0 and p['error']>0 for p in points)
    xmin=math.floor(min(math.log10(p['source_energy']) for p in points));xmax=math.ceil(max(math.log10(p['source_energy']) for p in points))
    ymin=min(-2,math.floor(min(math.log10(min(p['error'],p['target_median_min'])) for p in points)));ymax=max(0,math.ceil(max(math.log10(max(p['error'],p['target_median_max'])) for p in points)))
    target_count=cfg['targets_per_query']
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="725" viewBox="0 0 1200 725" role="img" aria-labelledby="title desc">',
         '<title id="title">Mean-free dynamic correspondence: all query medians</title>',
         '<desc id="desc">Two log-log panels show applied source hook difference energy versus centered-logit relative squared effect error, with all eight queries and all requested controls. Whiskers are the range of target-seed medians, not confidence intervals. Weak-source failures remain visible. Atom objectives, when shown, were fitted on discovery and reused unchanged. Development only; source-aligned maps are not native deletion.</desc>',
         '<rect width="1200" height="725" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#233044;font-size:12px}</style>',
         '<text x="50" y="35" style="font-size:24px">Mean-free dynamic correspondence</text>',
         f'<text x="50" y="59" style="font-size:14px">All eight source-selected queries · rank 1 · {target_count} targets each · descriptive development data</text>']
    def marker(x,y,k):
        if k==0:return f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[k]}"/>'
        if k==1:return f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="{colors[k]}"/>'
        if k==3:return f'<path d="M{x},{y-6}l6,6l-6,6l-6,-6z" fill="white" stroke="{colors[k]}" stroke-width="2"/>'
        if k==4:return f'<path d="M{x-5},{y-5}l10,10M{x-5},{y+5}l10,-10" stroke="{colors[k]}" stroke-width="2"/>'
        return f'<path d="M{x},{y-5}l5,9h-10z" fill="{colors[k]}"/>'
    for j,condition in enumerate(('positive','negative')):
        left=100+j*575;top=110;width=450;height=390
        X=lambda value:left+width*(math.log10(value)-xmin)/(xmax-xmin)
        Y=lambda value:top+height*(ymax-math.log10(value))/(ymax-ymin)
        svg.append(f'<text x="{left}" y="90" style="font-size:18px">{condition.capitalize()} recipient</text>')
        for exponent in range(xmin,xmax+1):
            x=X(10**exponent);svg.append(f'<path d="M{x},{top}v{height}" stroke="#e4e7eb"/><text x="{x}" y="522" text-anchor="middle">1e{exponent}</text>')
        for exponent in range(ymin,ymax+1):
            y=Y(10**exponent);svg.append(f'<path d="M{left},{y}h{width}" stroke="#e4e7eb"/><text x="{left-12}" y="{y+4}" text-anchor="end">1e{exponent}</text>')
        svg.append(f'<path d="M{left},{Y(1)}h{width}" stroke="#233044" stroke-dasharray="6,5"/><text x="{left+width-5}" y="{Y(1)-9}" text-anchor="end">Zero / constant difference = 1</text>')
        for p in points:
            if p['condition']!=condition:continue
            k=methods.index(p['method']);x=X(p['source_energy']);y=Y(p['error'])
            if target_count>1:
                lo=Y(p['target_median_min']);hi=Y(p['target_median_max'])
                svg.append(f'<path class="target-range" d="M{x},{lo}V{hi}M{x-3},{lo}h6M{x-3},{hi}h6" stroke="{colors[k]}" stroke-width="1.3"/>')
            svg.append('<g class="data-point">'+marker(x,y,k)+'</g>')
            if k==0:
                offset=-8 if x>left+width-65 else 8;anchor='end' if offset<0 else 'start'
                svg.append(f'<text x="{x+offset}" y="{y-7}" text-anchor="{anchor}">{p["query"]}</text>')
        svg.append(f'<text x="{left+width/2}" y="553" text-anchor="middle">Applied source hook difference energy (log10 axis)</text>')
    svg.append('<text transform="translate(25,320) rotate(-90)" text-anchor="middle">Relative squared logit-effect error (log10; lower is better)</text>')
    for k,label in enumerate(labels):
        x=65+225*k if len(labels)>3 else 120+330*k;svg.append(marker(x,590,k)+f'<text x="{x+14}" y="595" style="font-size:14px">{label}</text>')
    dose_note=f'Common source-defined dose: source norm / masked recipient hook norm at most {cfg["maximum_source_hook_fraction"]:g}; candidates share scale.' if cfg.get('maximum_source_hook_fraction') else 'Natural dose: positive s2:2176 source norm is 4.87x recipient hook norm; do not interpret as moderate-dose evidence.'
    pair_status={(r['source_seed'],r['source_atom'],r['condition'],r['sequence']):r.get('donor_status') for r in rows}
    missing_note=f'{sum(s=="NO_ELIGIBLE_PAIR" for s in pair_status.values())}/{len(pair_status)} pairs unsupported, retained as missing, not success. Entire missing groups: '+(', '.join(missing_groups) if missing_groups else 'none')+'.'
    svg.extend(['<text x="50" y="631">Points pool pairs and targets within query; whiskers span target medians, not CI. Shared documents/seeds are dependent.</text>',
                f'<text x="50" y="652">{missing_note}</text>',
                '<text x="50" y="673">Error = sum((source effect - candidate effect)^2) / sum(source effect^2); vocabulary-centered logits; no smoothing or CI.</text>',
                f'<text x="50" y="696">{dose_note}</text></svg>'])
    output=run/'difference_energy.svg';output.write_text('\n'.join(svg),encoding='utf-8')
    manifest={'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'svg_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'points':len(points),'log10_limits':{'x':[xmin,xmax],'y':[ymin,ymax]},'scope':'Provisional development plot; no visual pixel-preview or publisher compliance claim.'}
    manifest['whiskers']='Minimum to maximum of within-target pair medians; not confidence intervals.'
    manifest['targets_per_query']=target_count
    manifest['missing_query_conditions']=missing_groups
    manifest['methods']=methods
    (run/'difference_energy.provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
