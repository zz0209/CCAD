"""Plot all query medians against source-only difference energy, no smoothing."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    run=args.run_dir;raw=run/'metrics.raw.jsonl'
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    assert all(r['mean_terms_cancelled_in_intervention'] for r in rows)
    methods=['target','raw','wrong_query_matched_energy'];colors=['#157b80','#b26917','#8e4f99']
    points=[]
    for condition in ('positive','negative'):
        for s,a in sorted({(r['source_seed'],r['source_atom']) for r in rows}):
            selected=[r for r in rows if r['condition']==condition and (r['source_seed'],r['source_atom'])==(s,a) and r['rank']==1]
            source=[r for r in selected if r['method']=='target' and r['endpoints']['centered_logits']['normalized_error'] is not None]
            for method in methods:
                valid=[r for r in selected if r['method']==method and r['endpoints']['centered_logits']['normalized_error'] is not None]
                points.append(dict(condition=condition,query=f's{s}:{a}',method=method,source_energy=statistics.median(r['hook']['source_energy'] for r in source),error=statistics.median(r['endpoints']['centered_logits']['normalized_error'] for r in valid),observations=len(valid),missing=len(selected)//5-len(valid)))
    with (run/'difference_energy_points.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    assert all(p['source_energy']>0 and p['error']>0 for p in points)
    xmin=math.floor(min(math.log10(p['source_energy']) for p in points));xmax=math.ceil(max(math.log10(p['source_energy']) for p in points))
    ymin=min(-2,math.floor(min(math.log10(p['error']) for p in points)));ymax=max(0,math.ceil(max(math.log10(p['error']) for p in points)))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="725" viewBox="0 0 1200 725" role="img" aria-labelledby="title desc">',
         '<title id="title">Mean-free dynamic correspondence: all query medians</title>',
         '<desc id="desc">Two log-log scatter panels show source hook difference energy versus centered-logit relative squared effect error. Strong-source queries have lower target error; weak-source queries often exceed the zero baseline of one. Raw and energy-matched wrong-query controls are shown for every query. Development, no confidence intervals.</desc>',
         '<rect width="1200" height="725" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#233044;font-size:12px}</style>',
         '<text x="50" y="35" style="font-size:24px">Mean-free dynamic correspondence</text>',
         '<text x="50" y="59" style="font-size:14px">All eight source-selected queries · rank 1 · one target each · descriptive development data</text>']
    def marker(x,y,k):
        if k==0:return f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[k]}"/>'
        if k==1:return f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="{colors[k]}"/>'
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
            svg.append(marker(x,y,k))
            if k==0:
                offset=-8 if x>left+width-65 else 8;anchor='end' if offset<0 else 'start'
                svg.append(f'<text x="{x+offset}" y="{y-7}" text-anchor="{anchor}">{p["query"]}</text>')
        svg.append(f'<text x="{left+width/2}" y="553" text-anchor="middle">Source hook difference energy (log10 axis)</text>')
    svg.append('<text transform="translate(25,320) rotate(-90)" text-anchor="middle">Relative squared logit-effect error (log10; lower is better)</text>')
    for k,label in enumerate(('Target relation','Raw map','Wrong query, matched energy')):
        x=120+330*k;svg.append(marker(x,590,k)+f'<text x="{x+14}" y="595" style="font-size:14px">{label}</text>')
    svg.extend(['<text x="50" y="631">Each point is a within-query median over recipient/donor pairs. Repeated donors, documents and seeds are dependent.</text>',
                '<text x="50" y="652">s5:710 positive has 2 valid pairs; all other query/condition groups have 4. Two unsupported pairs retained as missing, not success.</text>',
                '<text x="50" y="673">Error = sum((source effect - candidate effect)^2) / sum(source effect^2); vocabulary-centered logits; no smoothing or CI.</text>',
                '<text x="50" y="696">Dose caveat: positive s2:2176 swap norm is 4.87x recipient hook norm; a common source-defined dose sensitivity is pending.</text></svg>'])
    output=run/'difference_energy.svg';output.write_text('\n'.join(svg),encoding='utf-8')
    manifest={'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'svg_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'points':len(points),'log10_limits':{'x':[xmin,xmax],'y':[ymin,ymax]},'scope':'Provisional development plot; no visual pixel-preview or publisher compliance claim.'}
    (run/'difference_energy.provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
