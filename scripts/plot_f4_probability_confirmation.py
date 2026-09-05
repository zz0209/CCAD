"""Provisional internal figure: all32requests, explicit missingness, common log axes."""
import json
import math
import sys
import platform
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
from summarize_f4_probability import ROOT,METHODS,sha256,write,median

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
NAMES=['Full FCC','Global rows','Raw hook','Fixed top16']


def main():
    run=ROOT/'runs/F4_probability_confirmation_expanded_v1_20260905'
    path=run/'probability_summary.json';j=json.loads(path.read_text());output=run/'probability_confirmation.png'
    if output.exists():raise ValueError('Figure already exists')
    # Replay displayed medians from saved raw; not an independent scientific review.
    verified=0
    for panel in ('original','expanded'):
        rawpath=ROOT/'runs'/f'F4_probability_confirmation_{panel}_v1_20260905/metrics.raw.jsonl'
        assert any(i['path']==str(rawpath) and i['sha256']==sha256(rawpath) for i in j['inputs'])
        raw=[json.loads(l) for l in rawpath.read_text().splitlines()]
        for c in [r for r in j['cases'] if r['panel']==panel and r['selected']]:
            for method in METHODS:
                for metric in ('normalized_kl_error','normalized_nll_delta_squared_error'):
                    values=[r['probability_endpoints']['intervention_positions'][metric] for r in raw if r['source_seed']==c['source_seed'] and r['source_atom']==c['source_atom'] and r['condition']==c['condition'] and r['method']==method]
                    assert len(values)==4 and median(values)==c['scopes']['intervention_positions']['methods'][method][metric];verified+=1
    values=[m[k] for c in j['cases'] if c['selected'] for m in c['scopes']['intervention_positions']['methods'].values() for k in ('normalized_kl_error','normalized_nll_delta_squared_error') if m[k] is not None]
    if any(v<0 or not math.isfinite(v) for v in values):raise ValueError('Invalid probability error')
    positive=[v for v in values if v>0];low=math.floor(math.log10(min(positive)));high=max(0,math.ceil(math.log10(max(positive))))
    image=Image.new('RGB',(2400,2130),'white');draw=ImageDraw.Draw(image);points=[]
    def text(x,y,s,size=28,anchor=None,bold=False,fill='#202020'):
        font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
        draw.text((x,y),s,font=font,fill=fill,anchor=anchor)
    def marker(x,y,k):
        color=COLORS[k];r=7
        if k==0:draw.ellipse((x-r,y-r,x+r,y+r),fill=color)
        elif k==1:draw.rectangle((x-r,y-r,x+r,y+r),fill='white',outline=color,width=3)
        elif k==2:draw.polygon([(x,y-r-2),(x-r-2,y+r),(x+r+2,y+r)],fill=color)
        else:draw.line((x-r,y-r,x+r,y+r),fill=color,width=3);draw.line((x-r,y+r,x+r,y-r),fill=color,width=3)
    text(1200,40,'Predictive correspondence on 120 fresh documents',46,'ma',True)
    text(1200,103,'Frozen source workflow | 32 requested conditions | median over four dependent target seeds',30,'ma')
    for k,name in enumerate(NAMES):
        xx=250+k*520;marker(xx,170,k);text(xx+23,170,name,31,'lm')
    text(1200,220,'Primary: original intervention positions | common log10 error axes | dashed line = no-intervention reference (1)',28,'ma')
    for row,panel in enumerate(('original','expanded')):
        pc=[r for r in j['cases'] if r['panel']==panel];cov=j['coverage'][panel]
        assert len(pc)==16
        for col,(metric,label) in enumerate([('normalized_kl_error','KL ratio'),('normalized_nll_delta_squared_error','Observed-token NLL change: squared error')]):
            left=65+col*1190;top=375+row*795;gx=left+210;gw=875;gh=608
            X=lambda v:gx+(math.log10(max(v,10**low))-low)/(high-low)*gw
            text(left,top-92,f"{panel.title()} | {label}",32,bold=True)
            text(left,top-46,f"Selected {cov['selected']}/16; class-matched support {cov['matched_supported']}/16",27)
            for exp in range(low,high+1):
                x=X(10**exp);draw.line((x,top-18,x,top+gh),fill='#DDDDDD',width=2);text(x,top+gh+22,f'1e{exp}',25,'ma')
            x=X(1)
            for yy in range(top-18,top+gh,23):draw.line((x,yy,x,min(yy+12,top+gh)),fill='#202020',width=3)
            for n,c in enumerate(pc):
                y=top+n*38
                text(gx-18,y,f"{c['source_seed']}:{c['source_atom']} {'+' if c['condition']=='positive' else '-'}",26,'rm')
                if not c['selected']:
                    label='source not selected' if c['entry'] is not None else 'no compatible donor / support'
                    text(gx+14,y,label,24,'lm',fill='#666666');continue
                for k,method in enumerate(METHODS):
                    v=c['scopes']['intervention_positions']['methods'][method][metric];yy=y+(k-1.5)*7
                    if v is None:text(gx+14,yy,f'{NAMES[k]}: null',21,'lm');continue
                    xx=X(v);marker(xx,yy,k)
                    if v==0:text(xx+13,yy,'0',19,'lm')
                    points.append(dict(panel=panel,source_seed=c['source_seed'],source_atom=c['source_atom'],condition=c['condition'],method=method,metric=metric,value=v,x=xx,y=yy))
            draw.line((gx,top+gh,gx+gw,top+gh),fill='#202020',width=2)
    footer=['Lower error is better; KL and NLL ratios have different meanings despite the shared scale.',
        'All requested rows are retained. Missing/not-selected rows are not zeros; zero, if present, is labeled at the left bound.',
        'Only source-selected pairs were evaluated. Shared queries, documents and seeds are not independent replicates.',
        'Joint interventions; no semantic uniqueness claim. Full source scales, secondary endpoints and failures are in the linked tables.',
        'Provisional internal figure, not verified for a specific submission format.']
    for n,line in enumerate(footer):text(1200,1900+n*39,line,26,'ma')
    image.save(output,dpi=(300,300))
    write(run/'figure_points.json',dict(summary_sha256=sha256(path),points=points,all_requests=32,verified_median_cells=verified,log10_limits=[low,high],zero_policy='labeled at left bound',missing_policy='explicit text, not plotted aszero'))
    write(run/'figure_manifest.json',dict(path=str(output),sha256=sha256(output),pixels=list(image.size),mode=image.mode,dpi=300,pillow=pillow_version,python=sys.executable,python_version=platform.python_version(),ML_runtime='not modified or used for plot',generator_script_sha256=sha256(Path(__file__)),summary_sha256=sha256(path),points=len(points),verified_median_cells=verified,replicate_unit='case median over4dependenttargets; no confidence interval',submission_requirements='pending',alt_text='Four panels compare primary KL and observed-token NLL effect errors for every requested original and expanded query-condition. Colored shapes identify four methods; no-donor and source-not-selected rows remain visible. The vertical line marks error1. Full numerical values and exceptions are available in PROBABILITY_TABLE.md.'))
    print(json.dumps(dict(path=str(output),points=len(points),verified_median_cells=verified,log10_limits=[low,high])))


if __name__=='__main__':main()
