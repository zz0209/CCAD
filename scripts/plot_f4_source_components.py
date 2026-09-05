"""All measured source-part errors, with raw controls and dependent target points."""
import csv
import json
import math
import sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
from summarize_f4_probability import ROOT,write,sha256


def main():
    run=ROOT/'runs/F4_source_components_interleaved_v2_20260905';path=run/'component_summary.json';j=json.loads(path.read_text());out=run/'source_component_fidelity.png'
    if out.exists():raise FileExistsError(out)
    for r in j['inputs']:assert sha256(Path(r['path']))==r['sha256']
    rows=list(csv.DictReader((run/'COMPONENT_ROWS.csv').open(encoding='utf-8')))
    rows=[r for r in rows if r['scope']=='intervention_positions' and r['method']!='source']
    metrics=['normalized_kl_error','normalized_nll_delta_squared_error'];values=[float(r[k]) for r in rows for k in metrics]
    assert len(rows)==75 and all(math.isfinite(v) and v>0 for v in values)
    lo=math.floor(math.log10(min(values)));hi=max(0,math.ceil(math.log10(max(values))))
    im=Image.new('RGB',(2400,1440),'white');d=ImageDraw.Draw(im);points=[];blue='#0072B2';black='#202020'
    def text(x,y,s,size=27,anchor=None,bold=False):
        font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size);d.text((x,y),s,font=font,fill=black,anchor=anchor)
    def dot(x,y,size=7,open_=False):d.ellipse((x-size,y-size,x+size,y+size),fill='white' if open_ else blue,outline=blue,width=2)
    def cross(x,y):d.line((x-8,y-8,x+8,y+8),fill=black,width=3);d.line((x-8,y+8,x+8,y-8),fill=black,width=3)
    text(1200,30,'Source-defined parts retain real effects across recipient seeds',44,'ma',True)
    text(1200,91,'Frozen source rank1 basis | Two discovery-ranked 16-member groups | Exposed development',28,'ma')
    dot(360,164);text(384,164,'Long k128: full readout (median of four targets)',28,'lm')
    cross(1370,164);text(1394,164,'Raw: same source component and fitting budget',28,'lm')
    for col,metric in enumerate(metrics):
        left=305+1180*col;width=835;top=257;bottom=1092
        text(left+width/2,213,'KL / source KL' if col==0 else 'NLL-change squared error / source energy',29,'ma',True)
        px=lambda v:left+(math.log10(v)-lo)/(hi-lo)*width
        for tick in range(lo,hi+1):
            x=px(10**tick)
            if tick==0:
                for y in range(top,bottom,17):d.line((x,y,x,min(y+9,bottom)),fill='#666666',width=2)
            else:d.line((x,top,x,bottom),fill='#E4E4E4',width=1)
            text(x,bottom+14,'1' if tick==0 else f'10^{tick}',25,'ma')
        for cid in range(5):
            first=next(r for r in j['cases'] if r['case_id']==cid)
            y0=280+cid*166
            if col==0:
                text(28,y0+40,f"{first['source_seed']}:{first['source_atom']}",27,'lm',True)
                text(28,y0+73,first['condition'],25,'lm')
            if cid:d.line((left-80,y0-20,left+width,y0-20),fill='#D9D9D9',width=1)
            for offset,component in enumerate(('full','A','B')):
                y=y0+offset*48;text(left-25,y,component,26,'rm')
                rr=[r for r in rows if int(r['case_id'])==cid and r['component']==component]
                targets=sorted((r for r in rr if r['method'].startswith('long_')),key=lambda r:r['method'])
                vals=[float(r[metric]) for r in targets]
                d.line((px(min(vals)),y-6,px(max(vals)),y-6),fill=blue,width=2)
                for index,r in enumerate(targets):
                    v=float(r[metric]);dot(px(v),y-7+index*4,4,True);points.append(dict(case_id=cid,component=component,metric=metric,method=r['method'],value=v,kind='dependent_target'))
                med=next(r[metric] for r in j['cases'] if r['case_id']==cid and r['component']==component and r['scope']=='intervention_positions' and r['method']=='long')
                dot(px(med),y-6,8);points.append(dict(case_id=cid,component=component,metric=metric,method='long',value=med,kind='case_median'))
                raw=float(next(r[metric] for r in rr if r['method']=='raw'));cross(px(raw),y+11);points.append(dict(case_id=cid,component=component,metric=metric,method='raw',value=raw,kind='raw'))
        d.line((left,bottom,left+width,bottom),fill=black,width=2)
        text(left+width/2,bottom+58,'Lower is better; log10 scale. Dashed line: no intervention.',24,'ma')
    footer=[
        '5 cases / 3 active queries / 20 dependent target-case pairs per component. Thin spans: observed target range, NOT confidence intervals.',
        'Source A and B both have nonzero measured effects in all five cases. They share one output direction; this is not evidence of two semantic mechanisms.',
        'All 8 requests retained: 2 unmatched and 1 source-rejected. The earlier head/tail split had zero source-tail effect and is preserved in the report.',
        'Common source-family dose, not matched candidate energy. No sparse target budget here; compact component reuse remains to be tested.',
        'Provisional internal figure. Full source effects, secondary endpoints, interactions and every target: COMPONENT_COMPARISON.md / COMPONENT_ROWS.csv.'
    ]
    for i,line in enumerate(footer):text(55,1203+i*40,line,24)
    im.save(out,dpi=(300,300))
    write(run/'source_component_figure_points.json',dict(points=points,log10_limits=[lo,hi],source_sha256=sha256(path)))
    write(run/'source_component_figure_manifest.json',dict(path=str(out),sha256=sha256(out),source_sha256=sha256(path),generator_sha256=sha256(Path(__file__)),
        python=sys.executable,pillow=pillow_version,pixels=list(im.size),mode=im.mode,points=len(points),dpi_nominal=300,uncertainty='observed four dependent target ranges, not CI',
        transform='primary positions only; common log10 limits, all60target errors plus15raw errors per metric; case medians shown; no exclusions among evaluated values',
        alt_text='Two log-scale panels show full, A and B component transfer errors for all five source cases. All long-recipient component errors are below no intervention, while same-interface raw controls are usually lower. Individual dependent targets and their case medians are visible. This is exposed-development rank1 component reuse, not separate semantic mechanisms or a compact target representation.'))
    print(json.dumps(dict(path=str(out),points=len(points),limits=[lo,hi])))


if __name__=='__main__':main()
