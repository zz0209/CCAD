"""All measured source-part errors, with raw controls and dependent target points."""
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
from summarize_f4_probability import ROOT,write,sha256


def main(compact=False,confirmation=False):
    run=ROOT/'runs'/('F4_compact_source_components_v1_20260905' if compact else 'F4_source_components_interleaved_v2_20260905')
    if confirmation:run=ROOT/'runs/F4_compact_component_confirmation_v1_20260905'
    count=6 if confirmation else 5
    path=run/('compact_component_summary.json' if compact else 'component_summary.json');j=json.loads(path.read_text());out=run/('compact_component_fidelity.png' if compact else 'source_component_fidelity.png')
    if out.exists():raise FileExistsError(out)
    for r in j['inputs']:assert sha256(Path(r['path']))==r['sha256']
    rows=list(csv.DictReader((run/('COMPACT_COMPONENT_ROWS.csv' if compact else 'COMPONENT_ROWS.csv')).open(encoding='utf-8')))
    methods=['short_shared16','long_shared16','long_single'] if compact else ['long']
    if compact:
        rows=[r for r in rows if r['method'] in methods+['raw']]
        rows=[dict(r,method=r['method']+'_'+r['target_seed']) if r['method']!='raw' else r for r in rows]
    rows=[r for r in rows if r['scope']=='intervention_positions' and r['method']!='source']
    metrics=['normalized_kl_error','normalized_nll_delta_squared_error'];values=[float(r[k]) for r in rows for k in metrics]
    assert len(rows)==(count*39 if compact else 75) and all(math.isfinite(v) and v>0 for v in values)
    lo=math.floor(math.log10(min(values)));hi=max(0,math.ceil(math.log10(max(values))))
    im=Image.new('RGB',(2400,1950 if confirmation else (1740 if compact else 1440)),'white');d=ImageDraw.Draw(im);points=[];blue='#0072B2';black='#202020'
    colors=['#D55E00',blue,'#009E73'] if compact else [blue]
    def text(x,y,s,size=27,anchor=None,bold=False):
        font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size);d.text((x,y),s,font=font,fill=black,anchor=anchor)
    def dot(x,y,size=7,open_=False):d.ellipse((x-size,y-size,x+size,y+size),fill='white' if open_ else blue,outline=blue,width=2)
    def cross(x,y):d.line((x-8,y-8,x+8,y+8),fill=black,width=3);d.line((x-8,y+8,x+8,y-8),fill=black,width=3)
    def marker(x,y,k,r=7,open_=False):
        color=colors[k];fill='white' if open_ else color
        if not compact or k==1:d.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=color,width=2)
        elif k==0:d.polygon([(x,y-r-2),(x-r-2,y+r),(x+r+2,y+r)],fill=fill,outline=color,width=2)
        else:d.rectangle((x-r,y-r,x+r,y+r),fill=fill,outline=color,width=2)
    text(1200,30,'Frozen shared16 transfers source parts to fresh documents' if confirmation else 'One shared16 interface transfers both source parts' if compact else 'Source-defined parts retain real effects across recipient seeds',44,'ma',True)
    text(1200,91,'One common 16-member target support for A and B | No new fitting | 108 fresh documents' if confirmation else 'One common 16-member target support for A and B | Frozen source parts | Exposed development' if compact else 'Frozen source rank1 basis | Two discovery-ranked 16-member groups | Exposed development',28,'ma')
    if compact:
        for k,label in enumerate(['Short: shared16','Long: shared16','Long: joint best1']):marker(130+600*k,164,k);text(156+600*k,164,label,28,'lm')
        cross(1930,164);text(1956,164,'Raw: full rank1',28,'lm')
    else:
        dot(360,164);text(384,164,'Long k128: full readout (median of four targets)',28,'lm')
        cross(1370,164);text(1394,164,'Raw: same source component and fitting budget',28,'lm')
    for col,metric in enumerate(metrics):
        left=305+1180*col;width=835;top=257;bottom=1530 if confirmation else (1320 if compact else 1092)
        text(left+width/2,213,'KL / source KL' if col==0 else 'NLL-change squared error / source energy',29,'ma',True)
        px=lambda v:left+(math.log10(v)-lo)/(hi-lo)*width
        for tick in range(lo,hi+1):
            x=px(10**tick)
            if tick==0:
                for y in range(top,bottom,17):d.line((x,y,x,min(y+9,bottom)),fill='#666666',width=2)
            else:d.line((x,top,x,bottom),fill='#E4E4E4',width=1)
            text(x,bottom+14,'1' if tick==0 else f'10^{tick}',25,'ma')
        for cid in range(count):
            first=next(r for r in j['cases'] if r['case_id']==cid)
            y0=280+cid*(210 if compact else 166)
            if col==0:
                text(28,y0+40,f"{first['source_seed']}:{first['source_atom']}",27,'lm',True)
                text(28,y0+73,first['condition'],25,'lm')
            if cid:d.line((left-80,y0-20,left+width,y0-20),fill='#D9D9D9',width=1)
            for offset,component in enumerate(('full','A','B')):
                y=y0+offset*(60 if compact else 48);text(left-25,y,component,26,'rm')
                rr=[r for r in rows if int(r['case_id'])==cid and r['component']==component]
                for k,method in enumerate(methods):
                    yy=y-18+k*12 if compact else y-6
                    targets=sorted((r for r in rr if r['method'].startswith(method+'_')),key=lambda r:r['method']);assert len(targets)==4
                    vals=[float(r[metric]) for r in targets];d.line((px(min(vals)),yy,px(max(vals)),yy),fill=colors[k],width=2)
                    for index,r in enumerate(targets):
                        v=float(r[metric]);marker(px(v),yy-3+index*2,k,3,True);points.append(dict(case_id=cid,component=component,metric=metric,method=r['method'],value=v,kind='dependent_target'))
                    med=median(vals);marker(px(med),yy,k,7);points.append(dict(case_id=cid,component=component,metric=metric,method=method,value=med,kind='case_median'))
                raw=float(next(r[metric] for r in rr if r['method']=='raw'));cross(px(raw),y+(18 if compact else 11));points.append(dict(case_id=cid,component=component,metric=metric,method='raw',value=raw,kind='raw'))
        d.line((left,bottom,left+width,bottom),fill=black,width=2)
        text(left+width/2,bottom+58,'Lower is better; log10 scale. Dashed line: no intervention.',24,'ma')
    footer=[
        f'{count} cases / 3 active queries / {4*count} dependent target-case pairs per component. Thin spans: observed target range, NOT confidence intervals.',
        'Source parts share one output direction, not two semantic mechanisms. '+('Exceptions include 1230-positive A NLL and 1230-negative B NLL.' if confirmation else 'Source A and B both have nonzero measured effects in all five cases.'),
        ('All 6 frozen-family requests evaluated. Original query 3:1144 has no frozen component map: its 2 signs remain a coverage boundary.' if confirmation else 'All 8 requests retained: 2 unmatched and 1 source-rejected. The earlier head/tail split had zero source-tail effect and is preserved in the report.'),
        ('A and B share the SAME 16 selected target members (union16). Raw and best single-atom controls are retained; no optimal-sparsity or native-edit claim.' if compact else 'Common source-family dose, not matched candidate energy. No sparse target budget here; compact component reuse remains to be tested.'),
        ('Provisional figure; same source dose, not matched candidate energy. All7methods/full controls/secondary endpoints: COMPACT_COMPONENT_COMPARISON.md.' if compact else 'Provisional internal figure. Full source effects, secondary endpoints, interactions and every target: COMPONENT_COMPARISON.md / COMPONENT_ROWS.csv.')
    ]
    for i,line in enumerate(footer):text(55,(1670 if confirmation else (1460 if compact else 1203))+i*40,line,24)
    im.save(out,dpi=(300,300))
    prefix='compact_component' if compact else 'source_component'
    write(run/(prefix+'_figure_points.json'),dict(points=points,log10_limits=[lo,hi],source_sha256=sha256(path)))
    write(run/(prefix+'_figure_manifest.json'),dict(path=str(out),sha256=sha256(out),source_sha256=sha256(path),generator_sha256=sha256(Path(__file__)),
        python=sys.executable,pillow=pillow_version,pixels=list(im.size),mode=im.mode,points=len(points),dpi_nominal=300,uncertainty='observed four dependent target ranges, not CI',
        transform='primary positions only; common log10 limits, all selected-method target errors and raw errors; case medians shown; no exclusions among evaluated values',
        alt_text=('Two log-scale panels show all six frozen-family cases and full/A/B source operations on fresh documents. Long shared16 usually improves on short shared16, but source1230 component exceptions and two worse-than-no-intervention A NLL targets remain visible. Joint single-atom and raw controls remain competitive. Individual dependent targets, medians and ranges are shown. Not native edits or independent semantic mechanisms.' if confirmation else 'Two log-scale panels show all five cases and full/A/B source operations with one shared16 target support. Long shared16 errors are lower than short shared16 in every case; joint single-atom controls sometimes remain competitive and raw is generally stronger. Individual dependent targets, case medians and ranges are visible. Rank1 mathematical components, not independent semantics or native edits.' if compact else 'Two log-scale panels show full, A and B component transfer errors for all five source cases. All long-recipient component errors are below no intervention, while same-interface raw controls are usually lower. Individual dependent targets and their case medians are visible. This is exposed-development rank1 component reuse, not separate semantic mechanisms or a compact target representation.')))
    print(json.dumps(dict(path=str(out),points=len(points),limits=[lo,hi])))


if __name__=='__main__':main('--compact' in sys.argv or '--confirmation' in sys.argv,'--confirmation' in sys.argv)
