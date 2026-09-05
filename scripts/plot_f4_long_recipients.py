"""Internal fixed-budget comparison; every eligible case and target is visible."""
import csv
import argparse
import json
import math
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
from summarize_f4_probability import ROOT, sha256, write, median

SKILL=Path('C:/Users/zz/.codex/skills/scientific-visualization')
sys.path.insert(0,str(SKILL/'assets'))
from color_palettes import OKABE_ITO_ON_WHITE


def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group();modes.add_argument('--confirmation-apply',type=Path);modes.add_argument('--five-seed-fit',type=Path);args=parser.parse_args()
    five=args.five_seed_fit is not None
    fresh=args.confirmation_apply is not None or five
    run=(args.five_seed_fit if five else args.confirmation_apply) if fresh else ROOT/'runs/F4_long_recipient_fit_v1_20260905'
    run=run if run.is_absolute() else ROOT/run
    path=run/'recipient_comparison.json';j=json.loads(path.read_text());out=run/'recipient_compactness_v2.png'
    if out.exists():raise ValueError('Figure already exists')
    for inp in j['inputs']:assert sha256(Path(inp['path']))==inp['sha256']
    rows=list(csv.DictReader((run/'RECIPIENT_ROWS.csv').open(encoding='utf-8')))
    cases=sorted({(r['panel'],r['source_seed'],r['source_atom'],r['condition']) for r in j['cases']},key=lambda x:(x[0]!='original',x[1],x[2],x[3]=='negative'))
    assert len(cases)==(5 if fresh else 6)
    methods=['readout_top16','long128_top16','long128_target' if five else 'long32_top16','raw']
    names=['Short k128: 16 terms','Long k128: 16 terms','Long k128: full' if five else 'Long k32: 16 terms','Raw predictor (rank1)']
    colors=[*OKABE_ITO_ON_WHITE[:3],OKABE_ITO_ON_WHITE[4]]
    metrics=['normalized_kl_error','normalized_nll_delta_squared_error']
    labels=['KL error / source effect','Observed-token NLL change: squared error']
    selected=[r for r in rows if r['method'] in methods and r['scope']=='intervention_positions']
    values=[float(r[m]) for r in selected for m in metrics]
    assert all(math.isfinite(v) and v>0 for v in values),'Explicitly handle nonpositive/missing before plotting'
    lo=math.floor(math.log10(min(values)));hi=max(0,math.ceil(math.log10(max(values))))
    canvas=Image.new('RGB',(2400,1330),'white');draw=ImageDraw.Draw(canvas);points=[]
    def text(x,y,value,size=27,anchor=None,bold=False):
        font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
        draw.text((x,y),value,font=font,fill='#202020',anchor=anchor)
    def marker(x,y,index,r=7,open_=False):
        color=colors[index];fill='white' if open_ else color
        if index==0:draw.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=color,width=2)
        elif index==1:draw.rectangle((x-r,y-r,x+r,y+r),fill=fill,outline=color,width=2)
        elif index==2:draw.polygon([(x,y-r-2),(x-r-2,y+r),(x+r+2,y+r)],fill=fill,outline=color,width=2)
        else:draw.line((x-r,y-r,x+r,y+r),fill=color,width=3);draw.line((x-r,y+r,x+r,y-r),fill=color,width=3)
    text(1200,35,'Recipient training changes compact readout fidelity',45,'ma',True)
    text(1200,98,('Five matched long-k128 recipient seeds | Fixed short-source interface | New seeds reported separately' if five else 'Frozen coefficients and supports | Fresh-document confirmation | No refit' if fresh else 'Fixed source / paired fit / donor / dose | Exposed development, not independent confirmation'),29,'ma')
    for k,name in enumerate(names):
        x=130+600*k;marker(x,170,k);text(x+23,170,name,27,'lm')
    for col,(metric,label) in enumerate(zip(metrics,labels)):
        gx=315+1185*col;top=310;gw=845;gh=690
        X=lambda v:gx+(math.log10(v)-lo)/(hi-lo)*gw
        text(gx+gw/2,244,label,30,'ma',True)
        for exp in range(lo,hi+1):
            x=X(10**exp);draw.line((x,top-24,x,top+gh),fill='#DDDDDD',width=2);text(x,top+gh+27,f'1e{exp}',25,'ma')
        for yy in range(top-24,top+gh,24):draw.line((X(1),yy,X(1),min(yy+12,top+gh)),fill='#555555',width=2)
        for n,(panel,s,a,condition) in enumerate(cases):
            y=top+48+(132 if fresh else 110)*n
            text(gx-23,y-13,f"{s}:{a} {'+' if condition=='positive' else '-'}",28,'rm',True)
            targetset={int(r['target_seed']) for r in selected if (r['panel'],int(r['source_seed']),int(r['source_atom']),r['condition'])==(panel,s,a,condition)}
            text(gx-23,y+23,f"{'Original' if panel=='original' else 'Expanded'} / n={len(targetset)}",24,'rm')
            for k,method in enumerate(methods):
                rr=[r for r in selected if (r['panel'],int(r['source_seed']),int(r['source_atom']),r['condition'],r['method'])==(panel,s,a,condition,method)]
                vv=[float(r[metric]) for r in rr];med=median(vv)
                saved=next(c for c in j['cases'] if (c['panel'],c['source_seed'],c['source_atom'],c['condition'],c['method'],c['scope'])==(panel,s,a,condition,method,'intervention_positions'))
                assert med==saved[metric]
                yy=y+(k-1.5)*19;draw.line((X(min(vv)),yy,X(max(vv)),yy),fill=colors[k],width=2)
                for r in rr:
                    value=float(r[metric]);marker(X(value),yy,k,r=4,open_=True)
                    points.append(dict(panel=panel,source_seed=s,source_atom=a,condition=condition,method=method,metric=metric,target_seed=int(r['target_seed']),value=value,kind='target'))
                marker(X(med),yy,k)
                points.append(dict(panel=panel,source_seed=s,source_atom=a,condition=condition,method=method,metric=metric,target_seed=None,value=med,kind='case_median'))
        draw.line((gx,top+gh,gx+gw,top+gh),fill='#202020',width=2)
    footer=[
        'Lower is better; log10 axes. Dashed line: no intervention (1). Markers: case median; thin spans: target range, NOT confidence intervals.',
        ('5 cases / 3 active queries / 20 dependent target-case pairs. Eight requests: 2 unmatched and 1 source-rejected. Added seeds use exposed documents.' if five else '5 evaluated cases / 3 active queries / 8 dependent target-case pairs. All 8 requests retained: 2 unmatched and 1 source-rejected, not zero errors.' if fresh else 'All 6 eligible cases / 4 source queries / 10 dependent target-case pairs. Five original cases excluded before target encoding for training overlap.'),
        'Same source dose, not equal candidate energy. Long configurations: 4.19M tokens; different training streams, not a nested learning curve.',
        'Full and single-atom results, both endpoint scopes, exclusions and all target values: RECIPIENT_COMPARISON.md / RECIPIENT_ROWS.csv.',
        'Provisional internal figure. No optimal-sparsity, independent-mechanism, semantic-uniqueness or journal-compliance claim.']
    for n,line in enumerate(footer):text(1200,1095+n*40,line,25,'ma')
    canvas.save(out,dpi=(300,300))
    with (run/'RECIPIENT_FIGURE_DATA_v2.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    write(run/'recipient_figure_manifest_v2.json',dict(path=str(out),sha256=sha256(out),pixels=list(canvas.size),mode=canvas.mode,dpi=300,
        pillow=pillow_version,python=sys.executable,generator_sha256=sha256(Path(__file__)),source_summary_sha256=sha256(path),
        points=len(points),case_median_cells=len(cases)*8,log10_limits=[lo,hi],nonpositive_missing='none among evaluated values; three unevaluated requests disclosed in caption' if fresh else 'none; script fails rather than silently dropping',
        uncertainty='none; spans show observed dependent targets, not CI',palette_asset_sha256=sha256(SKILL/'assets/color_palettes.py'),
        alt_text=('Two log-scale panels show all five evaluated cases across five matched long-k128 recipient seeds, excluding the source index in each case. Sixteen-term readouts have lower median KL and NLL errors than short sixteen-term readouts in every case. Full and raw remain stronger alternatives. Three of eight requests are unevaluated; added seeds use previously exposed documents; ranges are not confidence intervals.' if five else 'Two log-scale panels show all five evaluated fresh-document cases. Both long-trained sixteen-term readouts have lower median KL and NLL errors than short sixteen-term readout in each case; raw remains strong. Three of eight requested cases are not evaluated, not zero. Ranges show dependent targets, not confidence intervals.' if fresh else 'Two log-scale panels show all six cases. Both long-trained sixteen-term readouts reduce KL error versus short sixteen-term readout in every case; the original1230positive case has worse NLL with long k128. Full raw remains a strong comparison.')))
    print(json.dumps(dict(path=str(out),points=len(points),log10_limits=[lo,hi])))


if __name__=='__main__':main()
