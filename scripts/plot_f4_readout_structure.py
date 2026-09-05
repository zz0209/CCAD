"""Render measured readout-budget curves using the available Pillow runtime."""
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--summary-dir',type=Path,required=True);args=parser.parse_args()
    folder=args.summary_dir;source=folder/'query.csv';rows=list(csv.DictReader(source.open()))
    methods=['readout_top1','readout_top4','readout_top16','readout_top64','target'];budgets=[1,4,16,64,3072]
    points=[r for r in rows if r['subset']=='selected' and r['method'] in methods]
    numeric=[float(r['error']) for r in points if r['error']]
    if not numeric or any(v<=0 or not math.isfinite(v) for v in numeric):
        raise ValueError('Log plot requires finite positive data; never silently drop nonpositive values')
    from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
    output=folder/'readout_structure.png'
    if output.exists():raise FileExistsError(output)
    im=Image.new('RGB',(2200,1100),'white');draw=ImageDraw.Draw(im)
    font_path='C:/Windows/Fonts/arial.ttf';bold_path='C:/Windows/Fonts/arialbd.ttf'
    def text(x,y,value,size=25,anchor='la',bold=False,fill='#202b3c'):
        draw.text((x,y),value,font=ImageFont.truetype(bold_path if bold else font_path,size),anchor=anchor,fill=fill)
    def line(coords,fill='#777777',width=2):draw.line(coords,fill=fill,width=width)
    low=math.floor(math.log10(min(numeric)));high=math.ceil(math.log10(max(max(numeric),1)))
    if high==low:high+=1
    text(1100,42,'Fixed signed readout: structure and compressibility',43,'ma',True)
    line([(110,115),(165,115)],'#888888',3);text(182,115,'Each query x condition',24,'lm')
    line([(615,115),(670,115)],'#0072B2',6);text(686,115,'Median of 8 groups',24,'lm')
    line([(1150,115),(1205,115)],'#222222',3);text(1222,115,'Zero effect reference = 1',24,'lm')
    for panel,(endpoint,title) in enumerate([('centered_logits','Centered-logit effect'),('next_state','Next-state effect')]):
        left=145+panel*1080;top=230;width=890;height=585
        X=lambda b:left+width*math.log10(b)/math.log10(3072)
        Y=lambda v:top+height*(high-math.log10(v))/(high-low)
        text(left,174,title,33,bold=True)
        for exponent in range(low,high+1):
            y=Y(10**exponent);line([(left,y),(left+width,y)],'#e3e7eb',2)
            text(left-18,y,f'1e{exponent}',24,'rm')
        for start in range(int(left),int(left+width),24):
            line([(start,Y(1)),(min(start+13,left+width),Y(1))],'#222222',2)
        values={(r['source_seed'],r['source_atom'],r['condition'],r['method']):float(r['error']) if r['error'] else None for r in points if r['endpoint']==endpoint}
        groups=sorted({k[:3] for k in values})
        for group in groups:
            last=None
            for budget,method in zip(budgets,methods):
                value=values.get(group+(method,))
                if value is None:last=None;continue
                xy=(X(budget),Y(value))
                if last:line([last,xy],'#888888',2)
                draw.ellipse((xy[0]-5,xy[1]-5,xy[0]+5,xy[1]+5),outline='#666666',width=2)
                last=xy
        medians=[statistics.median(values[g+(m,)] for g in groups if values.get(g+(m,)) is not None) for m in methods]
        coords=[(X(b),Y(v)) for b,v in zip(budgets,medians)];line(coords,'#0072B2',6)
        for x,y in coords:draw.polygon([(x,y-9),(x+9,y),(x,y+9),(x-9,y)],fill='#0072B2')
        line([(left,top),(left,top+height),(left+width,top+height)],'#202b3c',2)
        for b in budgets:
            x=X(b);line([(x,top+height),(x,top+height+8)],'#202b3c',2)
            text(x,top+height+20,'Full (3072)' if b==3072 else str(b),23,'ma')
        text(left+width/2,top+height+66,'Retained target atoms (log scale)',27,'ma')
        text(left,205,'Source-normalized squared error (log scale; lower is better)',23)
    text(1100,955,'Frozen source-only range: 25 / 64 input pairs; 8 query-condition groups; shared seed models.',27,'ma')
    text(1100,998,'Discovery-ranked truncation, no refit. Lines join measured budgets; all group curves shown, not independent CI.',25,'ma')
    im.save(output,dpi=(200,200))
    manifest={'source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),
              'pillow':pillow_version,'pixels':list(im.size),'points':len(points),'budgets':budgets,
              'log10_y_limits':[low,high],'group_unit':'query x condition; document median then dependent target median',
              'missing':'Empty cells produce line gaps; nonpositive values cause explicit failure',
              'aggregation':'All eight group curves and median; no smoothing or confidence interval',
              'scope':'Provisional research figure, no publisher compliance claim',
              'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (folder/'readout_structure.provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
