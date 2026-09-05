"""Provisional component-effect figure; saved raw points and explicit selection."""
import json
import math
import sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
from summarize_f4_probability import ROOT,sha256,write


def main():
    run=ROOT/'runs/F4_component_probability_expanded_v1_20260905'
    path=run/'component_summary.json';j=json.loads(path.read_text());output=run/'component_effects_v2.png'
    if output.exists():raise ValueError('Figure already exists')
    for p in j['inputs']:assert sha256(Path(p['path']))==p['sha256']
    image=Image.new('RGB',(2400,1400),'white');d=ImageDraw.Draw(image);points=[]
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    def text(x,y,s,size=28,anchor=None,bold=False,fill='#202020'):
        font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
        d.text((x,y),s,fill=fill,font=font,anchor=anchor)
    def mark(x,y,k):
        color=colors[k];r=7
        if k==0:d.ellipse((x-r,y-r,x+r,y+r),fill=color)
        elif k==1:d.rectangle((x-r,y-r,x+r,y+r),outline=color,width=3)
        elif k==2:d.polygon([(x,y-r-2),(x-r-2,y+r),(x+r+2,y+r)],fill=color)
        else:d.line((x-r,y-r,x+r,y+r),fill=color,width=3);d.line((x-r,y+r,x+r,y-r),fill=color,width=3)
    text(1200,40,'Fixed components contribute to real predictive effects',46,'ma',True)
    text(1200,107,'Exposed-data development | 11 / 32 source-selected conditions | five SAE seeds; dependent observations',29,'ma')
    text(70,200,'A  Removing either group usually loses fidelity',34,bold=True)
    methods=['target','readout_top16','readout_tail16','readout_random16'];names=['Full FCC','Top16 only','Remaining atoms','Random16']
    for k,n in enumerate(names):mark(100+k*265,276,k);text(119+k*265,276,n,25,'lm')
    cases=[c for c in j['cases'] if c['selected']];lo=-2;hi=0;gx=340;gy=350;gw=740;gh=616
    X=lambda v:gx+(math.log10(v)-lo)/(hi-lo)*gw
    for power in range(lo,hi+1):
        x=X(10**power);d.line((x,gy-25,x,gy+gh),fill='#DDDDDD',width=2);text(x,gy+gh+20,f'1e{power}',26,'ma')
    for n,c in enumerate(cases):
        y=gy+n*55;text(gx-15,y,f"{'O' if c['panel']=='original' else 'E'} {c['source_seed']}:{c['source_atom']} {'+' if c['condition']=='positive' else '-'}",27,'rm')
        for k,m in enumerate(methods):
            value=c['scopes']['intervention_positions'][m]['normalized_nll_delta_squared_error']
            if value is None or not 10**lo<=value<=10**hi:raise ValueError('Point outside declared figure range')
            x=X(value);yy=y+(k-1.5)*9;mark(x,yy,k)
            points.append(dict(kind='case_median',panel=c['panel'],source_seed=c['source_seed'],source_atom=c['source_atom'],condition=c['condition'],method=m,value=value,x=x,y=yy))
    d.line((gx,gy+gh,gx+gw,gy+gh),fill='#202020',width=2)
    text(700,1040,'Relative NLL-change squared error (log10); lower is better',27,'ma')
    text(70,1110,'O: original, E: expanded; each point is a four-target median.',25)
    text(70,1150,'21 unselected / unsupported requests remain in COMPONENT_TABLE.md.',25)
    text(1280,200,'B  Real responses are approximately compositional',33,bold=True)
    text(1280,248,'x = full intervention; y = head-alone + tail-alone NLL changes',25)
    vals=[]
    for r in j['compositions']:
        s=r['scopes']['intervention_positions']
        for i,(f,h,t) in enumerate(zip(s['full_nll_delta'],s['head_nll_delta'],s['tail_nll_delta'])):
            vals.append(dict(panel=r['panel'],source_seed=r['source_seed'],source_atom=r['source_atom'],condition=r['condition'],target_seed=r['target_seed'],position_index=i,actual=f,composed=h+t))
    lower=math.floor(min(0.,min(min(v['actual'],v['composed']) for v in vals))*10)/10
    upper=math.ceil(max(0.,max(max(v['actual'],v['composed']) for v in vals))*10)/10
    sx=1450;sy=360;size=610
    fx=lambda v:sx+(v-lower)/(upper-lower)*size
    fy=lambda v:sy+size-(v-lower)/(upper-lower)*size
    for i in range(6):
        v=lower+(upper-lower)*i/5;x=fx(v);y=fy(v)
        d.line((x,sy,x,sy+size),fill='#DDDDDD',width=2);d.line((sx,y,sx+size,y),fill='#DDDDDD',width=2)
        text(x,sy+size+17,f'{v:.2f}',25,'ma');text(sx-18,y,f'{v:.2f}',25,'rm')
    d.line((fx(lower),fy(lower),fx(upper),fy(upper)),fill='#202020',width=2)
    for v in vals:
        x=fx(v['actual']);y=fy(v['composed']);mark(x,y,0 if v['panel']=='original' else 3)
        points.append(dict(kind='position_response',**v,x=x,y=y))
    text(1760,1040,'Full NLL change (nats; signed intervention - baseline)',26,'ma')
    text(1280,315,'Composed NLL change (nats)',25)
    mark(1400,1100,0);text(1423,1100,f"Original: {sum(v['panel']=='original' for v in vals)} position responses",25,'lm')
    mark(1400,1142,3);text(1423,1142,f"Expanded: {sum(v['panel']=='expanded' for v in vals)} position responses",25,'lm')
    text(1200,1230,'Actual standalone responses are measured, not inferred from the hook sum. No independence, semantic uniqueness or native equivalence claim.',25,'ma')
    text(1200,1275,'Complement has more atoms; random16 is sampled once from discovery top64. Raw and all strong controls remain in the full table.',25,'ma')
    text(1200,1320,'Provisional internal figure; publisher format pending. No smoothing, jitter, fitted line or hidden outliers.',25,'ma')
    image.save(output,dpi=(300,300))
    write(run/'component_figure_points_v2.json',dict(points=points,input_sha256=sha256(path),left_log10_limits=[lo,hi],scatter_equal_xy_limits=[lower,upper]))
    write(run/'component_figure_manifest_v2.json',dict(path=str(output),sha256=sha256(output),summary_sha256=sha256(path),generator_script_sha256=sha256(Path(__file__)),
        pixels=list(image.size),mode='RGB',requested_dpi=300,actual_dpi=Image.open(output).info.get('dpi'),python=sys.executable,pillow=pillow_version,points=len(points),
        selection='All11source-selectedcases;21otherrequests in linked table',uncertainty='no intervals; shared seed/document/query dependent',
        alt_text=f'Left: all eleven selected cases show full, top16, complement and random16 NLL effect errors. Right: all{len(vals)} actual position responses compare full intervention NLL changes to summed separately measured group changes, with identity line and visible deviations. Original and expanded panels use circle/cross markers. Full controls, missing conditions and nonlinearity statistics are in linked tables.',publisher='pending; internal'))
    print(json.dumps(dict(path=str(output),points=len(points),scatter_xy_limits=[lower,upper],actual_dpi=Image.open(output).info.get('dpi'))))


if __name__=='__main__':main()
