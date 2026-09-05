"""Replay source native development, retain all variants, export a provisional figure."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_f4_agreement_source import ROOT,margins,swap_indices,capped,source_native_group
from run_r011s1_raw_hook_asset import write_json as write,entry
from ccad.artifacts import sha256
sys.path.insert(0,'C:/Users/zz/.codex/skills/scientific-visualization/assets')
from color_palettes import OKABE_ITO_ON_WHITE


def main():
    names=['F4_agreement_source_native_v1_20260905','F4_agreement_source_adjoint_v1_20260905','F4_agreement_source_adjoint_box_v1_20260905']
    runs=[ROOT/'runs'/n for n in names];out=runs[-1]
    if (out/'source_comparison.json').exists():raise FileExistsError('Immutable comparison exists')
    origin=ROOT/'runs/F4_agreement_task_contrast_v1_20260905/metrics.raw.jsonl'
    read=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x]
    old={(r['method'],r.get('axis'),r['id']):{k:v for k,v in r.items() if k!='run_id'} for r in read(origin)}
    allrows=[];checks=[];data=[]
    decoder_asset=json.loads((ROOT/'configs/f4_agreement_relations_fit_v1.json').read_text())['inputs']['asset_manifest']
    manifest=json.loads(Path(decoder_asset['path']).read_text());dm=next(d for d in manifest['decoders'] if d['seed']==2)
    assert sha256(Path(dm['path']))==dm['sha256']
    decoder=np.asarray(np.memmap(dm['path'],dtype='<f4',mode='r',shape=tuple(dm['shape'])),dtype=float)
    for run in runs:
        rr=read(run/'metrics.raw.jsonl');base={r['id']:r for r in rr if r['method']=='baseline'}
        for r in rr:
            np.testing.assert_allclose(margins(np.array([r['logprobs']]),[r])[0],r['margins'],rtol=0,atol=1e-12)
            if r['method']!='baseline':np.testing.assert_allclose(np.array(base[r['id']]['margins'])-r['margins'],r['margin_loss'],rtol=0,atol=1e-12)
            if r['method']!='source_native_group':assert old[r['method'],r.get('axis'),r['id']]=={k:v for k,v in r.items() if k!='run_id'}
        cfg=json.loads((run/'config.resolved.json').read_text());cache=np.load(run/'development_activations.npz');f=np.load(run/'source_native_factors.npz')
        tasks=json.loads((run/'tokenized_development.json').read_text())['rows'];z=cache['codes_2'];idmap={r['id']:i for i,r in enumerate(tasks)}
        grads=np.load(run/'source_primary_gradients.npz')['gradients'] if (run/'source_primary_gradients.npz').exists() else None
        ids,g,detail=source_native_group(z,decoder,tasks,f['basis'],cfg['source_native'],grads)
        np.testing.assert_array_equal(ids,f['ids']);np.testing.assert_allclose(g,f['g'],rtol=1e-10,atol=1e-12)
        for axis in ('subject','attractor'):
            donor=swap_indices(tasks,axis);natural=((z[:,ids]-z[donor][:,ids])*g)@decoder[ids]
            delta,scale=capped(natural,cache['hidden'],cfg['maximum_source_hook_fraction'])
            for r in rr:
                if r['method']=='source_native_group' and r['axis']==axis:
                    i=idmap[r['id']];np.testing.assert_allclose([r['dose_scale'],r['hook_fraction']],[scale[i],np.linalg.norm(delta[i])/np.linalg.norm(cache['hidden'][i])],rtol=1e-10,atol=1e-12)
        allrows.append(rr);checks.append(dict(run=run.name,raw_rows=len(rr),anchors=320,fit_and_dose_replayed=True))
    np.testing.assert_array_equal(np.load(runs[1]/'source_native_factors.npz')['ids'],np.load(runs[2]/'source_native_factors.npz')['ids'])
    np.testing.assert_array_equal(np.load(runs[1]/'source_primary_gradients.npz')['gradients'],np.load(runs[2]/'source_primary_gradients.npz')['gradients'])
    specs=[('Projection',2,'source_task_projection'),('Full SAE',2,'source_full_sae_swap'),('Native vector',0,'source_native_group'),('Native adjoint/clip',1,'source_native_group'),('Native adjoint/box',2,'source_native_group')]
    for label,run_index,method in specs:
        for axis in ('subject','attractor'):
            rr=[r for r in allrows[run_index] if r['method']==method and r['axis']==axis];assert len(rr)==64
            v=np.array([r['margin_loss'] for r in rr]);templates=sorted({r['template'] for r in rr})
            groups=np.array([[np.mean([r['margin_loss'][j] for r in rr if r['template']==t]) for j in range(3)] for t in templates])
            abs_groups=np.array([np.mean([abs(r['margin_loss'][0]) for r in rr if r['template']==t]) for t in templates])
            data.append(dict(label=label,run=names[run_index],method=method,axis=axis,n=64,primary_mean=float(v[:,0].mean()),past_mean=float(v[:,1].mean()),
                primary_abs_mean=float(np.abs(v[:,0]).mean()),tense_abs_mean=float(np.abs(v[:,2]).mean()),mean_hook_fraction=float(np.mean([r['hook_fraction'] for r in rr])),
                positive=int(np.sum(v[:,0]>0)),positive_templates=int(np.sum(groups[:,0]>0)),template_means=groups.tolist(),template_primary_abs=abs_groups.tolist()))
    with (out/'SOURCE_COMPARISON.csv').open('w',newline='',encoding='utf-8') as handle:
        cols=[k for k in data[0] if k not in ('template_means','template_primary_abs')];writer=csv.DictWriter(handle,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(data)
    evidence=[entry(run/name,'CCAD immutable observation','evidence') for run in runs for name in ['metrics.raw.jsonl','source_native_fit.json','source_native_factors.npz','development_activations.npz']]
    write(out/'source_comparison.json',dict(data=data,checks=checks,same_adjoint_support_and_gradients=True,inputs=evidence,generator_sha256=sha256(Path(__file__)),
        statistics='64already-developed inputs/16lexical templates from one structural family and one source seed. Template points are descriptive spread, not independent-seed uncertainty; no exclusions or heldout inference.'))
    fontpath='C:/Windows/Fonts/arial.ttf';font=lambda n:ImageFont.truetype(fontpath,n)
    im=Image.new('RGB',(1600,1040),'white');draw=ImageDraw.Draw(im)
    draw.text((40,24),'A source-native operation acquires task effect',font=font(32),fill='black')
    draw.text((40,70),'Development only: one source seed; 64 inputs / 16 lexical templates; different doses',font=font(22),fill='#333333')
    mapping={(r['label'],r['axis']):r for r in data};labels=[s[0] for s in specs]
    panels=[('A  Subject agreement (nats)','primary_mean',0,0),('B  Past agreement, not fit (nats)','past_mean',800,0),('C  Attractor absolute effect (nats)','primary_abs_mean',0,450),('D  Actual mean hook fraction','mean_hook_fraction',800,450)]
    common=np.array([mapping[label,'subject']['template_means'] for label in labels])[:,:,:2]
    for title,key,ox,oy in panels:
        top=135+oy;draw.text((40+ox,top),title,font=font(24),fill='black');left=265+ox;right=735+ox;y0=top+55
        if key in ('primary_mean','past_mean'):lo=min(0.,float(common.min()))-.025;hi=max(0.,float(common.max()))+.025
        elif key=='primary_abs_mean':lo=0.;hi=max(max(mapping[l,'attractor']['template_primary_abs']) for l in labels)*1.05
        else:lo=0.;hi=.105
        xpos=lambda v:left+(v-lo)/(hi-lo)*(right-left)
        for tick in np.linspace(lo,hi,5):
            xx=xpos(tick);draw.line((xx,y0-8,xx,y0+252),fill='#dddddd',width=1);draw.text((xx-18,y0+262),f'{tick:.2f}',font=font(17),fill='black')
        draw.line((xpos(0),y0-8,xpos(0),y0+252),fill='#777777',width=2)
        for j,label in enumerate(labels):
            y=y0+20+50*j;draw.text((40+ox,y-10),label,font=font(19),fill='black')
            axis='attractor' if key=='primary_abs_mean' else 'subject';r=mapping[label,axis]
            if key!='mean_hook_fraction':
                vals=r['template_primary_abs'] if axis=='attractor' else np.array(r['template_means'])[:,0 if key=='primary_mean' else 1]
                for k,v in enumerate(vals):
                    x=xpos(v);yy=y+((k%4)-1.5)*4;draw.ellipse((x-3,yy-3,x+3,yy+3),fill='#666666')
            mean=r[key];x=xpos(mean);draw.ellipse((x-6,y-6,x+6,y+6),fill=OKABE_ITO_ON_WHITE[0],outline='black',width=1)
            draw.text((right+12,y-9),f'{mean:.3f}',font=font(18),fill='black')
            if key=='mean_hook_fraction':
                x=xpos(mapping[label,'attractor'][key]);draw.rectangle((x-5,y+9,x+5,y+19),fill=OKABE_ITO_ON_WHITE[1],outline='black')
        if key=='mean_hook_fraction':draw.text((left,y0+296),'Circle: subject; square: attractor',font=font(17),fill='black')
    draw.text((40,996),'Small dots: all 16 template means (no CI). Large circles: overall means. No target-seed or reserved-input results.',font=font(20),fill='black')
    figure=out/'source_native_development.png';im.save(figure,dpi=(150,150))
    write(out/'figure_provenance.json',dict(audience='provisional internal research review, not submission-ready',dimensions_px=[1600,1040],dpi=150,mode='RGB',
        font=fontpath,palette='scientific-visualization OKABE_ITO_ON_WHITE first2; direct labels and distinct markers',data_sha256=sha256(out/'source_comparison.json'),
        generator_sha256=sha256(Path(__file__)),figure_sha256=sha256(figure),uncertainty='none; descriptive template spread',missing_data='none; all inputs retained',
        alt_text='Native adjoint box recovers positive source subject and past effects, unlike vector fitting, but attractor effects remain large. Doses differ; no cross-seed or heldout claim.'))
    lines=['# Source原生组成操作：开发checkpoint','','向量拟合未产生目标作用；source逐输入主语梯度加box约束获得实际原生操作。全部旧版本保留，仍无跨seed或未见输入结论。','','|方法|主语均值|过去时均值|干扰绝对作用|主语dose|正模板|','|---|---:|---:|---:|---:|---:|']
    for label in labels:
        r=mapping[label,'subject'];a=mapping[label,'attractor'];lines.append(f"|{label}|{r['primary_mean']:.6f}|{r['past_mean']:.6f}|{a['primary_abs_mean']:.6f}|{r['mean_hook_fraction']:.6f}|{r['positive_templates']}/16|")
    lines+=['','![source native development](source_native_development.png)','','Primary参与source梯度拟合，past/tense未参与。Box与clip保持相同成员、同梯度、同ridge；box并非新算法。所有操作是有符号donor差分而非删除，剂量各自cap0.1且不放大，不能按均值比宣称等预算优势。干扰效应和时态连带变化见完整CSV；选择性不足不能隐藏。','','三个run共1344原始margin行重算、960历史锚点逐值复现；原生系数、剂量由保存codes/decoder/梯度重放。同adjoint成员和梯度完全相同。这是计算核验而非独立科学复核。没有使用reserved或audit。','','下一步冻结box source成员/系数及其实际native操作，以独立discovery拟合target对应，检验作用与连带影响。不再用人工投影作为native teacher；投影FCC和自然文本等旧路线仍保留。','']
    (out/'RESULTS_FOR_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(checks=checks,table=data,figure=str(figure)),indent=2))


if __name__=='__main__':main()
