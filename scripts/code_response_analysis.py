"""Summarize source-profile fidelity from actual R32 intervention rows."""
import argparse,json,hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    source=a.run/'metrics.raw.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines()]
    cfg=json.loads((a.run/'config.resolved.json').read_text());ops=list(cfg['operations'])
    index={(r['objective'],r['seed'],r['operation'],r['method'],r['task'],r['row_id']):r for r in rows}
    assert len(index)==len(rows)
    cells=[];overall=[]
    for obj in cfg['objectives']:
        for method in dict.fromkeys(r['method'] for r in rows):
            for s in cfg['seeds']:
                rr=[r for r in rows if (r['objective'],r['method'],r['seed'])==(obj,method,s)]
                correct=[];wrong=[];none=[]
                for r in rr:
                    key=(obj,s,r['operation'],'source',r['task'],r['row_id']);src=index[key]
                    correct.append(abs(r['margin']-src['margin']));none.append(abs(r['clean_margin']-src['margin']))
                    for other in ops:
                        if other!=r['operation']:wrong.append(abs(r['margin']-index[obj,s,other,'source',r['task'],r['row_id']]['margin']))
                cells.append(dict(objective=obj,method=method,source=s,target=s+10,mae=float(np.mean(correct)),wrong_component_mae=float(np.mean(wrong)),no_edit_mae=float(np.mean(none))))
            cc=[r for r in cells if (r['objective'],r['method'])==(obj,method)]
            overall.append(dict(objective=obj,method=method,mae=float(np.mean([r['mae'] for r in cc])),
                                wrong_component_mae=float(np.mean([r['wrong_component_mae'] for r in cc])),per_pair_mae=[r['mae'] for r in cc]))
    original=json.loads((a.run/'code_response_results.json').read_text())
    for cell in overall:
        rr=[r for r in original['cells'] if (r['objective'],r['method'])==(cell['objective'],cell['method'])]
        assert abs(cell['mae']-np.mean([r['source_margin_mae'] for r in rr]))<1e-10
        cell['relative_field_mse']=float(np.mean([r['relative_field_mse'] for r in rr]))
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=str(a.run),raw_rows=len(rows),raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             scope='Development: same five disjoint source-target pairs, three previously exposed grammars; source-ID permutation is a sanity comparison, not independent evidence of semantic discovery.',cells=cells,overall=overall)
    (a.output/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    plot(out,a.output)
    print(json.dumps(dict(rows=len(rows),overall=overall)))


def plot(data,output):
    overall=data['overall'];output=Path(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    methods=['code_binary','field_binary','decoder_binary','cached_binary','code_soft','field_soft','raw_field','dynamic_reencoding']
    labels=['Encoding response: binary','Contribution fit: binary','Decoder matching: binary','Cached attribution: binary','Encoding response: soft','Contribution fit: soft','Raw field regression','Dynamic reencoding']
    fig,axs=plt.subplots(1,2,figsize=(7.0,3.15),sharey=True,layout='constrained')
    for ax,obj,title in zip(axs,['topk','matryoshka'],['TopK','Matryoshka']):
        for i,method in enumerate(methods):
            c=next(r for r in overall if (r['objective'],r['method'])==(obj,method));color='#286854' if method.startswith('code') else '#795487' if method=='cached_binary' else '#686868'
            yy=i+np.linspace(-.13,.13,5);ax.scatter(c['per_pair_mae'],yy,s=14,color=color,alpha=.65,linewidth=0)
            ax.plot(c['mae'],i,marker='|',markersize=11,color=color,markeredgewidth=1.5)
        ax.axhline(3.5,color='.75',lw=.6);ax.axhline(5.5,color='.75',lw=.6);ax.set_title(title,fontsize=10);ax.set_xlabel('Source-margin MAE (nats)');ax.set_xlim(0,.7);ax.grid(axis='x',lw=.4,alpha=.25);ax.set_axisbelow(True)
    axs[0].set_yticks(range(len(methods)),labels);axs[0].invert_yaxis()
    fig.savefig(output/'code_response_fidelity.pdf');fig.savefig(output/'code_response_fidelity.svg');fig.savefig(output/'code_response_fidelity.png',dpi=220);plt.close(fig)


if __name__=='__main__':main()
