"""Connect source part responses with confirmed target-code execution."""
from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';A=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    x=json.loads((A/'r44_arithmetic_response_confirmation.json').read_text())
    source=json.loads((A/'r43_arithmetic_part_dose_confirmation.json').read_text())
    src={(r['method'],r['operation']):r for r in source['cells']}
    scores={(r['method'],r['dose']):r for r in x['fidelity']}
    methods=['raw_readout','activation_readout','response_code','native_code','member','assignment']
    labels=['Raw readout','Sparse readout','Response code','Euclidean code','Member relation','Assignment']
    colors=['#555555','#555555','#286956','#286956','#79506b','#888888']
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8.5,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,2.75))
        ax=fig.add_axes([.08,.25,.25,.57])
        for oi,op in enumerate(['unit','tens']):
            for part in [0,1,2]:
                for j,suffix in enumerate(['','_norm']):
                    if part==2 and j==1:continue
                    name='source_full' if part==2 else f'source_part{part}_bank0'+suffix
                    ax.plot(src[name,op]['H'],oi*3.6+part+(j-.5)*.20,'o' if j==0 else '^',color=['#555555','#286956'][j],ms=4.2,ls='none')
        ax.set(xlim=(-3,103),ylim=(6.3,-.6),xticks=[0,50,100],yticks=[0,1,2,3.6,4.6,5.6],yticklabels=['Units A','Units B','A+B','Tens A','Tens B','A+B'],xlabel='Complete source success (%)')
        ax.axhline(2.8,color='#ddd',lw=.5)
        ax.set_title('Source parts: 64 paired questions',fontsize=8.5,pad=9)
        for i,dose in enumerate(['original','_norm']):
            a=fig.add_axes([.535+i*.245,.25,.215,.57])
            for row,(m,col) in enumerate(zip(methods,colors)):
                r=scores[m,dose];v=r['balanced'];lo,hi=r['interval']
                a.plot([lo,hi],[row,row],color=col,lw=1)
                a.plot(v,row,'s' if m=='response_code' else 'o',color=col,ms=4.4,mfc='white' if m=='native_code' else col)
            a.set(xlim=(30,103),ylim=(5.6,-.6),xticks=[40,70,100],yticks=range(6),yticklabels=labels if i==0 else ['']*6,xlabel='Balanced agreement (%)',title=['Original request','Norm-matched request'][i])
            a.spines[['top','right','left']].set_visible(False);a.tick_params(axis='y',length=0,pad=4)
            a.grid(axis='x',color='#ddd',lw=.5);a.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=3)
        ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
        fig.text(.735,.965,'Target execution: 24 paired questions',ha='center',fontsize=8.5)
        fig.legend(handles=[Line2D([],[],color='#555555',marker='o',ls='none',ms=4,label='Original'),Line2D([],[],color='#286956',marker='^',ls='none',ms=4,label='Norm matched')],loc='lower left',bbox_to_anchor=(.07,.005),ncol=2,frameon=False,fontsize=8,handletextpad=.25,columnspacing=.6)
        fig.text(.515,.04,'Same five-SAE cohort; paired 95% intervals',fontsize=8)
        for ext in ['pdf','svg','png']:fig.savefig(P/f'figures/arithmetic_response_execution.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    t=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrr}',r'\toprule',r'Execution & Original & Norm matched \\',r'\midrule']
    for m,label in zip(methods,labels):
        vals=[scores[m,d] for d in ['original','_norm']]
        t.append(label+' & '+' & '.join(f"{r['balanced']:.2f} [{r['interval'][0]:.2f},{r['interval'][1]:.2f}]" for r in vals)+r' \\')
    t += [r'\bottomrule\end{tabular}',r'\caption{Frozen response-execution comparison. Balanced source-answer agreement gives equal weight to changed and unchanged source outcomes. Intervals resample 24 question clusters, retaining forms, parts, functions and all five SAE directions.}',r'\label{tab:response_execution}',r'\end{anchoredtable}']
    (P/'tables/arithmetic_response_execution.tex').write_text('\n'.join(t)+'\n')
    t=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Execution & Query & Units H & T & P & Tens H & T & P \\',r'\midrule']
    cells={(r['method'],r['query'],r['operation']):r for r in x['cells']}
    for m,label in zip(['source']+methods,['Source']+labels):
        for q,qn in [('part0_bank0','A'),('part1_bank0','B'),('full','A+B'),('part0_bank0_norm','A norm'),('part1_bank0_norm','B norm')]:
            t.append(label+' & '+qn+' & '+' & '.join(f"{cells[m,q,op][metric]:.2f}" for op in ['unit','tens'] for metric in ['H','T','P'])+r' \\')
    t += [r'\bottomrule\end{tabular}',r'\caption{Functional outcomes of every frozen response-execution request. H requires both modification T and preservation P. Each row contains 240 generations per requested digit.}',r'\label{tab:response_execution_profiles}',r'\end{anchoredtable}']
    (P/'tables/arithmetic_response_profiles.tex').write_text('\n'.join(t)+'\n')
    (P/'data/arithmetic_response_execution.json').write_text(json.dumps(x,indent=2)+'\n')
    with (P/'data/arithmetic_response_execution.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x['cells'][0]));w.writeheader();w.writerows(x['cells'])
    evidence=[run+'/'+s for run in x.get('runs',[x['run']]) for s in ['config.resolved.json','status.json','inputs.json','code_hashes.json','metrics.raw.jsonl','RESPONSE_EXECUTION.json','NATIVE_EXECUTION.json'] if (ROOT/run/s).is_file()]
    evidence += ['artifacts/correspondence_reform_20260913/R44_RESPONSE_FREEZE.json','artifacts/correspondence_reform_20260913/r44_arithmetic_response_confirmation.json','scripts/native_response_projection.py','scripts/analyze_arithmetic_response.py','scripts/arithmetic_response_execution_paper.py','paper/sections/arithmetic_role_queries_details.tex']
    claim=dict(id='response_based_part_execution',paper='Realizing source-part responses in the target code bank',result=x['primary'],evidence=evidence)
    (P/'data/arithmetic_response_execution_evidence.json').write_text(json.dumps(claim,indent=2)+'\n')
    manifest=P/'figures/FIGURE_MANIFEST.json';f=json.loads(manifest.read_text());f['outputs']=[r for r in f['outputs'] if 'arithmetic_response_execution' not in r['path']]
    for ext in ['pdf','svg','png']:
        p=P/f'figures/arithmetic_response_execution.{ext}';f['outputs'].append(dict(path=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest.write_text(json.dumps(f,indent=2)+'\n')
    print(json.dumps(x['primary']))


if __name__=='__main__':main()
