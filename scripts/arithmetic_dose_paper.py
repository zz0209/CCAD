"""Export the confirmed effects of changing a fixed functional part's dose."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';A=ROOT/'artifacts/correspondence_reform_20260913'

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    x=json.loads((A/'r43_arithmetic_part_dose_confirmation.json').read_text())
    t=json.loads((A/'r43_arithmetic_dose_transfer.json').read_text())
    cells={(r['method'],r['operation']):r for r in x['cells']}
    con={(r['operation'],r['metric']):r for r in x['contrasts'] if r['method']=='source'}
    names=['Original','Norm matched','Fourfold','Nonnegative norm']
    suffixes=['','_norm','_scale4','_norm_nonnegative']
    colors=['#555555','#286956','#79506b','#286956'];marks=['o','^','D','s']
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8.5,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,2.6));axs=[fig.add_axes([.065,.34,.255,.53]),fig.add_axes([.385,.34,.255,.53])]
        for ax,op,title in zip(axs,['unit','tens'],['Units replacement','Tens replacement']):
            for j,(suffix,color,marker) in enumerate(zip(suffixes,colors,marks)):
                for part in [0,1]:
                    v=cells[f'source_part{part}_bank0'+suffix,op]['H']
                    ax.plot(v,part+(j-1.5)*.12,marker=marker,color=color,ms=4.2,mfc='white' if j==3 else color,ls='none')
            full=cells['source_full',op]['H'];ax.plot(full,2,'o',color='#286956',ms=5)
            ax.text(min(full,88),2.24,f'{full:.1f}',ha='center',fontsize=8)
            ax.set(xlim=(-3,103),ylim=(2.6,-.5),yticks=[0,1,2],yticklabels=['Part A','Part B','A+B'],xticks=[0,50,100],xlabel='Complete success (%)',title=title)
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=3);ax.grid(axis='x',color='#dddddd',lw=.5);ax.set_axisbelow(True)
        ax=fig.add_axes([.755,.34,.23,.53])
        for i,op in enumerate(['unit','tens']):
            for j,key in enumerate(['full_success_original_halves_fail','full_success_all_tested_halves_fail']):
                r=con[op,key];v=r['points'];lo,hi=r['interval'];y=i*2+j
                ax.plot([lo,hi],[y,y],color=colors[j],lw=1.1);ax.plot(v,y,marks[j],color=colors[j],ms=4)
                ax.text(v+2,y-.16,f'{v:.1f}',fontsize=7.5,color=colors[j])
        ax.set(xlim=(-1,75),ylim=(3.7,-.6),yticks=[.5,2.5],yticklabels=['Units','Tens'],xticks=[0,30,60],xlabel='Fraction of cases (%)',title='Joint dependence')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=3);ax.grid(axis='x',color='#ddd',lw=.5);ax.set_axisbelow(True)
        handles=[Line2D([],[],color=c,marker=m,ls='none',ms=4,mfc='white' if j==3 else c,label=n) for j,(n,c,m) in enumerate(zip(names,colors,marks))]
        fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.035,.015),ncol=2,frameon=False,fontsize=8,handletextpad=.3,columnspacing=1.3)
        fig.text(.72,.095,'Circle: original halves',fontsize=7.5)
        fig.text(.72,.035,'Triangle: all tested doses',fontsize=7.5)
        for ext in ['pdf','svg','png']:fig.savefig(P/f'figures/arithmetic_role_queries.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Part & Execution & Units: H & T & P & Tens: H & T & P \\',r'\midrule']
    for part in [0,1]:
        for suffix,name in zip(suffixes,names):
            values=[cells[f'source_part{part}_bank0'+suffix,op][metric] for op in ['unit','tens'] for metric in ['H','T','P']]
            table.append('AB'[part]+' & '+name+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    table.append('A+B & Original & '+' & '.join(f"{cells['source_full',op][metric]:.2f}" for op in ['unit','tens'] for metric in ['H','T','P'])+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Source part responses in the frozen new-context dose panel. H requires both the requested digit T and retained digit P. Every row includes 640 generations; all outcomes remain in the denominator.}',r'\label{tab:part_doses}',r'\end{anchoredtable}']
    (P/'tables/arithmetic_part_doses.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Predictor & Original & Fourfold & Norm matched \\',r'\midrule']
    labels={'member':'Member relation','assignment':'Assignment','raw_readout':'Raw readout','activation_readout':'Sparse readout'}
    for method,label in labels.items():
        vals=[next(r for r in t['fidelity'] if r['method']==method and r['dose']==d) for d in ['original','_scale4','_norm']]
        table.append(label+' & '+' & '.join(f"{r['balanced']:.2f} [{r['interval'][0]:.2f},{r['interval'][1]:.2f}]" for r in vals)+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Balanced source-answer agreement for the same original and modified-dose part requests. Paired intervals resample 64 question clusters while retaining both parts, requests, formats and the five-SAE cohort.}',r'\label{tab:dose_transfer}',r'\end{anchoredtable}']
    (P/'tables/arithmetic_dose_transfer.tex').write_text('\n'.join(table)+'\n')
    dest=P/'data/arithmetic_part_doses.json';dest.write_text(json.dumps(dict(profiles=x,transfer=t),indent=2)+'\n')
    raw=[json.loads(s) for s in (ROOT/x['run']/'metrics.raw.jsonl').read_text().splitlines()]
    groups={}
    for r in raw:
        if r['kind']!='source_patch':continue
        key=r['method'],r['operation'];groups.setdefault(key,[]).append(r)
    with (P/'data/arithmetic_part_doses_all.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','operation','n','H','T','P','mean_final_prefix_edit_norm'])
        for (m,op),rr in sorted(groups.items()):w.writerow([m,op,len(rr)]+[np.mean([r[k] for r in rr]) for k in ['exact_hybrid','target_digit_success','preserve_digit_success','edit_norm']])
    fp=P/'figures/FIGURE_MANIFEST.json';fm=json.loads(fp.read_text());paths=['figures/arithmetic_role_queries.'+e for e in ['pdf','svg','png']]
    fm['outputs']=[r for r in fm['outputs'] if r['path'] not in paths]+[dict(path=p,bytes=(P/p).stat().st_size,sha256=hashlib.sha256((P/p).read_bytes()).hexdigest()) for p in paths]
    fm.setdefault('additional_sources',{})['arithmetic_role_queries']=dict(path=dest.relative_to(P).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator='scripts/arithmetic_dose_paper.py');fp.write_text(json.dumps(fm,indent=2)+'\n')
    evidence=['artifacts/correspondence_reform_20260913/'+n for n in ['R43_PART_DOSE_FREEZE.json','r43_arithmetic_part_dose_confirmation.json','r43_arithmetic_dose_transfer.json']]
    evidence += [x['run']+'/'+n for n in ['status.json','metrics.raw.jsonl','panel.json','MEMBER_QUERY_FREEZE.json','config.resolved.json','code_hashes.json']]
    evidence += ['scripts/'+n for n in ['analyze_arithmetic_part_dose.py','analyze_arithmetic_dose_transfer.py','arithmetic_dose_paper.py']]
    pilot=json.loads((A/'r43_arithmetic_native_pilot.json').read_text())
    evidence += ['artifacts/correspondence_reform_20260913/r43_arithmetic_native_pilot.json','scripts/analyze_arithmetic_native.py']
    evidence += [pilot['run']+'/'+n for n in ['status.json','metrics.raw.jsonl','NATIVE_EXECUTION.json','config.resolved.json','code_hashes.json']]
    (P/'data/arithmetic_part_doses_evidence.json').write_text(json.dumps(dict(id='functional_part_execution_dependence',paper='Functional parts and their execution strength',result=dict(primary=t['primary'],frozen_request_fidelity=t['fidelity'],development_native=pilot),scope=x['scope'],evidence=evidence),indent=2)+'\n')
    print(json.dumps(t['primary']))
if __name__=='__main__':main()
