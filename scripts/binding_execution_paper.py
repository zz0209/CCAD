"""Export the request-execution confirmation; preserve earlier binding panels."""
from pathlib import Path
import json, hashlib

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
P=ROOT/'paper'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    x=json.loads((ART/'r42_binding_native_confirmation.json').read_text())
    cells={(r['method'],r['operation']):r for r in x['macro']}
    labels={'sae64':'Source: fixed 64','country64':'Source: country 64','sae256':'Source: all 256',
        'union_member_country':'Shared interpolation','native_scalar_country':'Interpolation + scalar',
        'synthesized_code_shared_support':'Code update: same support',
        'synthesized_code_bank':'Code update: bank 256','assignment_country':'Assignment',
        'code_readout_country':'Code readout','raw_readout_country':'Raw readout',
        'target_country':'Direct target selection'}
    order=list(labels)
    run=ROOT/x['run'];panel=json.loads((run/'panel.json').read_text())
    raw=[json.loads(r) for r in (run/'metrics.raw.jsonl').read_text().splitlines()]
    lookup={(r['component'],r['task'],r['seed'],r['method'],r['operation'],r['query']):r
        for r in raw if r['kind']=='intervention'}
    eligible=[]
    operations=['first','second','both']
    for c in panel['contexts']:
        if c['split']=='fit':continue
        for seed in [1,2,3,4,5]:
            good=all(lookup[c['context'],'template0',seed,m,o,q]['correct']
                for m in ['country64','synthesized_code_bank'] for o in operations for q in [0,1])
            failure=any(not lookup[c['context'],'template0',seed,'union_member_country',o,q]['correct']
                for o in operations for q in [0,1])
            if good and failure:eligible.append((c,seed))
    c,seed=eligible[0];names=c['names'];places=c['places'];answerrows=[]
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8.5,'mathtext.fontset':'stix',
                        'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,3.05))
        ax=fig.add_axes([.015,.12,.355,.80]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
        ax.text(0,1,'One semantic request, two executions',fontsize=10,va='top')
        for j in range(2):ax.text(0,.85-j*.095,f'{names[j]}: {places[j][0]} → {places[j+2][0]}')
        ax.text(0,.58,'Request');ax.text(.59,.58,names[0],ha='center');ax.text(.9,.58,names[1],ha='center')
        ax.plot([0,1],[.54,.54],color='#555555',lw=.6)
        for i,op in enumerate(operations):
            answers=[lookup[c['context'],'template0',seed,'synthesized_code_bank',op,q]['answer'].strip() for q in [0,1]]
            old=[lookup[c['context'],'template0',seed,'union_member_country',op,q]['answer'].strip() for q in [0,1]]
            answerrows.append(dict(operation=op,code_update=answers,interpolation=old))
            y=.44-.12*i;ax.text(0,y,['First person','Second person','Both'][i])
            for j in [0,1]:ax.text([.59,.9][j],y,answers[j],ha='center',color='#286956')
        failed=next(r for r in answerrows if r['code_update']!=r['interpolation'])
        ax.text(0,.025,'Code updates give the requested pairs above.\nInterpolation ('+failed['operation']+'): '+', '.join(failed['interpolation'])+'.',fontsize=7.7)
        chart=fig.add_axes([.685,.15,.29,.76])
        for i,m in enumerate(order):
            r=cells[m,'macro'];v=100*r['complete_binding_accuracy'];lo,hi=r['interval_points']
            color='#286956' if m in ['country64','synthesized_code_bank'] else '#555555'
            chart.plot([lo,hi],[i,i],color=color,lw=1)
            chart.plot(v,i,'o' if color=='#286956' else 's',color=color,ms=3.7)
            chart.text(102,i,f'{v:.1f}',va='center',fontsize=7.6)
        chart.set(xlim=(0,104),ylim=(len(order)-.4,-.7),yticks=range(len(order)),
            yticklabels=[labels[m] for m in order],xticks=[0,50,100],xlabel='Complete binding accuracy (%)')
        chart.tick_params(axis='y',length=0,labelsize=7.8,pad=5)
        chart.spines[['top','right','left']].set_visible(False)
        chart.grid(axis='x',color='#dddddd',lw=.5);chart.set_axisbelow(True)
        for ext in ['pdf','svg','png']:fig.savefig(P/f'figures/binding_components.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Method & First & Second & Both & Macro \\',r'\midrule']
    for m in order:table.append(labels[m]+' & '+' & '.join(f"{100*cells[m,o]['complete_binding_accuracy']:.2f}" for o in ['first','second','both','macro'])+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Frozen execution confirmation: 128 new contexts, two forms and the fixed five-SAE cohort. Every answer is scored against the requested binding.}',r'\label{tab:binding_execution}',r'\end{anchoredtable}']
    (P/'tables/binding_execution.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Method & First: modify & First: preserve & Second: modify & Second: preserve \\',r'\midrule']
    for m in order:table.append(labels[m]+' & '+' & '.join(f"{100*cells[m,o][k]:.2f}" for o in ['first','second'] for k in ['requested_accuracy','preserved_accuracy'])+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Modification and preservation on the execution confirmation. Unedited model errors remain in each denominator.}',r'\label{tab:binding_execution_preserve}',r'\end{anchoredtable}']
    (P/'tables/binding_execution_preserve.tex').write_text('\n'.join(table)+'\n')
    selected=[r for r in x['execution_contrasts'] if r['operation']=='macro' and r['reference'] in
        ['union_member_country','native_scalar_country','synthesized_code_shared_support','code_readout_country','raw_readout_country','target_country']]
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrr}',r'\toprule',
        r'Code update minus reference & Difference & 95\% interval \\',r'\midrule']
    for r in selected:
        lo,hi=r['interval_points'];table.append(f"{labels[r['reference']]} & {r['difference_points']:+.2f} & [{lo:.2f},{hi:.2f}]"+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Paired differences for the frozen primary method. Context resampling retains all forms, requests and dependent SAE pairs.}',r'\label{tab:binding_execution_contrasts}',r'\end{anchoredtable}']
    (P/'tables/binding_execution_contrasts.tex').write_text('\n'.join(table)+'\n')
    example=dict(context=c,source_seed=seed,target_seed=seed%5+1,answers=answerrows,eligible=len(eligible),
        selection='First context/seed with all source and code-update requests correct and at least one shared-interpolation error in form0; descriptive, not a representative prevalence estimate.')
    dest=P/'data/binding_execution.json';dest.write_text(json.dumps(dict(confirmation=x,example=example),indent=2)+'\n')
    fm_path=P/'figures/FIGURE_MANIFEST.json';fm=json.loads(fm_path.read_text())
    paths=['figures/binding_components.'+e for e in ['pdf','svg','png']]
    fm['outputs']=[r for r in fm['outputs'] if r['path'] not in paths]+[dict(path=p,bytes=(P/p).stat().st_size,
        sha256=hashlib.sha256((P/p).read_bytes()).hexdigest()) for p in paths]
    fm.setdefault('additional_sources',{})['binding_components']=dict(path=dest.relative_to(P).as_posix(),
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator='scripts/binding_execution_paper.py')
    fm_path.write_text(json.dumps(fm,indent=2)+'\n')
    evidence=dict(id='semantic_request_native_execution',paper='Realizing source requests with nonnegative target-code changes',
        scope=x['scope'],result=selected,example=example,raw_run=x['run'],
        evidence=['artifacts/correspondence_reform_20260913/'+s for s in ['R42_CONFIRMATION_FREEZE.json','r42_binding_native_confirmation.json']],
        statistics=x['statistics'])
    evidence['evidence'] += [x['run']+'/'+s for s in ['metrics.raw.jsonl','config.resolved.json','code_hashes.json','panel.json','binding_relations.jsonl','status.json']]
    evidence['evidence'] += ['scripts/'+s for s in ['adaptive_native_execution.py','binding_correspondence.py','run_binding_components.py','analyze_binding_execution.py','binding_execution_paper.py']]
    (P/'data/binding_execution_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(example=example,contrasts=selected),indent=2))


if __name__=='__main__':main()
