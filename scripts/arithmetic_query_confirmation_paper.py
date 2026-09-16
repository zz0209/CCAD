"""Export independent member-query confirmation and its matched readout controls."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913';PAPER=ROOT/'paper'


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    d=json.loads((ART/'r39_member_confirmation.json').read_text())
    extra_path=ART/'r39_query_readouts.json'
    extra=json.loads(extra_path.read_text()) if extra_path.exists() else None
    labels={'member':'Member field','assignment':'One-to-one assignment','two_assignment':'Two matches per member',
            'wrong':'Swapped part','full_component':'Full component','unchanged':'Unedited',
            'raw':'Raw member readout','activation':'Sparse activation readout'}
    cells={c['method']:c for c in d['cells']}
    if extra:
        cells.update({c['method']:c for c in extra['cells']})
    order=['member','assignment','two_assignment']+(['raw','activation'] if extra else [])+['wrong','full_component','unchanged']
    contrasts=[c for c in d['contrasts'] if c['units']=='question_and_partition']
    if extra:
        contrasts += [c for c in extra['contrasts'] if c['units']=='question_and_partition']
    data=PAPER/'data/arithmetic_query_confirmation.json'
    data.write_text(json.dumps(dict(primary=d,readouts=extra),indent=2)+'\n')
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',
           r'Query predictor & Changed & Unchanged & Balanced \\',r'\midrule']
    for name in order:
        c=cells[name]
        table.append(f"{labels[name]} & {100*c['changed_agreement']:.2f} & {100*c['unchanged_agreement']:.2f} & {100*c['balanced_agreement']:.2f}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\caption{\textbf{New source-member queries on unused operand questions.} Exact parsed-answer agreement (\%). The balanced score weights changed and unchanged source answers equally. The fixed relation receives four new source-only partitions and 64 new question clusters; both prompts, requests and five dependent SAE directions are retained.}',r'\label{tab:arithmetic_query_confirmation}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_query_confirmation.tex').write_text('\n'.join(table)+'\n')
    func=d['full_function']+(extra['full_function'] if extra else [])
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{llrrr}',r'\toprule',
           r'Full operation & Request & Complete & Target digit & Preserved digit \\',r'\midrule']
    for c in func:
        table.append(f"{labels.get(c['method'],'Source operation')} & {'Units' if c['operation']=='unit' else 'Tens'} & {100*c['H']:.2f} & {100*c['T']:.2f} & {100*c['P']:.2f}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\caption{\textbf{Complete function transfer on the new operand panel.} These digit-replacement outcomes complement the source-response prediction test. Every unedited failure is retained.}',r'\label{tab:arithmetic_query_full}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_query_full.tex').write_text('\n'.join(table)+'\n')
    selected=['assignment','two_assignment']+(['raw','activation'] if extra else [])+['wrong','full_component']
    cmap={c['comparator']:c for c in contrasts}
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,2.25))
        axes=[fig.add_axes([.255,.24,.315,.62]),fig.add_axes([.79,.24,.19,.62])]
        groups=[selected[:-2],selected[-2:]]
        for ax,names,title,color in zip(axes,groups,
                ['(a) Correspondence and readout','(b) Query checks'],['#286956','#755882']):
            for i,name in enumerate(names):
                c=cmap[name];v=c['difference_points'];lo,hi=c['interval_points']
                ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,markersize=4,capsize=2,linewidth=1)
            ax.set(yticks=range(len(names)),yticklabels=[labels[n] for n in names],ylim=(len(names)-.6,-.4))
            ax.set_title(title,fontsize=9,pad=9,loc='left')
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        axes[0].axvline(0,color='#888888',linewidth=.7)
        axes[0].set_xlim(-8,19);axes[0].set_xticks([-5,0,5,10,15])
        axes[1].set_xlim(24,36);axes[1].set_xticks([25,30,35])
        fig.supxlabel('Member-field gain in balanced answer agreement (points)',fontsize=9,y=.04)
        for ext in ['pdf','svg','png']:
            fig.savefig(PAPER/f'figures/arithmetic_query_confirmation.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    text=f"The member relation gives {100*cells['member']['balanced_agreement']:.2f}\\% balanced source-answer agreement,\ncompared with {100*cells['assignment']['balanced_agreement']:.2f}\\% for one-to-one assignment and {100*cells['two_assignment']['balanced_agreement']:.2f}\\% for the 64-member assignment control.\n"
    for name in ['assignment','two_assignment']:
        c=cmap[name];lo,hi=c['interval_points']
        text+=f"The gain over {labels[name].lower()} is {c['difference_points']:.2f} points ([{lo:.2f},{hi:.2f}]).\n"
    text+="All four independently drawn partition banks retain a positive difference against both assignment controls.\n"
    if extra:
        text+=f"\nThe same-information raw member readout gives {100*cells['raw']['balanced_agreement']:.2f}\\%, and sparse activation prediction gives {100*cells['activation']['balanced_agreement']:.2f}\\%.\n"
        for name in ['raw','activation']:
            c=cmap[name];lo,hi=c['interval_points']
            text+=f"Member field minus {labels[name].lower()} is {c['difference_points']:+.2f} points ([{lo:.2f},{hi:.2f}]).\n"
    text+=f"\nSource interventions change {100*d['source_changed_fraction']:.2f}\\% of answers; {100*d['source_numeric_fraction']:.2f}\\% are two-digit integers.\nActual parsed values are retained when an intervention generates a different-length integer.\n"
    (PAPER/'sections/arithmetic_query_confirmation_findings.tex').write_text(text)
    evidence=[d['run']+'/'+f for f in ['metrics.raw.jsonl','config.resolved.json','MEMBER_QUERY_FREEZE.json','status.json']]
    evidence+=['artifacts/correspondence_reform_20260913/'+f for f in ['R39_CONFIRMATION_FREEZE.json','r39_member_confirmation.json','r39_member_confirmation.npz','R39_QUERY_OBJECTIVE.md','R39_QUERY_OBJECTIVE_CHECK.json']]
    evidence+=['scripts/'+f for f in ['arithmetic_member_queries.py','analyze_arithmetic_query_confirmation.py','arithmetic_query_confirmation_paper.py']]
    evidence+=['paper/'+f for f in ['sections/arithmetic_query_confirmation.tex','sections/arithmetic_query_confirmation_findings.tex','tables/arithmetic_query_confirmation.tex','tables/arithmetic_query_full.tex','figures/arithmetic_query_confirmation.pdf','data/arithmetic_query_confirmation.json']]
    if extra:
        evidence += [extra['run']+'/'+f for f in ['metrics.raw.jsonl','config.resolved.json','SOURCE_FREEZE.json','status.json']]
        evidence += ['artifacts/correspondence_reform_20260913/'+f for f in ['r39_query_readouts.json','r39_query_readouts.npz','R39_READOUT_FREEZE_v4.json','R39_READOUT_EVAL_FREEZE_v2.json','R39_READOUT_WEIGHTS_FREEZE.json','R39_READOUT_KERNEL_CHECK.json']]
        evidence += ['scripts/'+f for f in ['arithmetic_readout_queries.py','fit_arithmetic_query_readouts.py','analyze_arithmetic_query_readouts.py']]
    (PAPER/'data/arithmetic_query_confirmation_evidence.json').write_text(json.dumps(dict(id='independent_member_query_confirmation',paper='Member-query prediction on unused operands and partitions',scope=d['scope'],result=contrasts,evidence=evidence),indent=2)+'\n')
    mf=PAPER/'figures/FIGURE_MANIFEST.json';m=json.loads(mf.read_text());paths=['figures/arithmetic_query_confirmation.'+e for e in ['pdf','svg','png']]
    m['outputs']=[x for x in m['outputs'] if x['path'] not in paths]+[dict(path=p,bytes=(PAPER/p).stat().st_size,sha256=hashlib.sha256((PAPER/p).read_bytes()).hexdigest()) for p in paths]
    m.setdefault('additional_sources',{})['arithmetic_query_confirmation']=dict(path=data.relative_to(PAPER).as_posix(),sha256=hashlib.sha256(data.read_bytes()).hexdigest(),generator='scripts/arithmetic_query_confirmation_paper.py')
    mf.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(readouts_included=extra is not None,cells=cells)))


if __name__=='__main__':
    main()
