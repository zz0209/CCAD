"""Export member-resolved correspondence and unfitted query outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/correspondence_reform_20260913';PAPER=ROOT/'paper'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    q=json.loads((ART/'r38_member_queries.json').read_text())
    full=json.loads((ART/'r38_objectives.json').read_text())
    labels=[('member','Member relation'),('assignment','One-to-one assignment'),
            ('two_assignment','Two matches per member'),('wrong','Swapped part'),
            ('full_component','Full component'),('unchanged','Unedited')]
    cells={c['method']:c for c in q['cells']}
    out=PAPER/'data/arithmetic_member_queries.json'
    out.write_text(json.dumps(dict(full=full,queries=q,labels=labels),indent=2)+'\n')
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
                         'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
        fig.subplots_adjust(left=.255,right=.985,top=.86,bottom=.21,wspace=.15)
        for ax,key,title,color in zip(axes,['changed_source_agreement','unchanged_source_agreement'],
                                      ['Source answer changes','Source answer stays unchanged'],['#286956','#755882']):
            vals=[100*cells[n][key] for n,l in labels]
            ax.barh(range(len(labels)),vals,height=.52,color=color)
            for i,v in enumerate(vals):
                ax.text(v-1.8 if v>30 else v+1.8,i,f'{v:.1f}',ha='right' if v>30 else 'left',va='center',
                        color='white' if v>30 else '#222222',fontsize=8)
            ax.set(xlim=(0,100),xticks=[0,25,50,75,100],yticks=range(len(labels)),yticklabels=[l for n,l in labels],
                   xlabel='Exact source-answer agreement (%)')
            ax.set_title(title,fontsize=9,fontweight='normal')
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        axes[0].invert_yaxis()
        for ext in ['pdf','svg','png']:fig.savefig(PAPER/f'figures/arithmetic_member_queries.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    lines=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrr}',r'\toprule',
           r'Member relation versus & Difference & 95\% interval \\',r'\midrule']
    for c in q['contrasts']:
        label=dict(labels)[c['comparator']];lo,hi=c['interval_points']
        lines.append(f"{label} & {c['difference_points']:+.2f} & [{lo:.2f},{hi:.2f}]"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{\textbf{Source-response prediction for new member requests.} Differences in balanced exact-answer agreement, in percentage points. The paired bootstrap resamples 64 question clusters and retains both halves, digits, prompt forms and all five fixed SAE directions.}',r'\label{tab:arithmetic_member_queries}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_member_queries.tex').write_text('\n'.join(lines)+'\n')
    def mean(method):return 100*np.mean([c['H'] for c in full['cells'] if c['method']==method])
    text=f"For full requests, member-resolved fitting gives {mean('member_fields'):.2f}\\% complete hybrid accuracy,\ncompared with {mean('full_field'):.2f}\\% for aggregate fitting and {mean('assignment'):.2f}\\% for assignment.\n"
    for other,label in [('full_field','aggregate fitting'),('assignment','assignment')]:
        c=next(c for c in full['contrasts'] if c['reference']=='member_fields' and c['comparator']==other and c['operation']=='both')['metrics']['H']
        lo,hi=c['interval_points'];text+=f"The paired gain over {label} is {c['difference_points']:.2f} points ([{lo:.2f},{hi:.2f}]).\n"
    text+=f"\nThe source changes its answer in {q['source_changed_count']} of {q['n_source_queries']} subset interventions ({100*q['source_changed_fraction']:.2f}\\%).\n"
    text+=f"Balanced exact-answer agreement is {100*cells['member']['balanced_response_agreement']:.2f}\\% for the member relation,\n{100*cells['assignment']['balanced_response_agreement']:.2f}\\% for one-to-one assignment and {100*cells['two_assignment']['balanced_response_agreement']:.2f}\\% for the 64-member assignment control.\n"
    text+=f"Using the opposite part gives {100*cells['wrong']['balanced_response_agreement']:.2f}\\%; reusing the full operation gives {100*cells['full_component']['balanced_response_agreement']:.2f}\\%.\n"
    text+=r"Table~\ref{tab:arithmetic_member_queries} gives paired differences; Figure~\ref{fig:arithmetic_member_queries} separates changed and unchanged source responses."+'\n'
    (PAPER/'sections/arithmetic_member_findings.tex').write_text(text)
    evidence=['runs/REFORM_R38_qwen_member_fields_five_v1_20260914/metrics.raw.jsonl',
              'runs/REFORM_R38_qwen_member_queries_v1_20260914/metrics.raw.jsonl',
              'runs/REFORM_R38_qwen_member_queries_resume_v2_20260914/metrics.raw.jsonl',
              'runs/REFORM_R38_qwen_member_queries_v1_20260914/MEMBER_QUERY_FREEZE.json']
    evidence+=['artifacts/correspondence_reform_20260913/'+f for f in ['r38_member_queries.json','r38_member_queries.npz',
               'r38_objectives.json','r38_objectives.npz','R38_QUERY_ANALYSIS_SPEC.json','R38_MEMBER_FIELD_CHECK.json',
               'R38_QUERY_EXAMPLE.json','R38_QUERY_MASK_CHECK.json','R38_FOCUSED_CHECK.json']]
    evidence+=['scripts/'+f for f in ['arithmetic_relation_transfer.py','arithmetic_member_queries.py',
               'analyze_arithmetic_member_queries.py','arithmetic_member_paper.py']]
    evidence+=['paper/'+f for f in ['data/arithmetic_member_queries.json','sections/arithmetic_member_relations.tex',
               'sections/arithmetic_member_findings.tex','tables/arithmetic_member_queries.tex','figures/arithmetic_member_queries.pdf']]
    (PAPER/'data/arithmetic_member_queries_evidence.json').write_text(json.dumps(dict(id='member_resolved_query_prediction',
        paper='Member-resolved relations and unfitted source-part queries',scope=q['analysis'],result=q['contrasts'],evidence=evidence),indent=2)+'\n')
    mf=PAPER/'figures/FIGURE_MANIFEST.json';m=json.loads(mf.read_text())
    paths=['figures/arithmetic_member_queries.'+e for e in ['pdf','svg','png']]
    m['outputs']=[x for x in m['outputs'] if x['path'] not in paths]+[dict(path=p,bytes=(PAPER/p).stat().st_size,
        sha256=hashlib.sha256((PAPER/p).read_bytes()).hexdigest()) for p in paths]
    m.setdefault('additional_sources',{})['arithmetic_member_queries']=dict(path=out.relative_to(PAPER).as_posix(),
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),generator='scripts/arithmetic_member_paper.py')
    mf.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(balanced={n:cells[n]['balanced_response_agreement'] for n,l in labels})))


if __name__=='__main__':main()
