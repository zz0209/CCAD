"""Export source-role part effects, correspondence and the concrete query example."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
PAPER=ROOT/'paper'


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,colors
    primary=json.loads((ART/'r40_member_confirmation.json').read_text())
    readouts=json.loads((ART/'r40_query_readouts.json').read_text())
    profiles=json.loads((ART/'r40_role_profiles.json').read_text())
    examples=json.loads((ART/'r40_role_examples.json').read_text())
    eligible=[]
    for r in examples:
        a,b,c,d=r['operands'];desired=(a+b)//10*10+(c+d)%10
        if (r['operation']=='unit' and r['template']==0 and r['full_answers']['source']==desired
            and r['answers']['source']==[a+b,a+b] and r['answers']['member']==r['answers']['source']
            and r['full_answers']['member']==desired):eligible.append(r)
    example=eligible[0]
    example['display_selection']='First in complete question/operation/prompt/seed order among source and target cases where both halves retain the base answer and the full operation gives the requested hybrid; descriptive illustration.'
    example['eligible_cases']=len(eligible)
    a,b,c,d=example['operands'];target=example['target_seed'];source=5 if target==1 else target-1
    with np.load(ART/'r40_member_confirmation.npz') as z:
        assert z['baseline'][example['question'],example['template']]==a+b
    full={(r['method'],r['operation']):r for r in primary['full_function']+readouts['full_function']}
    part={(r['method'],r['part'],r['operation'],r['metric']):r for r in profiles['profiles']}
    labels={'source':'Source','member':'Member relation','assignment':'Assignment','two_assignment':'Two matches',
            'raw':'Raw readout','activation':'Sparse readout','wrong':'Swapped part','full_component':'Full component','unchanged':'Unedited'}
    order=['source','member','assignment','activation','raw']
    matrix=np.array([[100*part[n,p,op,'H']['mean'] if p<2 else 100*full[n,op]['H']
                      for op in ['unit','tens'] for p in [0,1,2]] for n in order])
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
                         'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,2.5))
        ax=fig.add_axes([.015,.08,.355,.85]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        ax.text(0,.98,'A concrete units request',fontsize=10,va='top')
        ax.text(0,.84,f'Recipient: {a} + {b} = {a+b}',fontsize=9)
        ax.text(0,.74,f'Donor: {c} + {d} = {c+d}; requested answer: {(a+b)//10*10+(c+d)%10}',fontsize=9)
        ax.text(0,.59,'Request',fontsize=8.5);ax.text(.62,.59,f'Source {source}',ha='center',fontsize=8.5);ax.text(.90,.59,f'Target {target}',ha='center',fontsize=8.5)
        ax.plot([0,1],[.555,.555],color='#555555',lw=.6)
        for j,label in enumerate(['No edit','Part A','Part B','A + B']):
            y=.46-j*.115;ax.text(0,y,label,va='center')
            for x,n in [(.62,'source'),(.9,'member')]:
                v=(a+b if j==0 else example['answers'][n][j-1] if j<3 else example['full_answers'][n])
                ax.text(x,y,str(v),ha='center',va='center',color='#286956' if j==3 else '#333333',fontweight='bold' if j==3 else 'normal')
        hm=fig.add_axes([.54,.24,.445,.57])
        cmap=colors.LinearSegmentedColormap.from_list('functional_success',['#f5f5f2','#286956'])
        hm.imshow(matrix,cmap=cmap,vmin=0,vmax=100,aspect='auto')
        for i in range(len(order)):
            for j in range(6):hm.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8.5,color='white' if matrix[i,j]>55 else '#222222')
        hm.set(xticks=range(6),xticklabels=['A','B','A+B']*2,yticks=range(5),yticklabels=[labels[n] for n in order])
        hm.tick_params(length=0,pad=4,labelsize=8.5);hm.spines[:].set_visible(False)
        hm.axvline(2.5,color='white',lw=4)
        hm.text(1,-.8,'Units replacement',ha='center',fontsize=9)
        hm.text(4,-.8,'Tens replacement',ha='center',fontsize=9)
        fig.text(.76,.94,'Complete functional success (%)',ha='center',fontsize=10)
        fig.text(.54,.06,'A/B: source role-defined member halves',fontsize=8.5)
        for ext in ['pdf','svg','png']:fig.savefig(PAPER/f'figures/arithmetic_role_queries.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    cells={r['method']:r for r in primary['cells']+readouts['cells']}
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Method & Changed & Unchanged & Balanced \\',r'\midrule']
    for n in ['member','assignment','two_assignment','activation','raw','wrong','full_component','unchanged']:
        r=cells[n];table.append(f"{labels[n]} & {100*r['changed_agreement']:.2f} & {100*r['unchanged_agreement']:.2f} & {100*r['balanced_agreement']:.2f}"+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Source-answer prediction for both role-defined parts in two new prompt contexts. Values are percentages. All five fixed SAE directions and both digit requests are retained.}',r'\label{tab:role_query_agreement}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_role_queries.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lllrrr}',r'\toprule',r'Method & Request & Part & Complete & Modified & Preserved \\',r'\midrule']
    for n in ['source','member','assignment','two_assignment','activation','raw','wrong']:
        for op in ['unit','tens']:
            for p in [0,1]:
                vals=[100*part[n,p,op,k]['mean'] for k in ['H','T','P']]
                table.append(f"{labels[n]} & {op} & {'AB'[p]} & "+' & '.join(f'{x:.2f}' for x in vals)+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Complete part-response profiles. Each row includes 640 generations over 64 question clusters, two prompt forms and five SAEs.}',r'\label{tab:role_query_profiles}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_role_profiles.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Method & Request & Both needed & A suffices & B suffices & Either & Half only & None \\',r'\midrule']
    pats=['both_parts_required','part0_sufficient','part1_sufficient','either_part_sufficient','half_succeeds_full_fails','none_succeeds']
    for n in order:
        for op in ['unit','tens']:
            rows=[r for r in profiles['descriptive_coalition_patterns'] if r['method']==n and r['operation']==op]
            table.append(f'{labels[n]} & {op} & '+' & '.join(f"{100*next(r['fraction'] for r in rows if r['pattern']==p):.2f}" for p in pats)+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Descriptive response patterns across the two halves and their full union. The first four columns require full success; ``half only'' means a half succeeds while the full operation fails; ``none'' means no tested request succeeds. ``Both needed'' is relative to these two tested halves. All six patterns are retained.}',r'\label{tab:role_query_patterns}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_role_patterns.tex').write_text('\n'.join(table)+'\n')
    data=dict(primary=primary,readouts=readouts,profiles=profiles,example=example,figure_matrix=matrix.tolist(),figure_methods=order)
    dest=PAPER/'data/arithmetic_role_queries.json';dest.write_text(json.dumps(data,indent=2)+'\n')
    paths=['figures/arithmetic_role_queries.'+e for e in ['pdf','svg','png']]
    fp=PAPER/'figures/FIGURE_MANIFEST.json';fm=json.loads(fp.read_text())
    fm['outputs']=[r for r in fm['outputs'] if r['path'] not in paths]+[dict(path=p,bytes=(PAPER/p).stat().st_size,sha256=hashlib.sha256((PAPER/p).read_bytes()).hexdigest()) for p in paths]
    fm.setdefault('additional_sources',{})['arithmetic_role_queries']=dict(path=dest.relative_to(PAPER).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),generator='scripts/arithmetic_role_queries_paper.py')
    fp.write_text(json.dumps(fm,indent=2)+'\n')
    evidence=['artifacts/correspondence_reform_20260913/'+n for n in ['R40_ROLE_QUERY_FREEZE_v2.json','r40_member_confirmation.json','r40_query_readouts.json','r40_role_profiles.json','r40_role_examples.json']]
    evidence += [r['run']+'/'+name for r in [primary,readouts] for name in ['config.resolved.json','metrics.raw.jsonl','MEMBER_QUERY_FREEZE.json','status.json']]
    evidence += ['scripts/'+n for n in ['analyze_arithmetic_role_queries.py','arithmetic_role_queries_paper.py','arithmetic_member_queries.py']]
    (PAPER/'data/arithmetic_role_queries_evidence.json').write_text(json.dumps(dict(id='role_defined_member_queries',
        paper='Source-role parts and their functional cooperation in new contexts',scope=primary['scope'],
        result=primary['contrasts']+readouts['contrasts'],descriptive_patterns=profiles['coalition_scope'],evidence=evidence),indent=2)+'\n')
    print(json.dumps(dict(example=example,balanced={n:100*r['balanced_agreement'] for n,r in cells.items()})))


if __name__=='__main__':main()
