"""Export the source-coverage experiment and frozen transfer confirmation."""
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/correspondence_reform_20260913';PAPER=ROOT/'paper'
CONDITIONS=['same_answer_opposite_carry','same_carry_different_answer']


def main():
    names=['mixed_source_two','mixed_source_three','arity_source_two','arity_source_three',
           'code_source_two','code_source_three','frozen_source','frozen_targets']
    data={n:json.loads((ART/f'r51_{n}.json').read_text()) for n in names}
    costs=[]
    for run in dict.fromkeys(d['run'] for d in data.values()):
        p=ROOT/run;s=json.loads((p/'metrics.summary.json').read_text());assert s['status']=='PASS'
        raw=hashlib.sha256((p/'metrics.raw.jsonl').read_bytes()).hexdigest()
        assert all(d['raw_sha256']==raw for d in data.values() if d['run']==run)
        costs.append(dict(run=run,**{k:s[k] for k in ['wall_seconds','process_cpu_seconds','peak_allocated_bytes','rows','sequence_forwards','token_forwards']},
                          events=json.loads((p/'status.json').read_text())))
    def cell(panel,method,condition):return next(c for c in data[panel]['cells'] if c['method']==method and c['condition']==condition)
    def values(panel,method):return [100*cell(panel,method,c)['success']['mean'] for c in CONDITIONS]
    # Check duplicated comparator executions across all three development runs.
    replay=[]
    for arity in ['two','three']:
        for name in ['arity_source','code_source']:
            for method in ['noop','raw_carry_direction','function_ce_binary_64']:
                assert values('mixed_source_'+arity,method)==values(name+'_'+arity,method)
                replay.append([arity,name,method])
    development=[('No edit','mixed','noop'),('Raw, two-only','mixed','raw_carry_direction'),
        ('Raw, mixed','mixed','raw_mixed_direction'),('Two-only CE, binary','mixed','function_ce_binary_64'),
        ('Mixed CE, binary','mixed','mixed_ce_binary_64'),('Mixed CE, weighted','mixed','mixed_ce_weighted_64'),
        ('Arity member weights','arity','arity_members_64'),('Code range, scalar','code','code_scalar_64'),
        ('Code range, member weights','code','code_members_64')]
    table=['\\begin{anchoredtable}\\centering\\small','\\begin{tabular}{lrrrr}','\\toprule',
           '& \\multicolumn{2}{c}{Two operands} & \\multicolumn{2}{c}{Three operands} \\\\',
           'Source learner & Change & Preserve & Change & Preserve \\\\','\\midrule']
    csvrows=[]
    for label,prefix,method in development:
        v=values(prefix+'_source_two',method)+values(prefix+'_source_three',method)
        table.append(label+' & '+' & '.join(f'{x:.2f}' for x in v)+' \\\\')
        csvrows.append(dict(panel='development',method=method,two_change=v[0],two_preserve=v[1],three_change=v[2],three_preserve=v[3]))
    table += ['\\bottomrule\\end{tabular}',
      '\\caption{Source training coverage and code range. Complete rule success and preservation (percent) on the common development panel. Source learners use 405 questions, 256 backward batches, and at most 64 members. Mixed training includes 277 two-operand and 128 three-operand questions. The arity-scalar interpolation learner exactly reproduces the mixed weighted row; its two scalars equal 1. Code-range learners permit gains up to 4 and clamp final codes to nonnegative values. Only the first coverage comparison was frozen before this panel was exposed; later changes are development.}',
      '\\label{tab:carry_source_coverage}\\end{anchoredtable}']
    (PAPER/'tables/arithmetic_source_coverage.tex').write_text('\n'.join(table)+'\n')
    table=['\\begin{anchoredtable}\\centering\\small','\\begin{tabular}{llrr}','\\toprule',
           'Dictionary & Method & Change & Preserve \\\\','\\midrule']
    groups=[('Source','frozen_source',[('No edit','noop'),('Raw, two-only','raw_carry_direction'),('Raw, mixed','raw_mixed_direction'),
        ('Two-only CE, binary','function_ce_binary_64'),('Mixed interpolation','arity_scalar_64'),('Native code range','code_scalar_64')]),
        ('Targets 2--5','frozen_targets',[('Raw, mixed','raw_mixed_direction'),('Assignment','assignment_64'),
        ('Direct raw-field fit','direct_code_64'),('Source-field translation','translated_code_64')])]
    for group,panel,methods in groups:
        for label,method in methods:
            v=values(panel,method);table.append(group+' & '+label+' & '+' & '.join(f'{x:.2f}' for x in v)+' \\\\')
            csvrows.append(dict(panel=panel,method=method,three_change=v[0],three_preserve=v[1]))
        table.append('\\midrule')
    table[-1]='\\bottomrule\\end{tabular}'
    table += ['\\caption{Frozen confirmation on 256 new three-operand prompts, excluding all 384 previous triples. Operands range from 10 to 49; the unedited model solves 203/256 prompts. The source learner is fixed before the new outputs and target fits. Target fits use 256 pairs from the same 405 source-training questions, 512 candidates, 256 field updates, and a 64-member union. Assignment inherits source gains; direct fitting uses the mixed-source raw field. These target fits use no edited target answers or output gradients.}',
              '\\label{tab:carry_range_confirmation}\\end{anchoredtable}']
    assert data['frozen_source']['baseline_prompts']==256 and data['frozen_source']['baseline_accuracy']==203/256
    (PAPER/'tables/arithmetic_carry_range_confirmation.tex').write_text('\n'.join(table)+'\n')
    with (PAPER/'data/arithmetic_source_coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['panel','method','two_change','two_preserve','three_change','three_preserve']);writer.writeheader();writer.writerows(csvrows)
    # Keep exact source/target agreement separate from correctness of the rule.
    raw=[json.loads(s) for s in (ROOT/data['frozen_source']['run']/'metrics.raw.jsonl').read_text().splitlines()]
    source={r['row_id']:r for r in raw if r['kind']=='rule_intervention' and r['seed']==1 and r['method']=='code_scalar_64'}
    outcomes=[]
    for condition in CONDITIONS:
        counts=dict(same_correct=0,same_wrong=0,source_correct_target_wrong=0,source_wrong_target_correct=0,both_wrong_different=0)
        rr=[r for r in raw if r['kind']=='rule_intervention' and r['seed']!=1 and r['method']=='translated_code_64' and r['operation']==condition]
        for r in rr:
            s=source[r['row_id']]
            key=('same_correct' if r['correct'] else 'same_wrong') if s['answer']==r['answer'] else ('source_correct_target_wrong' if s['correct'] else ('source_wrong_target_correct' if r['correct'] else 'both_wrong_different'))
            counts[key]+=1
        outcomes.append(dict(condition=condition,requests=len(rr),counts=counts,answer_agreement=(counts['same_correct']+counts['same_wrong'])/len(rr)))
    plot=[]
    for label,panel,method,reference in [('Two operands, development','code_source_two','code_scalar_64','arity_scalar_64'),
            ('Three operands, development','code_source_three','code_scalar_64','arity_scalar_64'),
            ('Three operands, confirmation','frozen_source','code_scalar_64','arity_scalar_64'),
            ('Translation minus assignment','frozen_targets','translated_code_64','assignment_64'),
            ('Translation minus direct fit','frozen_targets','translated_code_64','direct_code_64')]:
        for condition in CONDITIONS:
            c=next(c for c in data[panel]['contrasts'] if c['method']==method and c['reference']==reference and c['condition']==condition)
            plot.append(dict(label=label,panel=panel,**c))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),analyses=data,costs=costs,source_target_outcomes=outcomes,
        total_driver_seconds=sum(c['wall_seconds'] for c in costs),total_process_cpu_seconds=sum(c['process_cpu_seconds'] for c in costs),
        replay_checks=replay,plot_data=plot,scope='Source coverage and code-range learning plus independent new-question confirmation; no claim of independent seed-population replication or universally optimal raw/native method.')
    (ART/'R51_COMBINED_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (PAPER/'data/arithmetic_source_coverage_effects.json').write_text(json.dumps(plot,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.25),gridspec_kw={'width_ratios':[1.3,1]})
    for ax,subset,title in [(axes[0],plot[:6],'Source: wider code range minus interpolation'),(axes[1],plot[6:],'Targets: frozen source-field translation')]:
        labels=list(dict.fromkeys(r['label'] for r in subset))
        for ci,(condition,color,marker) in enumerate(zip(CONDITIONS,['#24634C','#775076'],['o','s'])):
            rr=[r for r in subset if r['condition']==condition];y=np.arange(len(labels))+(ci-.5)*.22
            means=np.array([r['difference_points'] for r in rr]);lo=np.array([r['interval_points'][0] for r in rr]);hi=np.array([r['interval_points'][1] for r in rr])
            ax.errorbar(means,y,xerr=np.vstack([means-lo,hi-means]),fmt=marker,color=color,ms=3.5,capsize=2,lw=.8,label='Carry change' if ci==0 else 'Preservation')
        short=[s.replace(' operands, ',' operands\n').replace('Translation minus ','') for s in labels]
        ax.set_yticks(range(len(labels)),short);ax.invert_yaxis();ax.axvline(0,color='.65',lw=.6)
        ax.set_title(title,fontsize=8,pad=9);ax.set_xlabel('Paired difference (percentage points)')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.set_xlim(-6,17)
    axes[1].legend(frameon=False,loc='center right',fontsize=7)
    fig.tight_layout(w_pad=1.5)
    for ext in ['pdf','svg','png']:fig.savefig(PAPER/f'figures/arithmetic_source_coverage.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(driver_seconds=result['total_driver_seconds'],cpu_seconds=result['total_process_cpu_seconds'],outcomes=outcomes)))


if __name__=='__main__':main()
