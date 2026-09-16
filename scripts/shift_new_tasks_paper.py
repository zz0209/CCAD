"""Export the frozen new-task analysis to paper tables and a part-contrast figure."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['none', 'source', 'geometry', 'geometry_gain', 'native', 'raw']
LABELS = dict(none='Unedited', source='Source explanation', geometry='Geometry',
              geometry_gain='Calibrated geometry', native='Member relation',
              raw='Source-direction readout')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--head-analysis', type=Path)
    p.add_argument('--paper', type=Path, default=ROOT/'paper')
    a = p.parse_args()
    data = json.loads(a.analysis.read_text())
    assert data['target_seeds'] == [2, 3, 4, 5]
    results = data['results']
    exports = a.paper/'data'
    tables = a.paper/'tables'
    exports.mkdir(exist_ok=True); tables.mkdir(exist_ok=True)
    export = dict(analysis_path=str(a.analysis),
                  analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(),
                  **data)
    (exports/'human_new_tasks.json').write_text(json.dumps(export, indent=2)+'\n')
    head = r'''\begin{tabular}{lrrrr}\toprule
& \multicolumn{2}{c}{Full program} & \multicolumn{2}{c}{Parts}\\
& Acc. & Worst group & Acc. & Agreement\\\midrule
'''
    rows = []
    for m in METHODS:
        r = results[m]
        acc = [100*r['full']['profession']['mean'],
               100*r['full']['worst_group']['mean'],
               100*r['parts_mean']['profession']['mean']]
        b = '--' if m == 'none' else f"{100*r['parts_mean']['balanced_source_agreement']['mean']:.2f}"
        rows.append(f'{LABELS[m]} & {acc[0]:.2f} & {acc[1]:.2f} & {acc[2]:.2f} & {b}'+r'\\')
    (tables/'human_new_task_main.tex').write_text(head+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n')
    lines = [r'\begin{tabular}{llrrrrrrrr}\toprule',
             r'& & \multicolumn{2}{c}{Full program} & \multicolumn{3}{c}{Part accuracy} & \multicolumn{3}{c}{Part mean}\\',
             r'Task pair & Realization & Acc. & WG & Pronouns & Names & Words & Acc. & WG & Agreement\\\midrule']
    records = []
    for pair in data['task_pairs']:
        pretty = pair.replace('_', '--')
        for mi, m in enumerate(METHODS):
            # Stored query x metric ordering: full/pronouns/names/words;
            # profession/worst-group/balanced agreement.
            r = data['by_pair'][pair][m]
            values = [r[0][0], r[0][1], r[1][0], r[2][0], r[3][0], sum(x[0] for x in r[1:])/3,
                      sum(x[1] for x in r[1:])/3, sum(x[2] for x in r[1:])/3]
            formatted = [f'{100*x:.2f}' for x in values]
            if m == 'none': formatted[-1] = '--'
            lines.append(('\\textit{'+pretty+'}' if mi == 0 else '')+' & '+LABELS[m]+' & '+' & '.join(formatted)+r'\\')
            records.append(dict(pair=pair, method=m, **dict(zip(
                ['full_accuracy','full_worst_group','pronoun_accuracy','name_accuracy',
                 'word_accuracy','part_accuracy','part_worst_group','part_agreement'], [100*x for x in values]))))
        lines.append(r'\addlinespace[4pt]')
    lines.append(r'\bottomrule\end{tabular}')
    (tables/'human_new_task_pairs.tex').write_text('\n'.join(lines)+'\n')
    with (exports/'human_new_tasks_plot.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    if a.head_analysis:
        heads = json.loads(a.head_analysis.read_text())
        assert heads['primary_replay_max_logit_error'] == 0
        assert not heads['undefined_balanced_strata']
        (exports/'human_new_task_head_sensitivity.json').write_text(json.dumps(dict(
            analysis_path=str(a.head_analysis),
            analysis_sha256=hashlib.sha256(a.head_analysis.read_bytes()).hexdigest(), **heads), indent=2)+'\n')
        lines = [r'\begin{tabular}{llrrrr}\toprule',
                 r'Epochs & Realization & Full accuracy & Part accuracy & Part worst group & Part agreement\\\midrule']
        for epoch in heads['epochs']:
            for i,m in enumerate(METHODS):
                h = heads['results'][str(epoch)][m]
                values = []
                for q,k in [('full','accuracy'),('parts_mean','accuracy'),
                            ('parts_mean','worst_group'),('parts_mean','balanced_source_agreement')]:
                    v = h[q][k]
                    values.append('--' if k=='balanced_source_agreement' and m=='none' else
                                  f"{100*v['mean']:.2f} [{100*v['range'][0]:.2f},{100*v['range'][1]:.2f}]")
                lines.append((str(epoch) if i==0 else '')+' & '+LABELS[m]+' & '+' & '.join(values)+r'\\')
            lines.append(r'\addlinespace[4pt]')
        lines.append(r'\bottomrule\end{tabular}')
        (tables/'human_new_task_head_sensitivity.tex').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
        'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5})
    order = METHODS[1:]
    colors = {'source':'#202020','geometry':'#9a9a9a','geometry_gain':'#626262',
              'native':'#216b57','raw':'#78517b'}
    markers = {'source':'D','geometry':'o','geometry_gain':'^','native':'s','raw':'v'}
    contrasts = data['pronouns_minus_names']
    bounds = [100*x for m in order for pair in data['task_pairs']
              for x in contrasts[m]['by_pair'][pair]['profession']['ci95']]
    lower = 5*math.floor(min(0,min(bounds)-2)/5)
    upper = 5*math.ceil(max(0,max(bounds)+2)/5)
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.20), sharey=True)
    fig.subplots_adjust(left=.195,right=.985,bottom=.27,top=.81,wspace=.16)
    for ax, pair in zip(axes, data['task_pairs']):
        for i, m in enumerate(order):
            r = contrasts[m]['by_pair'][pair]['profession']
            value, (lo,hi) = 100*r['mean'], [100*x for x in r['ci95']]
            ax.errorbar(value,4-i,xerr=[[value-lo],[hi-value]],marker=markers[m],
                color=colors[m],ms=3.8,lw=.75,capsize=1.8,linestyle='none')
        ax.axvline(0,color='.65',lw=.65,ls=':')
        ax.set_xlim(lower,upper); ax.set_ylim(-.5,4.5)
        ax.set_title(pair.replace('_','\n'),fontsize=8.8,pad=7)
        ax.set_yticks(range(4,-1,-1),[LABELS[m] for m in order],fontsize=8)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=6)
        ax.tick_params(axis='x',length=3,labelsize=8)
        ax.set_xticks([v for v in range(10*math.ceil(lower/10),10*math.floor(upper/10)+1,10)
                       if lower < v < upper])
    fig.text(.60,.10,'Pronoun deletion − name deletion (accuracy points)',ha='center',fontsize=8.5)
    fig.text(.60,.02,'Four held-out task pairs; paired 95% intervals over documents and target dictionaries.',ha='center',fontsize=8)
    out = a.paper/'figures/human_new_task_parts'
    for ext in ['pdf','svg','png']: fig.savefig(out.with_suffix('.'+ext),dpi=240)
    plt.close(fig)

    # Put the primary use result beside the old, interpretable part judgment.
    # The per-task plot above remains an export; the paper's complete table
    # retains every pair and method.
    fig = plt.figure(figsize=(6.75, 2.35))
    left = fig.add_axes([.205, .235, .285, .61])
    right = fig.add_axes([.755, .235, .23, .61])
    primary = data['contrasts']['native minus geometry_gain']
    endpoints = [('Full accuracy', 'full', 'profession'),
                 ('Part accuracy', 'parts_mean', 'profession'),
                 ('Part worst group', 'parts_mean', 'worst_group'),
                 ('Part agreement', 'parts_mean', 'balanced_source_agreement')]
    primary_records = []
    for i, (label, query, metric) in enumerate(endpoints):
        r = primary[query][metric]
        value, (lo,hi) = 100*r['mean'], [100*x for x in r['ci95']]
        c = '#626262' if query == 'full' else colors['native']
        left.errorbar(value,3-i,xerr=[[value-lo],[hi-value]],marker='s',
                      color=c,ms=3.7,lw=.85,capsize=2,linestyle='none')
        left.text(value, 3-i+.20, f'{value:+.2f}', ha='center', va='bottom',
                  fontsize=7.6,color=c)
        primary_records.append(dict(panel='primary',label=label,mean=value,lo=lo,hi=hi))
    left.set_yticks(range(3,-1,-1),[x[0] for x in endpoints])
    left.set_xlim(-2,16);left.set_ylim(-.55,3.65);left.set_xticks([0,5,10,15])
    left.set_xlabel('Relation − calibrated geometry (points)',fontsize=8,labelpad=5)
    for i,m in enumerate(order):
        r = contrasts[m]['mean']['profession']
        value, (lo,hi) = 100*r['mean'], [100*x for x in r['ci95']]
        right.errorbar(value,4-i,xerr=[[value-lo],[hi-value]],marker=markers[m],
                       color=colors[m],ms=3.8,lw=.85,capsize=2,linestyle='none')
        primary_records.append(dict(panel='part_judgment',label=LABELS[m],mean=value,lo=lo,hi=hi))
    right.set_yticks(range(4,-1,-1),[LABELS[m] for m in order])
    right.set_xlim(-2,20);right.set_ylim(-.7,4.7);right.set_xticks([0,10,20])
    right.set_xlabel('Pronouns − names (accuracy points)',fontsize=8,labelpad=5)
    for ax in [left,right]:
        ax.axvline(0,color='.65',lw=.65,ls=':')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=5,labelsize=8)
        ax.tick_params(axis='x',length=3,labelsize=8)
    fig.text(.035,.95,'(a) Similar full accuracy, different partial effects',fontsize=8.7,va='top')
    fig.text(.535,.95,'(b) Retaining a functional judgment',fontsize=8.7,va='top')
    fig.text(.5,.025,'Four new tasks, both confounding directions; paired 95% intervals over documents and target dictionaries.',
             ha='center',fontsize=8)
    current = a.paper/'figures/human_new_task_reuse'
    for ext in ['pdf','svg','png']: fig.savefig(current.with_suffix('.'+ext),dpi=240)
    plt.close(fig)
    with (exports/'human_new_task_reuse_plot.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(primary_records[0]));writer.writeheader();writer.writerows(primary_records)
    print(json.dumps(dict(source_sha256=export['analysis_sha256'],figure=str(out),table=str(tables/'human_new_task_main.tex'))))


if __name__ == '__main__': main()
