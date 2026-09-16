"""Export the fixed conditional-restoration confirmation to a figure and tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['source', 'geometry', 'geometry_gain', 'native', 'raw']
LABEL = dict(source='Source explanation', geometry='Geometry',
    geometry_gain='Calibrated geometry', native='Member relation', raw='Source-direction readout')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.analysis.read_text())
    paper = ROOT/'paper'
    export = dict(analysis_path=a.analysis.as_posix(),
        analysis_sha256=hashlib.sha256(a.analysis.read_bytes()).hexdigest(), **data)
    (paper/'data/human_restoration.json').write_text(json.dumps(export, indent=2)+'\n')
    lines = [r'\begin{tabular}{lrrr}\toprule',
        r'Realization & Response error & Final RMSE & Agreement (\%)\\\midrule']
    for m in ORDER:
        row = data['summary'][m]
        lines.append(LABEL[m]+' & '+f"{row['relative_restoration_rmse']['value']:.3f} & {row['final_logit_rmse']['value']:.3f} & {100*row['balanced_agreement']['value']:.2f}"+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    (paper/'tables/human_restoration_summary.tex').write_text('\n'.join(lines)+'\n')
    profiles = [p for p in data['profiles'] if p['early'] == 'names' and p['cut'] == 2]
    parts = ['pronouns', 'names', 'associated_words']
    profiles = [next(p for p in profiles if p['restored'] == s) for s in parts]
    records = []
    for p in data['profiles']:
        for m in ORDER:
            v = p['methods'][m]
            records.append(dict(request=p['request'], early=p['early'], cut=p['cut'],
                restored=p['restored'], method=m, coefficient=v['value'],
                lower95=v['ci95'][0], upper95=v['ci95'][1],
                source_changed=p['source_changed'], source_unchanged=p['source_unchanged'],
                balanced_in_fixed_family=p['balanced_in_fixed_family']))
    with (paper/'data/human_restoration_plot.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    short = dict(pronouns='Pronouns', names='Names', associated_words='Words')
    lines = [r'\begin{tabular}{llrrrrr}\toprule',
        r'Early deletion & Later restoration & Source & Geometry & Cal.\ geometry & Relation & Readout\\\midrule']
    for p in data['profiles']:
        values = [f"{p['methods'][m]['value']:.3f}" for m in ORDER]
        lines.append(short[p['early']]+f" (through {p['cut']}) & "+short[p['restored']]+' & '+' & '.join(values)+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    (paper/'tables/human_restoration_profiles.tex').write_text('\n'.join(lines)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
        'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5})
    fig = plt.figure(figsize=(6.75, 2.55))
    fig.text(.015,.97,'Delete early name members',fontsize=9,va='top')
    fig.text(.35,.97,'Restore one later part',fontsize=9,va='top')
    fig.text(.71,.97,'Measure the response',fontsize=9,va='top')
    for x1,x2 in [(.285,.33),(.62,.69)]:
        ax_arrow=fig.add_axes([0,0,1,1],frameon=False)
        ax_arrow.set_axis_off()
        ax_arrow.annotate('',xy=(x2,.94),xytext=(x1,.94),xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->',lw=.65,color='.3'))
    ax=fig.add_axes([.105,.29,.855,.505])
    colors=dict(source='#222222',geometry='#999999',geometry_gain='#626262',
                native='#216b57',raw='#78517b')
    markers=['o','v','s','D','^']
    for j,m in enumerate(ORDER):
        x=np.arange(3)+(j-2)*.12
        val=np.array([p['methods'][m]['value'] for p in profiles])
        lo=np.array([p['methods'][m]['ci95'][0] for p in profiles])
        hi=np.array([p['methods'][m]['ci95'][1] for p in profiles])
        ax.errorbar(x,val,yerr=[val-lo,hi-val],fmt=markers[j],ms=4.2,
            color=colors[m],capsize=2,lw=.8,label=LABEL[m])
    ax.axhline(0,color='.55',lw=.6)
    ax.axhline(1,color='.75',lw=.6,ls='--')
    ax.set_xticks(range(3),['Later pronouns','Later names','Later associated words'])
    ax.tick_params(axis='both',length=3)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(-.45,2.45)
    ax.set_ylim(-.12,max(1.4,max(p['methods'][m]['ci95'][1] for p in profiles for m in ORDER)+.08))
    ax.set_ylabel('Restoration coefficient',labelpad=4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,
        frameon=False,fontsize=7.7,columnspacing=1.2,handletextpad=.4)
    base=paper/'figures/human_restoration'
    for ext in ['pdf','svg','png']:
        fig.savefig(base.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    print(json.dumps(dict(analysis=export['analysis_sha256'],figure=str(base))))


if __name__ == '__main__':
    main()
