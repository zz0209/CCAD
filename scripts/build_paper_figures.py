"""Build vector manuscript figures from the exported, retained result tables."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GREEN = '#286956'
PURPLE = '#785481'
INK = '#262626'
GREY = '#777777'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=ROOT / 'paper')
    args = parser.parse_args()
    paper = args.paper.resolve()
    source = paper / 'data/figure_data.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    out = paper / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    # Times is the manuscript text family; STIX supplies compatible math glyphs.
    font = Path('C:/Windows/Fonts/times.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = 'Times New Roman'
    else:
        family = 'STIXGeneral'
    plt.rcParams.update({'font.family': family, 'font.size': 9, 'mathtext.fontset': 'stix',
        'axes.labelsize': 9, 'axes.titlesize': 9, 'axes.titleweight': 'normal',
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'lines.linewidth': 1,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'xtick.major.width': .6, 'ytick.major.width': .6, 'savefig.facecolor': 'white'})
    outputs = []
    def save(fig, name):
        for ext in ['pdf', 'svg', 'png']:
            p = out / f'{name}.{ext}'
            fig.savefig(p, dpi=220, metadata={'Creator': 'CCAD source-backed figure builder'} if ext == 'pdf' else None)
            outputs.append(dict(path=p.relative_to(paper).as_posix(), bytes=p.stat().st_size,
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        plt.close(fig)

    # A complete, fixed counterfactual pair in each role; probabilities are conditional
    # on the four stated labels, never presented as full-vocabulary probabilities.
    fig = plt.figure(figsize=(7, 3.08))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.65], left=.07, right=.98, top=.97, bottom=.18, wspace=.3, hspace=.32)
    methods = ['source_teacher', 'fcc_group', 'role_swap_norm_matched']
    labels = ['Unedited', 'Source', 'FCC', 'Role swap']
    for col, ex in enumerate(data['fixed_examples']):
        color = [GREEN, PURPLE][col]
        ax = fig.add_subplot(gs[0, col]); ax.axis('off')
        ax.text(0, 1, ['(a) Temporal cue', '(b) Quoted title'][col], va='top', fontweight='bold', color=color)
        if col == 0:
            text = 'Right now, the barber near the porter\nBack then, the barber near the porter'
        else:
            text = 'The title is “Right now”. Right now,\nthe barber near the porter\nDonor: change only the title to “Back then”.'
        ax.text(0, .68, text, va='top', fontsize=8.5, linespacing=1.45)
        rows = {x['method']: x for x in ex['outputs']}
        vals = [1 / (1 + np.exp(-rows[methods[0]]['baseline']['past_logodds']))]
        vals += [1 / (1 + np.exp(-rows[m]['past_logodds'])) for m in methods]
        ax = fig.add_subplot(gs[1, col])
        ax.bar(np.arange(4), vals, width=.56, color=['#c6c6c6', INK, color, 'white'],
               edgecolor=[GREY, INK, color, color], linewidth=.9, hatch=[None, None, None, '///'])
        for i, v in enumerate(vals):
            ax.text(i, v + .035, f'{v:.3f}', ha='center', fontsize=8)
        ax.set(ylim=(0, 1.14), xticks=np.arange(4), xticklabels=labels, yticks=[0,.5,1])
        if col == 0: ax.set_ylabel('Conditional past probability')
        ax.tick_params(axis='x', length=0)
    save(fig, 'role_example')

    # Effects and task accuracy use tables/intervals at their true dependency unit.
    lookup = {(x['partition'], x['factor'], x['method']): x for x in data['context_rows']}
    ms = ['source_teacher','fcc_group','raw_native_units','full_code_ridge','das_style_raw_rank1','global_factor_mean']
    names = ['Source operation','Compact FCC','Raw ridge','Full-code ridge','DAS-style rank-1','Global factor mean']
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.05), sharey=True)
    fig.subplots_adjust(left=.21, right=.985, bottom=.19, top=.83, wspace=.13)
    for col, cue in enumerate(['familiar_cue','new_cue']):
        ax=axs[col]
        for i, m in enumerate(ms):
            for j, role in enumerate(['temporal','quoted']):
                row=lookup[(role+'/'+cue,'time',m)]
                y=i+[-.13,.13][j]
                seeds=[s['abs_time_shift'] for s in row['source_seed_means']]
                ax.plot([min(seeds),max(seeds)],[y,y],color=[GREEN,PURPLE][j],lw=1.3)
                ax.plot(row['abs_time_shift'],y,marker=['o','s'][j],color=[GREEN,PURPLE][j],ms=4)
        ax.set(xlim=(-.1,5.7), xticks=[0,1,2,3,4,5], yticks=range(len(ms)), yticklabels=names,
               xlabel='Absolute time log-odds change (nat)')
        ax.set_title(['(a) Familiar cue pairs','(b) New cue pairs'][col],loc='left',pad=10)
        ax.grid(axis='x',color='#e8e8e8',lw=.55); ax.set_axisbelow(True); ax.tick_params(axis='y',length=0)
    axs[0].invert_yaxis()
    fig.text(.63,.975,'● Temporal',color=GREEN,fontsize=8,va='top')
    fig.text(.80,.975,'■ Quoted',color=PURPLE,fontsize=8,va='top')
    save(fig,'role_effects')

    fig, axs = plt.subplots(1,2,figsize=(7,2.92),gridspec_kw={'width_ratios':[1,1.2]})
    fig.subplots_adjust(left=.105,right=.985,bottom=.22,top=.85,wspace=.55)
    ax=axs[0]
    arrays=[]
    for factor in ['number','time']:
        for key in ['individual_energy_ratio','total_energy_ratio']:
            arrays.append([x[key] for x in data['geometry'] if x['factor']==factor])
    bp=ax.boxplot(arrays,positions=[0,1,2.5,3.5],widths=.55,patch_artist=True,showfliers=False,
                  whis=(0,100),medianprops={'color':INK,'lw':1},boxprops={'lw':.7},whiskerprops={'lw':.7},capprops={'lw':.7})
    for p in bp['boxes']:p.set_facecolor('#dedade')
    ax.axhline(1,color=GREY,ls=':',lw=.8)
    ax.set(xticks=[0,1,2.5,3.5],xticklabels=['Individual','Total','Individual','Total'],ylabel='Quoted / temporal energy')
    ax.text(.17,-.24,'Number',transform=ax.transAxes,ha='center',fontsize=8)
    ax.text(.80,-.24,'Time',transform=ax.transAxes,ha='center',fontsize=8)
    ax.set_title('(a) Member energy across 20 directions',loc='left',pad=12)
    ax=axs[1]
    mech={(x['partition'],x['factor'],x['method']):x for x in data['mechanism_rows']}
    pairs=[('temporal','number'),('temporal','time'),('quoted','number'),('quoted','time')]
    maximum=0.
    for y,(role,factor) in enumerate(pairs):
        color=GREEN if role=='temporal' else PURPLE
        a=mech[role,factor,'without_member']['kl'];b=mech[role,factor,'without_member_norm_control']['kl']
        maximum=max(maximum,a,b)
        ax.plot([a,b],[y,y],color=color,lw=1)
        ax.plot(a,y,'o',color=color,ms=4.5)
        ax.plot(b,y,'o',mfc='white',mec=color,ms=4.5)
    ax.set(yticks=range(4),yticklabels=['Temporal: number','Temporal: time','Quoted: number','Quoted: time'],
           xlim=(-.002,maximum*1.12),xticks=[0,.02,.04,.06],xlabel='KL from intact FCC (nat)')
    ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    ax.set_title('(b) Removing a member changes direction',loc='left',pad=12)
    fig.text(.63,.04,'● Member removal    ○ Equal-norm control',ha='center',fontsize=8)
    save(fig,'member_mechanism')

    fig, axs=plt.subplots(2,1,figsize=(7,7.9),gridspec_kw={'height_ratios':[1,2]})
    fig.subplots_adjust(left=.095,right=.88,bottom=.07,top=.955,hspace=.48)
    cmap=LinearSegmentedColormap.from_list('signed',['#785481','#fffefa','#286956'])
    for ax, x, letter in zip(axs,data['memberships'],['a','b']):
        h=np.asarray(x['source_lift']); vmax=float(np.max(np.abs(h)))
        im=ax.imshow(h,cmap=cmap,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),aspect='auto',interpolation='nearest')
        ax.set(yticks=range(len(x['members'])),yticklabels=x['members'],
               xticks=range(h.shape[1]),xticklabels=range(1,h.shape[1]+1),xlabel='Source member (stored support order)',ylabel='Target member ID')
        ax.tick_params(axis='both',length=0,labelsize=7)
        ax.set_title(f'({letter}) {x["factor"].title()}: {h.shape[0]} target × {h.shape[1]} source members',loc='left',pad=9)
        cax=fig.add_axes([.90,ax.get_position().y0,.016,ax.get_position().height])
        cb=fig.colorbar(im,cax=cax);cb.set_label('Signed source lift $H$',fontsize=8);cb.ax.tick_params(labelsize=7)
    save(fig,'signed_memberships')

    fig, axs=plt.subplots(1,2,figsize=(7,2.65),gridspec_kw={'width_ratios':[1,1.15]})
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.85,wspace=.42)
    q=data['training_quality']
    for metric,name,color,marker in [('fve','FVE',INK,'o'),('ce_recovered','CE recovered',GREEN,'s')]:
        means=[]
        for step in [256,1024,4096]:
            v=[100*x[metric] for x in q if x['step']==step];means.append(np.mean(v))
        axs[0].plot(range(3),means,marker=marker,color=color,label=name,ms=4)
    axs[0].set(xticks=range(3),xticklabels=['256','1,024','4,096'],xlabel='Training updates',ylabel='Natural validation (%)',ylim=(50,100))
    axs[0].legend(frameon=False,fontsize=8,loc='upper left');axs[0].set_title('(a) Reconstruction over training',loc='left',pad=10)
    arr=np.array([[100*next(x['accuracy'] for x in data['functional_learning'] if x['partition']=='all' and x['factor']==f and x['step']==s) for s in [256,1024,4096]] for f in ['number','time','joint']])
    axs[1].imshow(arr,cmap=LinearSegmentedColormap.from_list('function',['#f6f3f6',PURPLE]),vmin=20,vmax=100,aspect='auto')
    for (i,j),v in np.ndenumerate(arr):axs[1].text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v>70 else INK,fontsize=9)
    axs[1].set(xticks=range(3),xticklabels=['256','1,024','4,096'],yticks=range(3),yticklabels=['Number','Time','Joint'],xlabel='Training updates')
    axs[1].tick_params(length=0);axs[1].set_title('(b) Source-operation label accuracy (%)',loc='left',pad=10)
    save(fig,'training_learning')
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),font_family=family,
        font_source=str(font) if font.exists() else 'Matplotlib STIXGeneral',outputs=outputs,
        scope='Descriptive plots of retained data; intervals are source-seed ranges or dependent-direction distributions, not confidence intervals.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=5,exports=len(outputs),out=str(out),font=family)))


if __name__=='__main__':
    main()
