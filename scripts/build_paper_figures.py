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
    fig = plt.figure(figsize=(7, 4.15))
    # The recipe is an operator diagram, not a discovered causal graph. Each
    # horizontal path is evaluated separately at the same recipient hook.
    recipe = fig.add_axes([.015,.655,.975,.32]); recipe.set(xlim=(0,7),ylim=(0,1.6));recipe.axis('off')
    recipe.text(0,1.56,'(a) A correspondence specifies a physical edit',va='top',fontsize=9)
    for y,kind,code,operator,edit in [
        (1.04,'Source',r'$\Delta z_{s,S}$',r'$D_{s,S}$',r'$q_s$'),
        (.43,'Target',r'$\Delta z_{t,T}$',r'$P=UW_T^{\mathsf{T}}$',r'$\hat q=\sum_j\Delta z_{t,j}p_j$')]:
        recipe.text(.10,y,kind+' change',ha='left',va='center',fontsize=8.5)
        recipe.text(1.28,y,code,ha='center',va='center',fontsize=11)
        recipe.annotate('',xy=(2.03,y),xytext=(1.62,y),arrowprops=dict(arrowstyle='->',lw=.65,color=GREY))
        recipe.text(2.68,y,operator,ha='center',va='center',fontsize=11)
        recipe.annotate('',xy=(3.59,y),xytext=(3.28,y),arrowprops=dict(arrowstyle='->',lw=.65,color=GREY))
        recipe.text(4.39,y,edit,ha='center',va='center',fontsize=11)
        recipe.annotate('',xy=(5.43,y),xytext=(5.12,y),arrowprops=dict(arrowstyle='->',lw=.65,color=GREY))
        recipe.text(6.12,y,'Add to recipient\nthen run model',ha='center',va='center',fontsize=8.5,linespacing=1.2)
    recipe.text(2.68,.71,'fit to source edits on development pairs',ha='center',va='center',fontsize=7.8,color=GREY)
    recipe.text(.1,-.005,r'Native editing substitutes $D_{t,T}$ for $P$; it is evaluated as a separate operation.',fontsize=8)
    gs = fig.add_gridspec(2, 2, height_ratios=[.8, 1.5], left=.085, right=.985, top=.61, bottom=.12, wspace=.28, hspace=.18)
    methods = ['source_teacher', 'fcc_group', 'role_swap_norm_matched']
    labels = ['Unedited', 'Source', 'FCC', 'Role swap']
    for col, ex in enumerate(data['fixed_examples']):
        color = [GREEN, PURPLE][col]
        ax = fig.add_subplot(gs[0, col]); ax.axis('off')
        ax.text(0, 1, ['(b) Temporal cue', '(c) Quoted title'][col], va='top', color=color)
        if col == 0:
            text = 'Recipient: Right now, the barber near the porter\nDonor: Back then, the barber near the porter'
        else:
            text = 'The title is “Right now”. Right now,\nthe barber near the porter\nDonor: change only the title to “Back then”.'
        ax.text(0, .67, text, va='top', fontsize=8, linespacing=1.15)
        rows = {x['method']: x for x in ex['outputs']}
        vals = [1 / (1 + np.exp(-rows[methods[0]]['baseline']['past_logodds']))]
        vals += [1 / (1 + np.exp(-rows[m]['past_logodds'])) for m in methods]
        ax = fig.add_subplot(gs[1, col])
        ax.bar(np.arange(4), vals, width=.56, color=['#c6c6c6', INK, color, 'white'],
               edgecolor=[GREY, INK, color, color], linewidth=.9, hatch=[None, None, None, '///'])
        for i, v in enumerate(vals):
            ax.text(i, v + .035, f'{v:.3f}', ha='center', fontsize=8)
        ax.set(ylim=(0, 1.12), xticks=np.arange(4), xticklabels=labels, yticks=[0,.5,1])
        if col == 0: ax.set_ylabel('Conditional past probability')
        ax.tick_params(axis='x', length=0)
    save(fig, 'role_example')

    # Effects and task accuracy use tables/intervals at their true dependency unit.
    lookup = {(x['partition'], x['factor'], x['method']): x for x in data['context_rows']}
    diagnostic = {(x['partition'], x['factor'], x['method']): x for x in data.get('diagnostic_rows', [])}
    ms = ['source_teacher','fcc_group','raw_native_units','full_code_ridge','das_style_raw_rank1','global_factor_mean']
    names = ['Source operation','Compact FCC','Raw ridge','Full-code ridge','DAS-style rank-1','Global factor mean']
    if diagnostic:
        ms.insert(2,'fcc_wrong_donor_norm_matched'); names.insert(2,'Wrong donor, same norm')
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.96), sharey=True)
    fig.subplots_adjust(left=.245, right=.985, bottom=.19, top=.85, wspace=.13)
    for col, cue in enumerate(['familiar_cue','new_cue']):
        ax=axs[col]
        for i, m in enumerate(ms):
            for j, role in enumerate(['temporal','quoted']):
                row=(diagnostic if m=='fcc_wrong_donor_norm_matched' else lookup)[(role+'/'+cue,'time',m)]
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

    # Preserve all coefficients and actual source/target IDs. A separate ink
    # palette reserves green/purple elsewhere for the contextual roles.
    fig = plt.figure(figsize=(7,7.1))
    axs=[fig.add_axes([.11,.725,.76,.225]),fig.add_axes([.11,.12,.76,.46])]
    cmap=LinearSegmentedColormap.from_list('signed',['#454550','#fffefa','#a14f62'])
    for ax, x, letter in zip(axs,data['memberships'],['a','b']):
        h=np.asarray(x['source_lift']); vmax=float(np.max(np.abs(h)))
        im=ax.imshow(h,cmap=cmap,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),aspect='auto',interpolation='nearest')
        ax.set(yticks=range(len(x['members'])),yticklabels=x['members'],
               xticks=range(h.shape[1]),xticklabels=x['source_members'],xlabel='Source member ID',ylabel='Target member ID')
        ax.tick_params(axis='both',length=0,labelsize=8)
        ax.tick_params(axis='x',labelrotation=90)
        ax.set_title(f'({letter}) {x["factor"].title()}: {h.shape[0]} target × {h.shape[1]} source members',loc='left',pad=9)
        cax=fig.add_axes([.90,ax.get_position().y0,.016,ax.get_position().height])
        cb=fig.colorbar(im,cax=cax);cb.set_label('Signed source lift $H$',fontsize=8);cb.ax.tick_params(labelsize=8)
    save(fig,'signed_memberships')

    # All six members were selected before these fixed-case outcomes. Plot the
    # signed quantities, including zeros and countervailing removal effects.
    fig,axs=plt.subplots(1,2,figsize=(7,2.62),sharey=True)
    fig.subplots_adjust(left=.085,right=.955,bottom=.21,top=.82,wspace=.36)
    members=data['display_members']
    for col,(suffix,title,xlabel) in enumerate([
        ('code_change','(a) Target code change',r'$\Delta z_{t,j}$'),
        ('removal_time_shift_loss','(b) Effect lost when the member is removed','Signed time-shift loss (nat)')]):
        ax=axs[col];ax.axvline(0,color=GREY,lw=.65)
        values=[]
        for i,m in enumerate(members):
            for j,role in enumerate(['temporal','quoted']):
                value=m[f'{role}_{suffix}'];values.append(value);y=i+[-.14,.14][j];color=[GREEN,PURPLE][j]
                ax.plot([0,value],[y,y],color=color,lw=1)
                ax.plot(value,y,marker=['o','s'][j],color=color,ms=3.7)
                ax.annotate(f'{value:+.2f}' if value else '0',xy=(value,y),xytext=(4 if value>=0 else -4,0),
                            textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=7.4,color=color)
        lo=min(0,min(values));hi=max(values);span=hi-lo
        ax.set(xlim=(lo-.18*span,hi+.23*span),yticks=range(len(members)),yticklabels=[str(m['member']) for m in members],xlabel=xlabel)
        ax.set_title(title,loc='left',fontsize=8.5,pad=9);ax.tick_params(axis='y',length=0)
    axs[0].set_ylabel('Time-group member ID');axs[0].invert_yaxis()
    fig.text(.72,.975,'● Temporal',color=GREEN,fontsize=8,va='top')
    fig.text(.87,.975,'■ Quoted',color=PURPLE,fontsize=8,va='top')
    save(fig,'member_cases')

    fig, axs=plt.subplots(1,2,figsize=(7,2.65),gridspec_kw={'width_ratios':[1,1.15]})
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.85,wspace=.42)
    q=data['training_quality']
    for metric,name,color,marker in [('fve','FVE',INK,'o'),('ce_recovered','CE recovered',GREEN,'s')]:
        means=[]
        for step in [256,1024,4096]:
            v=[100*x[metric] for x in q if x['step']==step];means.append(np.mean(v))
        axs[0].plot([256,1024,4096],means,marker=marker,color=color,label=name,ms=4)
    axs[0].set_xscale('log',base=4)
    axs[0].set(xticks=[256,1024,4096],xticklabels=['256','1,024','4,096'],xlabel='Training updates (log scale)',ylabel='Natural validation (%)',ylim=(50,100))
    axs[0].legend(frameon=False,fontsize=8,loc='upper left');axs[0].set_title('(a) Reconstruction over training',loc='left',pad=10)
    arr=np.array([[100*next(x['accuracy'] for x in data['functional_learning'] if x['partition']=='all' and x['factor']==f and x['step']==s) for s in [256,1024,4096]] for f in ['number','time','joint']])
    axs[1].imshow(arr,cmap=LinearSegmentedColormap.from_list('function',['#f5f5f3','#4c4c4c']),vmin=20,vmax=100,aspect='auto')
    for (i,j),v in np.ndenumerate(arr):axs[1].text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v>70 else INK,fontsize=9)
    axs[1].set(xticks=range(3),xticklabels=['256','1,024','4,096'],yticks=range(3),yticklabels=['Number','Time','Joint'],xlabel='Training updates')
    axs[1].tick_params(length=0);axs[1].set_title('(b) Source-operation label accuracy (%)',loc='left',pad=10)
    save(fig,'training_learning')
    if data.get('pair_completion_rows'):
        # The two consumers share an edit object but require different fitted
        # information. Show the executable decomposition and both measured costs.
        fig=plt.figure(figsize=(7,4.8))
        diagram=fig.add_axes([.025,.725,.95,.23]);diagram.set(xlim=(0,7),ylim=(0,2));diagram.axis('off')
        diagram.text(0,2,'(a) Complete a recipient contribution using its reciprocal donor',va='top',fontsize=9)
        diagram.text(.12,1.02,'Recipient + donor\nabsolute target codes',va='center',fontsize=8.2)
        for y,symbol,weight,label in [(1.30,r'$X_-=(X-\Pi X)/2$',r'$W_0$','Frozen contrast map'),(.57,r'$X_+=(X+\Pi X)/2$',r'$V$','Learn common part')]:
            diagram.annotate('',xy=(2.02,y),xytext=(1.55,1.02),arrowprops=dict(arrowstyle='->',lw=.6,color=GREY))
            diagram.text(2.92,y,symbol,ha='center',va='center',fontsize=10)
            diagram.annotate('',xy=(4.18,y),xytext=(3.80,y),arrowprops=dict(arrowstyle='->',lw=.6,color=GREY))
            diagram.text(4.50,y,weight,ha='center',va='center',fontsize=10)
            diagram.text(4.50,y-.30,label,ha='center',va='center',fontsize=7.3)
            diagram.annotate('',xy=(5.13,1.02),xytext=(4.91,y),arrowprops=dict(arrowstyle='->',lw=.6,color=GREY))
        diagram.text(5.27,1.02,'+',ha='center',va='center',fontsize=13)
        diagram.annotate('',xy=(5.62,1.02),xytext=(5.42,1.02),arrowprops=dict(arrowstyle='->',lw=.6,color=GREY))
        diagram.text(5.82,1.02,r'$U^{\mathsf{T}}$',ha='center',va='center',fontsize=10)
        diagram.annotate('',xy=(6.20,1.02),xytext=(6.02,1.02),arrowprops=dict(arrowstyle='->',lw=.6,color=GREY))
        diagram.text(6.67,1.02,'Subtract\nfrom recipient',ha='center',va='center',fontsize=8.2)
        diagram.text(.10,.04,'Contrast cancels the common path; complete removal uses both paths.',fontsize=8)
        lookup={(r['factor'],r['consumer'],r['role'],r['method']):r for r in data['pair_completion_rows']}
        groups=[('number','temporal'),('number','quoted'),('time','temporal'),('time','quoted')]
        styles=[('legacy_contrast','Original map','o','#949494',False),
                ('paired_complete','Complete-state fit','s','#323232',False),
                ('anchor_compact','Paired completion','D','#323232',True)]
        axs=[fig.add_axes([.22,.125,.31,.45]),fig.add_axes([.64,.125,.31,.45])]
        for col,consumer in enumerate(['contrast','complete_removal']):
            ax=axs[col];maxv=0
            for i,(factor,role) in enumerate(groups):
                for j,(method,label,marker,color,open_marker) in enumerate(styles):
                    row=lookup[factor,consumer,role,method];y=i+(j-1)*.19
                    ax.plot([row['min_seed_kl'],row['max_seed_kl']],[y,y],color=color,lw=.9)
                    ax.plot(row['kl'],y,marker=marker,color=color,markerfacecolor='white' if open_marker else color,ms=4)
                    maxv=max(maxv,row['max_seed_kl'])
            ax.set(ylim=(3.47,-.47),xlim=(0,maxv*1.08),yticks=range(4),
                   yticklabels=['Number · temporal','Number · quoted','Time · temporal','Time · quoted'] if col==0 else [],
                   xlabel='Source-to-candidate KL (nat)')
            ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=8)
            ax.set_title(['(b) Reciprocal contrast','(c) Complete-group removal'][col],loc='left',pad=9)
            for sep in [.5,1.5,2.5]:ax.axhline(sep,color='#eeeeeb',lw=.5,zorder=0)
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],linestyle='none',marker=m,color=c,markerfacecolor='white' if op else c,label=l,markersize=4) for _,l,m,c,op in styles]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.667),ncol=3,frameon=False,fontsize=8,columnspacing=1.7)
        save(fig,'pair_completion')
    from complete_support_paper import plot as plot_complete_support
    plot_complete_support(data.get('complete_support'),save)
    from component_paper import plot as plot_components
    plot_components(data.get('component_reuse'),save)
    from state_projection_paper import plot as plot_states
    plot_states(data.get('source_state_projection'),save)
    from toy_paper import plot as plot_toys
    plot_toys(data.get('learned_superposition'),save)
    from native_transfer_paper import plot as plot_native
    plot_native(data.get('external_native_transfer'),save)
    from training_curve_paper import plot as plot_training32
    plot_training32(data.get('training32_curve'),save)
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),font_family=family,
        font_source=str(font) if font.exists() else 'Matplotlib STIXGeneral',outputs=outputs,
        scope='Descriptive plots of retained data; intervals are source-seed ranges or dependent-direction distributions, not confidence intervals.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=len(outputs)//3,exports=len(outputs),out=str(out),font=family)))


if __name__=='__main__':
    main()
