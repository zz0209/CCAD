"""Paper exports for a frozen city-family experiment; no inference or fitting."""
from __future__ import annotations
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import fmean
import numpy as np

BASE=Path('artifacts/final_five_research_20260908/r18_family_confirmation')
LABELS={'frozen_source':'Source','unedited':'Unedited','raw_supervised_family':'Raw, family selected','raw_supervised_singleton':'Raw, singleton selected',
    'raw_shared_space':'Raw, source blocks','direct_shared_space':'Decoded readout','atom_pw_mcc':'PW-MCC readout',
    'wrong_attribute_norm_matched':'Wrong attribute','target_reencode':'Re-encode',
    'adaptive_native64':'Native 64','adaptive_native256':'Native 256','adaptive_native512':'Native 512',
    'adaptive_native_reencode_count':'Native, matched count','random_base_native256':'Random + base 256'}
METHODS=list(LABELS)
CONTROLS=['100','010','001','110','101','011','111']
CONTROL_NAMES=['Country','Continent','Language','Ctry + cont','Ctry + lang','Cont + lang','All three']


def export(root,out,read):
    path=BASE/'CONFIRMATION_SUMMARY.json'
    if not (root/path).is_file():return None
    data=read(path);source=read(BASE/'SOURCE_COMPARISON.json');procedure=read(BASE/'SOURCE_PROCEDURE.json')
    data['source_comparison']=source;data['procedure']=procedure
    data['input_summary_paths']+=source['input_summary_paths']+[path.as_posix(),(BASE/'SOURCE_COMPARISON.json').as_posix(),(BASE/'SOURCE_PROCEDURE.json').as_posix()]
    data['input_summary_paths']=list(dict.fromkeys(data['input_summary_paths']))
    (out/'data/semantic_confirmation.json').write_text(json.dumps(data,indent=2)+'\n')
    def csvout(name,rows,fields):
        with (out/'data'/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    csvout('semantic_confirmation.csv',data['families'],['source_seed','target_seed','method','family','controls','cause','iso','score','source_kl'])
    csvout('semantic_confirmation_controls.csv',data['controls'],['source_seed','target_seed','method','operation','cause','iso','score','source_kl','mean_edit_norm'])
    csvout('semantic_confirmation_differences.csv',data['comparisons'],['method','comparator','family','metric','difference','conditional_city_pair_bootstrap95','leave_incident_seed_out'])
    csvout('semantic_source_replication.csv',data['source_replication'],['source_seed','target_seed','family','updates','selected_step','calibration_family','calibration_single','held_family','held_single'])
    def table(name,rows):(out/'tables'/name).write_text('\n'.join(rows)+'\n',encoding='utf-8')
    def tex(s):
        for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
        return s
    def value(v,percent=False):
        if v is None:return '--'
        return str((Decimal(str(v))*(100 if percent else 1)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
    pooled={(r['method'],r['family']):r for r in data['pooled']}
    lines=[]
    for method in METHODS:
        single=pooled[method,'single'];binary=pooled.get((method,'binary'));frac=pooled.get((method,'fractional'))
        lines.append(LABELS[method]+' & '+' & '.join(value(single[k],True) for k in ['cause','iso','score'])+' & '+value(binary['score'] if binary else None,True)+f" & {single['source_kl']:.4f} & "+('--' if frac is None else f"{frac['source_kl']:.4f}")+r' \\')
    table('semantic_confirmation_main.tex',lines)
    lines=[];direction_blocks={'readouts':[],'writers':[]}
    for method in ['frozen_source','raw_supervised_family','raw_supervised_singleton','direct_shared_space','atom_pw_mcc','target_reencode','adaptive_native_reencode_count','adaptive_native256','adaptive_native512','random_base_native256']:
        block=direction_blocks['readouts' if method in ['frozen_source','raw_supervised_family','raw_supervised_singleton','direct_shared_space','atom_pw_mcc'] else 'writers']
        for seed in range(1,6):
            rr={r['family']:r for r in data['families'] if r['method']==method and r['source_seed']==seed}
            line=(LABELS[method] if seed==1 else '')+f' & {seed} to {seed%5+1} & '+value(rr['single']['score'],True)+' & '+value(rr['binary']['score'],True)+f" & {rr['single']['source_kl']:.4f} & {rr['fractional']['source_kl']:.4f}"+r' \\'
            lines.append(line);block.append(line)
        lines.append(r'\addlinespace[3pt]');block.append(r'\addlinespace[3pt]')
    table('semantic_confirmation_directions.tex',lines)
    for name,lines in direction_blocks.items():table('semantic_confirmation_directions_'+name+'.tex',lines)
    lines=[]
    for method in METHODS:
        vals=[]
        for control in CONTROLS:
            rr=[r for r in data['controls'] if r['method']==method and r['operation']==control]
            vals.append(value(fmean(r['score'] for r in rr),True) if rr else '--')
        lines.append(LABELS[method]+' & '+' & '.join(vals)+r' \\')
    table('semantic_confirmation_controls.tex',lines)
    lines=[]
    for r in source['methods']:
        lines.append(tex(r['label'])+f" & {r['updates']} & "+value(r['calibration_family'],True)+' & '+value(r['calibration_single'],True)+' & '+value(r['held_family'],True)+' & '+value(r['held_single'],True)+r' \\')
    table('semantic_source_family_selection.tex',lines)
    lines=[]
    for r in data['source_replication']:
        lines.append(f"{r['source_seed']} & {r['updates']} & {r['selected_step']} & "+' & '.join(value(r[k],True) for k in ['calibration_family','calibration_single','held_family','held_single'])+r' \\')
    table('semantic_source_replication.tex',lines)
    lines=[]
    for r in data['comparisons']:
        if r['family']!='single' or r['metric']!='score':continue
        lo,hi=np.asarray(r['conditional_city_pair_bootstrap95'])*100;deletions=[d['difference']*100 for d in r['leave_incident_seed_out']]
        lines.append(LABELS[r['method']]+' & '+LABELS[r['comparator']]+f" & {100*r['difference']:.2f} & [{lo:.2f}, {hi:.2f}] & [{min(deletions):.2f}, {max(deletions):.2f}]"+r' \\')
    table('semantic_confirmation_comparisons.tex',lines)
    lines=[]
    for method in ['target_reencode','adaptive_native64','adaptive_native_reencode_count','adaptive_native256','adaptive_native512','random_base_native256']:
        rr=[r for r in data['native_diagnostics'] if r['method']==method];kkt='--' if method=='target_reencode' else f"{max(r['max_relative_projected_gradient'] for r in rr):.2g}"
        lines.append(LABELS[method]+f" & {fmean(r['candidate_members'] for r in rr):.1f} & {fmean(r['changed_members'] for r in rr):.1f} & {fmean(r['relative_squared_writer_error'] for r in rr):.4f} & {kkt}"+r' \\')
    table('semantic_confirmation_native.tex',lines)
    lines=[]
    for method in ['frozen_source','raw_supervised_family','raw_supervised_singleton','direct_shared_space','target_reencode','adaptive_native256','adaptive_native512']:
        for control in ['100','010','001','111']:
            words=[]
            for attr in ['Country','Continent','Language']:
                r=next(r for r in data['examples'] if r['method']==method and r['operation']==control and r['task']==attr and r['seed']==1)
                words.append(tex(r['predicted_token_text'].strip())+r' $'+(r'\checkmark' if r['first_token_correct'] else r'\times')+'$')
            lines.append((LABELS[method] if control=='100' else '')+' & '+CONTROL_NAMES[CONTROLS.index(control)]+' & '+' & '.join(words)+r' \\')
        lines.append(r'\addlinespace[3pt]')
    table('semantic_confirmation_fixed_case.tex',lines)
    return data


def plot_sources(source,save):
    import matplotlib.pyplot as plt
    ink='#262626'
    # Control-level source comparison: matched fits and earlier references stay visible.
    src=source['methods'];fig,axes=plt.subplots(1,2,figsize=(7,2.85));fig.subplots_adjust(left=.27,right=.985,bottom=.19,top=.85,wspace=.30)
    for ax,split,title in zip(axes,['calibration','held'],['(a) Source calibration','(b) Exposed held cities']):
        matrix=np.array([[100*next(c['score'] for c in r[split+'_controls'] if c['operation']==op) for op in CONTROLS] for r in src])
        ax.imshow(matrix,cmap='Greys',vmin=0,vmax=100,aspect='auto')
        for i in range(len(src)):
            for j in range(7):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>55 else ink)
        ax.set(yticks=range(len(src)),yticklabels=[r['short_label'] for r in src] if split=='calibration' else [],xticks=range(7),xticklabels=['C','T','L','CT','CL','TL','CTL'])
        ax.tick_params(length=0,labelsize=8);ax.set_title(title,loc='left',pad=10)
    fig.text(.27,.025,'Cell: per-request score (%). C/T/L: country/continent/language. CTL has Cause only.',fontsize=7)
    save(fig,'semantic_family_sources')


def plot(data,save):
    if not data:return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    green,purple,gray,ink='#286956','#785481','#888880','#262626'
    plot_sources(data['source_comparison'],save)
    # Each seed edge is visible; full readouts are separated from native writers.
    methods=['frozen_source','raw_supervised_family','raw_supervised_singleton','raw_shared_space','direct_shared_space','atom_pw_mcc','target_reencode','adaptive_native_reencode_count','adaptive_native256','adaptive_native512','random_base_native256']
    fig,axes=plt.subplots(1,2,figsize=(7,3.85));fig.subplots_adjust(left=.285,right=.99,bottom=.18,top=.91,wspace=.16)
    for ax,family,title in zip(axes,['single','binary'],['(a) Single requested attribute','(b) Seven binary requests']):
        for i,method in enumerate(methods):
            rr=[r for r in data['families'] if r['method']==method and r['family']==family]
            values=[100*r['score'] for r in rr];color=green if method.startswith('adaptive') else purple if method.startswith('raw_') else ink
            ax.plot([min(values),max(values)],[i,i],color=gray,lw=.65)
            for r,v in zip(rr,values):ax.plot(v,i+(r['source_seed']-3)*.065,marker=['o','s','^','D','v'][r['source_seed']-1],ms=3.2,color=color,mfc='white' if method in ['frozen_source','direct_shared_space'] else color)
            ax.plot(fmean(values),i,marker='|',ms=9,color=ink,mew=.9)
        ax.set(yticks=range(len(methods)),yticklabels=[LABELS[m] for m in methods] if family=='single' else [],ylim=(len(methods)-.5,-.5),xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Score (%)')
        ax.grid(axis='x',lw=.4,color='#e8e8e3');ax.tick_params(axis='y',length=0,labelsize=8);ax.set_title(title,loc='left',pad=10)
    handles=[Line2D([],[],linestyle='none',marker=m,ms=3,color=ink,label=f'{s} to {s%5+1}') for s,m in zip(range(1,6),['o','s','^','D','v'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.63,.005),ncol=5,frameon=False,fontsize=7.5,handletextpad=.35,columnspacing=.8)
    save(fig,'semantic_city_confirmation')
    # KL is a separate endpoint; values preserve the actual five dependent edges.
    kmethods=['raw_supervised_family','raw_supervised_singleton','raw_shared_space','direct_shared_space','atom_pw_mcc','target_reencode','adaptive_native_reencode_count','adaptive_native256','adaptive_native512','random_base_native256']
    fig,axes=plt.subplots(1,2,figsize=(7,3.5));fig.subplots_adjust(left=.285,right=.99,bottom=.19,top=.86,wspace=.22)
    vals=[r['source_kl'] for r in data['families'] if r['method'] in kmethods and r['family'] in ['single','fractional']]
    upper=max(vals)
    for ax,family,title in zip(axes,['single','fractional'],['(a) Single controls','(b) Untrained fractional controls']):
        matrix=np.array([[next(r['source_kl'] for r in data['families'] if r['method']==m and r['family']==family and r['source_seed']==s) for s in range(1,6)] for m in kmethods])
        ax.imshow(matrix,cmap='Greys',vmin=0,vmax=upper,aspect='auto')
        for i in range(len(kmethods)):
            for j in range(5):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7.5,color='white' if matrix[i,j]>.55*upper else ink)
        ax.set(yticks=range(len(kmethods)),yticklabels=[LABELS[m] for m in kmethods] if family=='single' else [],xticks=range(5),xticklabels=[f'{s}\u2192{s%5+1}' for s in range(1,6)])
        ax.tick_params(length=0,labelsize=7.5);ax.set_title(title,loc='left',pad=10)
    fig.text(.285,.025,'Cell: full-vocabulary KL to the same source control (nats); darker means larger error.',fontsize=7)
    save(fig,'semantic_city_fidelity')
    plot_fixed_case(data,save)


def plot_fixed_case(data,save):
    """Show the preselected city pair and actual answers, including failures."""
    import matplotlib.pyplot as plt
    import textwrap
    ink,green,purple,gray='#262626','#286956','#785481','#888880'
    fixed=data['fixed_example'];entity,donor=fixed['entity'],fixed['donor_entity']
    methods=['unedited','frozen_source','raw_supervised_singleton','direct_shared_space','target_reencode','adaptive_native_reencode_count','adaptive_native256','adaptive_native512']
    attrs=['Country','Continent','Language']
    def row(method,attr):
        return next(r for r in data['examples'] if r['seed']==1 and r['method']==method and r['task']==attr and r['operation']=='100')
    fig=plt.figure(figsize=(7,3.6))
    header=fig.add_axes([.025,.77,.96,.20]);header.axis('off')
    header.text(0,.92,f'Change country: {entity} to {donor}; retain continent and language',fontsize=9,va='top')
    header.text(0,.51,'Country prompt: '+fixed['text'],fontsize=8,va='top')
    header.text(0,.16,'The donor replaces only the city. Each answer column below uses its own attribute prompt.',fontsize=7.5,va='top',color=gray)
    ax=fig.add_axes([.025,.12,.36,.60]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(0,.98,'(a) Read the request, then write it',fontsize=9,va='top')
    for y,label,formula in [(.73,'Source operation',r'$q_s=B_s(c)D_s\Delta z_s$'),(.48,'Target readout',r'$\hat q=B_s(c)D_t\Delta z_t$'),(.15,'Target-native write',r'$D_{t,J}u,\quad u\geq-z_{t,J}$')]:
        ax.text(0,y+.10,label,fontsize=8,color=gray)
        ax.text(0,y,formula,fontsize=10,va='center')
    ax.annotate('',xy=(.47,.24),xytext=(.47,.39),arrowprops=dict(arrowstyle='->',lw=.7,color=ink))
    ax.text(.58,.315,'bounded\nrefit',fontsize=7.5,va='center')
    ax.text(0,-.035,r'$c=(1,0,0)$; the hook is at the city token.',fontsize=7.5)
    ax=fig.add_axes([.425,.12,.56,.60]);ax.set(xlim=(0,3),ylim=(9.5,-1.1));ax.axis('off')
    ax.text(0,-.90,'(b) Actual answers, source 1 to target 2',fontsize=9,va='bottom')
    for j,attr in enumerate(attrs):
        ax.text(1.53+j*.54,-.22,attr,fontsize=7.3,ha='center')
        r=row('frozen_source',attr);expected=r['donor_label'] if attr=='Country' else r['base_label']
        ax.text(1.53+j*.54,.48,expected,fontsize=7.5,ha='center',color=green)
    ax.text(0,.48,'Requested answer',fontsize=7.5,color=green)
    ax.plot([0,3],[.84,.84],color=gray,lw=.5)
    short={'unedited':'Unedited','frozen_source':'Source','raw_supervised_singleton':'Raw reference','direct_shared_space':'Decoded readout','target_reencode':'Re-encode','adaptive_native_reencode_count':'Native, matched count','adaptive_native256':'Native 256','adaptive_native512':'Native 512'}
    for i,method in enumerate(methods):
        y=i+1.5;ax.text(0,y,short[method],fontsize=7.5,va='center')
        for j,attr in enumerate(attrs):
            r=row(method,attr);answer=r['predicted_token_text'].strip()
            # No answer is replaced by a guessed full word or accepted alias.
            display=textwrap.fill(answer,12)
            ax.text(1.53+j*.54,y,display,fontsize=7.2,ha='center',va='center',color=ink if r['first_token_correct'] else purple)
    fig.text(.425,.025,'Purple: misses the requested answer. All requests and failures remain in the appendix.',fontsize=7)
    save(fig,'semantic_city_fixed_case')
