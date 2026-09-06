"""Conditional composition, dose confound and operator-class comparison."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']=str(ROOT/'.aris/mplconfig')
import numpy as np
import matplotlib.pyplot as plt


def main():
    out=ROOT/'artifacts/calendar_composition_20260906'
    d=json.loads((out/'summary.json').read_text());run=ROOT/'runs'/d['run']
    capacity=ROOT/'runs/F4_calendar_operator_capacity_v1_20260906'
    cap=[json.loads(x) for x in (capacity/'metrics.raw.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r.get('record_kind')=='conditional_composition']
    stats=d['statistics'];blue='#0072B2';orange='#C66A00';gray='#677680';green='#087F65'
    spec=[('conditional64','Conditional 64',blue,'s'),('positive_only_energy_matched','Same 64, energy matched',orange,'D'),
          ('positive_only_same64','Same 64, positive only','#A37900','^'),('source_full','Full SAE',gray,'o'),
          ('conditional_atom','One atom','#956CB4','v'),('random64_refit','Random 64','#8C564B','P'),
          ('raw_conditional','Raw linear',green,'*')]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(13,9.8),facecolor='white')
        gs=fig.add_gridspec(2,2,left=.09,right=.96,bottom=.15,top=.84,hspace=.76,wspace=.35)
        fig.text(.045,.965,'Conditional function survives; fixed decoder masks lose much of it',fontsize=17,weight='bold')
        fig.text(.045,.927,'Same month words, different roles: preserve calendar changes and suppress password/title changes',fontsize=11,color='#45535B')
        ax=fig.add_subplot(gs[0,0])
        for method,label,color,marker in spec:
            a=[r for r in stats if r['scope']=='role' and r['operation']=='source' and r['method']==method]
            xx=[next(r['mean_conditional_kl'] for r in a if r['source']==s and r['key']=='calendar') for s in range(1,6)]
            yy=[next(r['mean_conditional_kl'] for r in a if r['source']==s and r['key']=='noncalendar') for s in range(1,6)]
            ax.scatter(xx,yy,c=color,marker=marker,s=30 if marker!='*' else 80,alpha=.85,label=label)
        ax.set(xlim=(0,.185),ylim=(0,.105),xlabel='Calendar KL error to raw intervention',
               ylabel='Noncalendar KL error to no edit',title='A   Raw linear is better on both role objectives')
        ax.text(.97,.97,'Lower left is better',ha='right',va='top',transform=ax.transAxes,fontsize=8,color=gray)
        ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(0,-.27),frameon=False,ncol=3)
        ax.grid(alpha=.15)
        ax=fig.add_subplot(gs[0,1])
        methods=['conditional64','positive_only_energy_matched','raw_conditional','source_full']
        names=['Conditional 64','Energy-matched 64','Raw linear','Full SAE']
        for j,(role,template,color) in enumerate([('calendar','test_calendar_our',blue),('noncalendar','test_password_your',orange)]):
            a=[r for r in rows if r['operation']=='source' and r['source_seed']==1 and r['template']==template and
               r['recipient_value']=='January' and r['donor_value']=='February']
            vals=[next(r['donor_margin_change'] for r in a if r['method']==m) for m in methods]
            ax.bar(np.arange(4)+(j-.5)*.32,vals,width=.3,color=color,label='Calendar effect' if j==0 else 'Password leakage')
        ax.axhline(0,color=gray,lw=.7)
        ax.set_xticks(range(4),names,rotation=25,ha='right',fontsize=8)
        ax.set(ylabel='Donor-answer versus original-answer margin change',title='B   First new-prefix pair: January to February')
        ax.text(0,1.20,'“Our current month is January. The following month is”\n“Your secret password is January. Your secret password is”',
                transform=ax.transAxes,fontsize=7.7,color='#45535B',linespacing=1.5)
        ax.legend(fontsize=7,frameon=False,loc='upper right');ax.grid(axis='y',alpha=.15)
        ax=fig.add_subplot(gs[1,0])
        methods=['native64','native256','native_full_ridge','code_readout','raw_linear']
        names=['Native 64','Native 256','Full native\n590–630','SAE-code\nreadout','Raw linear']
        for role,color,marker,style,label in [('calendar',blue,'o','-','Calendar change error'),
                                            ('noncalendar',orange,'s','--','Noncalendar leakage energy')]:
            arr=np.array([[next(r['relative_conditional_error'] for r in cap if r['phase']=='test_exposed' and
                               r['role']==role and r['method']==m and r['seed']==s) for m in methods] for s in range(1,6)])
            for s,y in enumerate(arr):
                ax.scatter(np.arange(5)+(s-2)*.024,y,c=color,marker=marker,s=13,alpha=.65)
            med=np.median(arr,axis=0);ax.plot(range(5),med,style,color=color,lw=1.5,label=label)
        ax.set_xticks(range(5),names,fontsize=8)
        ax.set(ylim=(0,1),ylabel='Squared error / raw variation energy',title='C   More members leave a large residual')
        ax.legend(fontsize=7,frameon=False,loc='upper right');ax.grid(axis='y',alpha=.15)
        ax=fig.add_subplot(gs[1,1])
        names=['One atom','Native 64','Raw linear','Full target SAE']
        methods=['best_native_atom','native64','raw_linear','target_full']
        for j,m in enumerate(methods):
            a=[r for r in stats if r['scope']=='all' and r['operation']=='cross_seed' and r['method']==m]
            if m=='raw_linear':
                a=[next(r for r in a if r['source']==s) for s in range(1,6)]
            y=np.array([r['aggregate_reference_kl_over_noop'] for r in a])
            xx=np.linspace(j-.15,j+.15,len(y));ax.scatter(xx,y,c=green if m=='raw_linear' else blue,s=17,alpha=.8)
            ax.plot([j-.2,j+.2],[np.median(y)]*2,color='black',lw=1.5)
            ax.text(j,max(y)*1.15,f'n={len(y)}',ha='center',fontsize=8)
        ax.axhline(1,color=gray,ls='--',lw=1)
        ax.set_xticks(range(4),names,rotation=15,ha='right',fontsize=8)
        ax.set_yscale('log');ax.set(ylim=(.015,9),ylabel='KL to source operation / no-edit KL (log scale)',
                                title='D   Cross-seed reproduction of the conditional source')
        ax.grid(axis='y',alpha=.15)
        fig.text(.045,.065,'A–B: new-prefix test cases only; source selection and maps froze before these interventions. Five source seeds share the same 12 month values.',fontsize=8,color='#45535B')
        fig.text(.045,.043,'C: adaptive cached-data diagnostic after test exposure, with no new LM evaluation. Points are seeds; lines connect descriptive medians, not confidence intervals.',fontsize=8,color='#45535B')
        fig.text(.045,.021,'D: 20 dependent directions; raw is identical for four targets of each source, so only five distinct raw values are drawn. No new FCC method advantage is established.',fontsize=8,color='#45535B')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'figure_conditional_composition.{ext}',dpi=240,facecolor='white')
        plt.close(fig)
    manifest=dict(destination='internal diagnostic,not submission-ready claim',dimensions_inches=[13,9.8],dpi=240,
       sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in
                [out/'summary.json',capacity/'metrics.raw.jsonl',run/'metrics.raw.jsonl']],
       code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       transformations=['A role mean KL per source over all96test donor pairs per role',
            'B first January-to-February pair in first new-prefix calendar/password templates,source1,unselected by target outcomes',
            'C within-role total squared error divided by total raw variation energy,5seeds',
            'D summed KL to actual source divided by summed no-edit KL; rawtarget duplicates shown once'],
       uncertainty='none; all plotted observations are dependent/descriptive',
       alt_text='Conditional64 largely overlaps an energy-matched control and raw linear performs better. Extending native masks to all fit-active members improves little; SAE code readout preserves substantially more conditional information. Cross-seed native64 beats one atom but raw linear is stronger.')
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out/'figure_conditional_composition.png')


if __name__=='__main__':main()
