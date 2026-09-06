"""Internal material diagnostic with all prompt conditions and a fixed example."""
import os
import json
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(ROOT/'.aris/mplconfig')
import numpy as np
import matplotlib.pyplot as plt


def main():
    out = ROOT/'artifacts/calendar_material_20260906'
    p = out/'summary.json'
    d = json.loads(p.read_text())
    run = ROOT/'runs'/d['run']
    base = json.loads((run/'baseline_results.json').read_text())['rows']
    labels = ['Weekday: next / Today','Weekday: next / after',
              'Weekday: previous / Today','Weekday: previous / before',
              'Weekday: identity / meeting','Weekday: identity / calendar',
              'Month: next / This month','Month: next / after',
              'Month: previous / This month','Month: previous / before',
              'Month: identity / event','Month: identity / calendar']
    blue, orange, gray = '#0072B2', '#C66A00', '#717A80'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(13.2,9.4),facecolor='white')
        gs = fig.add_gridspec(2,2,left=.21,right=.97,bottom=.13,top=.85,
                              wspace=.35,hspace=.70,height_ratios=[1.35,1])
        fig.text(.045,.96,'A functional variable is visible, but the scope is narrow',fontsize=18,weight='bold')
        fig.text(.045,.924,'114 authored calendar prompts; three raw layers; five existing SAEs at layer 6 of a 12-layer model',fontsize=11,color='#45535B')
        a = fig.add_subplot(gs[0,0])
        yy = np.arange(12)
        acc = np.array([x['family_correct']/x['n'] for x in d['baseline']])
        full = np.array([x['full_correct']/x['n'] for x in d['baseline']])
        a.barh(yy,acc,color=blue,alpha=.85,height=.62,label='Within calendar candidates')
        a.plot(full,yy,'D',ms=4,color=orange,label='Full-vocabulary top-1')
        for y,x,r in zip(yy,acc,d['baseline']):
            a.text(x+.025,y,f"{r['family_correct']}/{r['n']}",va='center',fontsize=8)
        a.set_yticks(yy,labels,fontsize=8)
        a.invert_yaxis()
        a.set(xlim=(-.025,1),xlabel='Immediate-next-token accuracy',title='A   Baseline depends strongly on wording')
        a.legend(fontsize=7,loc='upper left',bbox_to_anchor=(0,-.25),frameon=False)
        a.grid(axis='x',alpha=.15)
        a = fig.add_subplot(gs[0,1])
        recipient = next(r for r in base if r['template']=='month_next_plain' and r['value']=='January')
        donor = next(r for r in base if r['template']=='month_next_plain' and r['value']=='February')
        ex = [x for x in d['example'] if x['mode']=='slot']
        pair_probs = [[r['candidate_probabilities'][i] for i in [1,2]] for r in [recipient,donor]]
        pair_probs += [[r['candidate_probabilities'][i] for i in [1,2]] for r in ex]
        names = ['Base','Donor','Raw L3','Raw L6','Raw L9']+[f'SAE {s}' for s in range(1,6)]
        pos = np.arange(len(names))
        pp = np.array(pair_probs)
        a.bar(pos-.17,pp[:,0],width=.32,color=gray,label='February (original answer)')
        a.bar(pos+.17,pp[:,1],width=.32,color=blue,label='March (donor answer)')
        a.set_xticks(pos,names,rotation=60,ha='right',fontsize=8)
        a.set(ylim=(0,max(pp.ravel())*1.35),ylabel='Full-vocabulary token probability',
              title='B   Change January to February at its token')
        a.text(0,1.13,'“This month is January. Next month is”',transform=a.transAxes,fontsize=9)
        a.legend(fontsize=7,frameon=False,loc='upper left')
        a.grid(axis='y',alpha=.15)
        a = fig.add_subplot(gs[1,0])
        pooled = [r['centered_reconstruction_fve'] for r in d['material']['rows']]
        targeted = [r['variable_centered_fve'] for r in d['material_contrasts'] if r['template']=='month_next_plain']
        for s,(x,y) in enumerate(zip(pooled,targeted),1):
            offset = (s-3)*.025
            a.plot([0+offset,1+offset],[x,y],'-',lw=.9,color=gray,alpha=.8)
            a.plot(0+offset,x,'o',color=gray,ms=4)
            a.plot(1+offset,y,'s',color=blue,ms=4)
        a.set_xticks([0,1],['Pooled task states\n(all nonpadding tokens)',
                            'Month-value variation\n(one template, value slot)'])
        a.set(xlim=(-.35,1.35),ylim=(0,1.05),ylabel='Centered reconstruction FVE',
              title='C   High pooled FVE hides variable-level error')
        a.text(.02,.12,'Each line: one SAE seed.\nDifferent centering distributions are explicit.',
               transform=a.transAxes,fontsize=8,color='#45535B')
        a.grid(axis='y',alpha=.15)
        a = fig.add_subplot(gs[1,1])
        m = [r for r in d['operations'] if r['template']=='month_next_plain' and
             r['method']=='sae_delta' and r['mode']=='slot']
        kl = [r['aggregate_relative_reference_kl'] for r in m]
        a.axhline(1,color=gray,ls='--',lw=1,label='No edit = 1')
        a.plot(range(1,6),kl,'s-',color=blue,label='SAE decoded donor difference',lw=1.4)
        for s,y in enumerate(kl,1):
            a.text(s,y+.05,f'{y:.3f}',ha='center',fontsize=8)
        a.set(xticks=range(1,6),xlabel='SAE seed',ylim=(0,1.12),
              ylabel='Aggregate KL error / no-edit error',
              title='D   Most raw output effect survives reconstruction')
        a.text(5,1.035,'No edit = 1',ha='right',fontsize=8,color=gray)
        a.grid(axis='y',alpha=.15)
        fig.text(.045,.06,'B: first predeclared month-next pair, not selected for success. Raw patches change one token; SAE edits preserve the recipient residual.',fontsize=8,color='#45535B')
        fig.text(.045,.036,'D: all 24 month-next pairs, including wrong baseline answers; KL uses the raw layer-6 patch as reference. Shared inputs and seeds are dependent.',fontsize=8,color='#45535B')
        fig.text(.045,.013,'This is a material diagnostic. No FCC map, semantic uniqueness, unseen-wording generalization, or new method advantage is established.',fontsize=8,color='#45535B')
        for ext in ['png','pdf','svg']:
            fig.savefig(out/f'figure_calendar_material.{ext}',dpi=240,facecolor='white')
        plt.close(fig)
    manifest = dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dimensions_inches=[13.2,9.4],dpi=240,destination='internal material report; no venue compliance claim',
        transformations=['A: correct / all values within each of 12 templates',
          'B: full-vocabulary probabilities for February and March, fixed first month-next pair',
          'C: pooled state centering versus value-slot centering within one template; all five seeds',
          'D: sum KL(raw patch || SAE patch) / sum KL(raw patch || no edit), all 24 pairs per seed'],
        uncertainty='descriptive; no independence assumption or CI',exclusions='none from experiment; B and D scope explicitly shown',
        alt_text='Month-next in one wording is a positive signal. Alternative wording and most weekday prompts are weak. All five SAEs preserve much of the raw intervention output effect although variable-centered reconstruction is around 0.51 versus pooled reconstruction around 0.999.')
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out/'figure_calendar_material.png')


if __name__=='__main__':
    main()
