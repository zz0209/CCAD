"""Export functional effects of correspondence objectives to the current paper."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
PAPER=ROOT/'paper'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,colors
    import numpy as np
    panels=[json.loads((ART/f'{stem}.json').read_text()) for stem in ['r38_objectives','r38_objectives_range']]
    arrays=[]
    for stem in ['r38_objectives','r38_objectives_range']:
        with np.load(ART/f'{stem}.npz') as z:
            arrays.append({m:a for m,a in zip(z['methods'].tolist(),z['outcomes'])})
    labels=[('source','Role source'),('assignment','Role assignment'),('full_field','Full field'),
            ('random_rank','Random projection'),('requested_only','Requested contrast'),('conditional','Conditional contrast')]
    def mean(p,name,key):
        return 100*np.mean([c[key] for c in p['cells'] if c['method']==name])
    def contrast(p,other):
        return next(c for c in p['contrasts'] if c['comparator']==other and c['operation']=='both')['metrics']['H']
    out=PAPER/'data/arithmetic_objectives.json'
    out.write_text(json.dumps(dict(panels=panels,labels=labels),indent=2)+'\n')
    lines=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r' & \multicolumn{3}{c}{Original range} & \multicolumn{3}{c}{Larger operands} \\',
           r'Method & H & T & P & H & T & P \\',r'\midrule']
    for method,label in labels:
        vals=[mean(p,method,key) for p in panels for key in ['H','T','P']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\caption{\textbf{Selecting the comparison geometry.} Complete hybrid (H), requested-digit (T) and protected-digit (P) success, in percent. Both exposed panels use 64 question-pair clusters, two prompts, two requests and the same five SAEs. Each translated operation uses 64 target members across roles. All objective variants use the same 64 source-fit pairs; larger-operand evaluation performs no refitting.}',
              r'\label{tab:arithmetic_objectives}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_objectives.tex').write_text('\n'.join(lines)+'\n')
    values=[]
    for a in arrays:
        values.append(np.column_stack([(100*(a['conditional']-a[other])[...,0]).mean((0,2)).T
                                       for other in ['full_field','assignment']]))
    bound=max(5,int(np.ceil(max(abs(v).max() for v in values)/5)*5))
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    cmap=colors.LinearSegmentedColormap.from_list('functional_difference',['#644576','#faf9f6','#286956'])
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
                         'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(7,3.3),sharey=True)
        fig.subplots_adjust(left=.12,right=.865,bottom=.28,top=.86,wspace=.16)
        for ax,v,title in zip(axes,values,['Original range','Larger operands']):
            im=ax.imshow(v,cmap=cmap,vmin=-bound,vmax=bound,aspect='auto')
            for i in range(5):
                for j in range(4):
                    ax.text(j,i,f'{v[i,j]:+.1f}',ha='center',va='center',fontsize=8,
                            color='white' if abs(v[i,j])>.58*bound else '#222222')
            ax.set(xticks=range(4),xticklabels=['Units','Tens','Units','Tens'],
                   yticks=range(5),yticklabels=['5 → 1','1 → 2','2 → 3','3 → 4','4 → 5'])
            ax.set_title(title,fontsize=10,fontweight='normal')
            ax.text(.25,-.26,'Versus full field',transform=ax.transAxes,ha='center',fontsize=8)
            ax.text(.75,-.26,'Versus assignment',transform=ax.transAxes,ha='center',fontsize=8)
            ax.axvline(1.5,color='white',lw=2)
            ax.tick_params(length=0)
            for spine in ax.spines.values():spine.set_visible(False)
        axes[0].set_ylabel('Source → target SAE')
        cax=fig.add_axes([.89,.29,.018,.55]);bar=fig.colorbar(im,cax=cax)
        bar.set_label('Complete-success difference (points)',fontsize=8)
        bar.ax.tick_params(labelsize=8,length=2)
        for ext in ['pdf','svg','png']:
            fig.savefig(PAPER/f'figures/arithmetic_objectives.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    paragraphs=[]
    for p,title in zip(panels,['On the original range','On the larger-operand replay']):
        a,b=contrast(p,'full_field'),contrast(p,'assignment')
        paragraphs.append(f"{title}, conditional contrast gives {mean(p,'conditional','H'):.2f}\\% complete success. "
                          f"Its paired differences are {a['difference_points']:+.2f} points "
                          f"([{a['interval_points'][0]:.2f},{a['interval_points'][1]:.2f}]) versus full-field fitting and "
                          f"{b['difference_points']:+.2f} points ([{b['interval_points'][0]:.2f},{b['interval_points'][1]:.2f}]) "
                          "versus assignment.")
    paragraphs.append('All intervals resample whole question-pair clusters within each exposed panel. '+
                      r'Figure~\ref{fig:arithmetic_objectives} shows how the differences vary across requests and the fixed SAE cycle.')
    (PAPER/'sections/arithmetic_objective_findings.tex').write_text('\n\n'.join(paragraphs)+'\n')
    evidence=[x['run']+'/metrics.raw.jsonl' for p in panels for x in p['inputs']]
    evidence+=['artifacts/correspondence_reform_20260913/'+f for f in [
        'r38_objectives.json','r38_objectives.npz','r38_objectives_range.json','r38_objectives_range.npz',
        'R38_REPLAY_FREEZE.json','R38_CONTRAST_CHECK.json']]
    evidence+=['scripts/analyze_arithmetic_objectives.py','scripts/arithmetic_relation_transfer.py',
               'scripts/arithmetic_objectives_paper.py','paper/sections/arithmetic_objective_learning.tex',
               'paper/sections/arithmetic_objective_findings.tex','paper/tables/arithmetic_objectives.tex',
               'paper/figures/arithmetic_objectives.pdf','paper/data/arithmetic_objectives.json']
    (PAPER/'data/arithmetic_objectives_evidence.json').write_text(json.dumps(dict(
        id='task_contrast_correspondence',paper='Task contrast geometry and functional transfer',
        scope='Two exposed development panels; no new independent confirmation.',
        result=[p['contrasts'] for p in panels],evidence=evidence),indent=2)+'\n')
    mf=PAPER/'figures/FIGURE_MANIFEST.json';m=json.loads(mf.read_text())
    paths=['figures/arithmetic_objectives.'+x for x in ['pdf','svg','png']]
    m['outputs']=[x for x in m['outputs'] if x['path'] not in paths]+[dict(
        path=p,bytes=(PAPER/p).stat().st_size,sha256=hashlib.sha256((PAPER/p).read_bytes()).hexdigest()) for p in paths]
    m.setdefault('additional_sources',{})['arithmetic_objectives']=dict(
        path=out.relative_to(PAPER).as_posix(),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),generator='scripts/arithmetic_objectives_paper.py')
    mf.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(means=[[mean(p,n,'H') for n,l in labels] for p in panels],figure_scale=bound)))


if __name__=='__main__':main()
