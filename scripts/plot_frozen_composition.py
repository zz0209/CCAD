"""Manuscript-width frozen transfer comparisons and concrete syntax cases."""
import json,hashlib,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.aris/mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullFormatter
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/seven_round_rebuild_20260906/r4_frozen'
METHODS=[('fcc_group','FCC, compact'),('dense_select','Dense selection'),('full_code_ridge','Full-code ridge'),('rrr_rank4','Reduced rank 4'),('rrr_rank1','Reduced rank 1'),('dsca_rank1','dSCA, rank 1'),('raw_native_units','Raw-state ridge'),('one_to_one','One-to-one'),('conditional_ot','Conditional OT'),('single_atom','Single atom'),('random_refit','Random members'),('same_members_native','Same members, native'),('same_members_behavior_gain','Native, calibrated gain'),('direct_target_native','Direct target group'),('das_style_raw_rank1','DAS-style, rank 1')]
PARTS=[('familiar_syntax','Familiar syntax'),('object_relative','Object relative'),('subject_late','Subject after distractor')]
STYLE={'font.family':'DejaVu Sans','font.size':7.3,'axes.titlesize':8,'axes.labelsize':7.4,'xtick.labelsize':6.8,'ytick.labelsize':7.1,'axes.linewidth':.55,'lines.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'}
COLORS=['#A9BCC8','#236B8E','#D9B6A6','#A74824']


def save(fig,name,source,description):
    files=[]
    for ext in ['png','pdf','svg']:
        p=OUT/(name+'.'+ext);fig.savefig(p,dpi=300,facecolor='white');files.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    (OUT/(name+'_manifest.json')).write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),description=description,figure_inches=fig.get_size_inches().tolist(),files=files),indent=2)+'\n');plt.close(fig)


def main():
    source=OUT/'R4_FROZEN_SUMMARY.json';d=json.loads(source.read_text());lookup={(r['partition'],r['factor'],r['method']):r for r in d['rows']}
    displayed=np.array([v for part,_ in PARTS for method,_ in METHODS for v in lookup[part,'joint',method]['source_seed_means']]);assert np.all(displayed>0)
    limits=(10**np.floor(np.log10(displayed.min())),10**np.ceil(np.log10(displayed.max())))
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(7.1,5.9));gs=fig.add_gridspec(2,3,left=.265,right=.99,top=.91,bottom=.10,hspace=.44,wspace=.16,height_ratios=[1,4.6]);axes=[]
        for j,(part,title) in enumerate(PARTS):
            ax=fig.add_subplot(gs[0,j]);ax.set_xlim(-.5,2.5);ax.set_ylim(-.4,1.12);ax.set_xticks(range(3),['Number','Time','Joint']);ax.tick_params(axis='both',length=0,pad=3);ax.set_yticks([])
            for k,factor in enumerate(['number','time','joint']):
                r=lookup[part,factor,'source_teacher'];ax.add_patch(Rectangle((k-.42,0),.84,.8,facecolor=plt.cm.Blues(.10+.80*r['accuracy']),edgecolor='white',linewidth=.8));ax.text(k,.4,f"{100*r['accuracy']:.1f}%",ha='center',va='center',color='white' if r['accuracy']>.65 else '#222222',fontsize=8)
            ax.set_title(title,pad=9)
            for spine in ax.spines.values():spine.set_visible(False)
            ax=fig.add_subplot(gs[1,j]);axes.append(ax);ax.set_xscale('log');ax.set_xlim(*limits);ax.set_ylim(len(METHODS)-.3,-.7);ax.set_yticks(range(len(METHODS)),[label for _,label in METHODS] if j==0 else ['']*len(METHODS));ax.tick_params(axis='y',length=0,pad=6);ax.tick_params(axis='x',length=3);ax.xaxis.set_major_locator(LogLocator(base=10,numticks=7));ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(axis='x',color='#e2e2e2',linewidth=.45);ax.set_axisbelow(True)
            for i,(name,_) in enumerate(METHODS):
                r=lookup[part,'joint',name];v=np.array(r['source_seed_means']);assert np.all(v>0)
                color='#0072B2' if name=='fcc_group' else '#333333';offset=np.linspace(-.16,.16,len(v));ax.scatter(v,i+offset,s=10,facecolor='white',edgecolor=color,linewidth=.6,zorder=3);ax.plot([r['kl'],r['kl']],[i-.25,i+.25],color=color,lw=1.7,zorder=4)
            ax.spines[['top','right','left']].set_visible(False)
        fig.text(.04,.955,'a',weight='bold',fontsize=10);fig.text(.073,.955,'Source intervention reaches the intended answer',fontsize=8.2)
        fig.text(.04,.73,'b',weight='bold',fontsize=10);fig.text(.073,.73,'Frozen transfer of the joint operation',fontsize=8.2)
        fig.text(.63,.025,'KL from source to transferred output (nats)',ha='center',fontsize=8)
        save(fig,'figure_frozen_transfer',source,'All512newinputs, no refitting. Top:source expected-label accuracy, five-seed mean. Bottom:joint full-vocabulary KL; five dots are source-seed means across four targetseeds, ticks are pooled means; DAS hasfive sourcefits. Shared seeds aredependent. Fifteen declared primarycontrols shown; all22 remain in summary. Commonlogaxis, no row exclusions.')
        examples={(r['template'],r['factor'],r['method']):r for r in d['examples']};cfg=json.loads((ROOT/'configs/seven_r4_frozen_correspondence_v1.json').read_text());panel=json.loads((ROOT/cfg['material_run']/'panel.json').read_text());baseline={}
        for line in (ROOT/cfg['material_run']/'metrics.raw.jsonl').open():
            r=json.loads(line)
            if r['kind']=='baseline':baseline[r['row_id']]=r
        fig=plt.figure(figsize=(7.1,5.25));gs=fig.add_gridspec(4,2,left=.04,right=.99,top=.91,bottom=.09,wspace=.08,hspace=.50,width_ratios=[1.4,1]);names=[('source_teacher','Source'),('fcc_group','FCC'),('same_members_native','Native members'),('direct_target_native','Direct target')]
        for i,template in enumerate(['pp','subject_relative','object_relative','subject_late']):
            ax=fig.add_subplot(gs[i,0]);ax.set_axis_off();r=examples[template,'joint','fcc_group'];row=panel['rows'][r['row_id']];text=row['text'].replace('<|endoftext|>','');label={'pp':'Prepositional phrase','subject_relative':'Subject relative','object_relative':'Object relative','subject_late':'Subject after distractor'}[template]
            ax.text(0,.92,f'{chr(97+i)}  {label}',weight='bold',fontsize=8.1,transform=ax.transAxes)
            # Break only at the clause boundary; the complete original text is retained.
            chunks=text.split(', ',1);shown=chunks[0]+',\n'+chunks[1] if len(chunks)==2 else text
            ax.text(.035,.62,shown,fontsize=8.0,va='top',linespacing=1.5,transform=ax.transAxes)
            ax.text(.035,.02,'Joint change: plumber → plumbers; now → then',fontsize=6.7,color='#444444',transform=ax.transAxes)
            bx=fig.add_subplot(gs[i,1]);bx.set_xlim(0,4);bx.set_ylim(3.6,-.6);bx.set_yticks(range(4),[x[1] for x in names]);bx.tick_params(axis='both',length=0,pad=4);bx.set_xticks(np.arange(4)+.5,['is','are','was','were'] if i==3 else ['']*4)
            for j,(name,_) in enumerate(names):
                lp=np.array(examples[template,'joint',name]['label_logprobs']);prob=np.exp(lp-lp.max());prob/=prob.sum()
                for k,p in enumerate(prob):
                    bx.add_patch(Rectangle((k,j-.36),1,.72,facecolor=COLORS[k],alpha=.08+.92*p,lw=0));bx.text(k+.5,j,f'{100*p:.0f}',ha='center',va='center',fontsize=7,color='white' if p>.60 and k in [1,3] else '#222222')
                winner=int(np.argmax(prob));bx.add_patch(Rectangle((winner,j-.36),1,.72,fill=False,lw=.6,edgecolor='#222222'))
            bx.spines[:].set_visible(False)
        fig.text(.04,.975,'A fixed new lexical pair across four sentence structures',fontsize=9,weight='normal')
        fig.text(.80,.953,'Joint output, four-label probability (%)',ha='center',fontsize=7)
        save(fig,'figure_frozen_contexts',source,'Fixed firstlexicalblock/cue/direction/source1target2, chosenbyindex rather than outcome. Complete stems andfour-label conditional joint outputs; outlinedwinninglabel. Full-vocabulary KL remains in separate quantitativefigure. Allsources/contexts/controloutcomes retained. This is a transfer illustration, not a causal circuit graph.')


if __name__=='__main__':main()
