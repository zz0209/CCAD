"""Paired full-answer and preservation effects for source-profile reuse."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    src=ROOT/'artifacts/correspondence_reform_20260913/r36_confirmation/confirmation.json'
    c=json.loads(src.read_text());out=ROOT/'paper/figures/arithmetic_positions'
    labels=[('role_scalar','Role scalar'),('static_member','Constant members'),('role_swapped','Wrong source'),('direct_320','Direct gradient (320 batches)')]
    metrics=[('exact_hybrid','Complete hybrid answer'),('preserve_digit_success','Preserved digit')]
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix','axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(7,2.4),sharey=True)
        fig.subplots_adjust(left=.245,right=.99,bottom=.24,top=.85,wspace=.15)
        for ax,(metric,title),color in zip(axes,metrics,['#286956','#785481']):
            ax.axvline(0,color='#777777',linewidth=.7,linestyle=':')
            for i,(method,label) in enumerate(labels):
                a=next(x for x in c['secondary'] if x['reference']=='role_member' and x['comparator']==method and x['operation']=='both')['metrics'][metric]
                mean=a['difference_points'];lo,hi=a['interval_points']
                ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o' if i<2 else 's',markersize=4,linewidth=.9,capsize=2,color=color)
            ax.set_title(title,fontsize=10,fontweight='normal',pad=7)
            ax.set(yticks=range(4),yticklabels=[x[1] for x in labels],xlabel='Difference (percentage points)')
            ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=7);ax.tick_params(axis='x',width=.6,length=3)
        axes[0].invert_yaxis()
        for ext in ['pdf','svg','png']:fig.savefig(out.with_suffix('.'+ext),dpi=220,facecolor='white')
        plt.close(fig)
    out.with_suffix('.json').write_text(json.dumps(dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),size_inches=[7,2.4],labels=labels,metrics=metrics,inference='Paired64question-cluster bootstrap conditional on five fixed SAEs. Circle full-answer comparisons are primary; preservation and square comparisons are diagnostics.'),indent=2)+'\n')
if __name__=='__main__':main()
