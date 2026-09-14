"""Plot complete answer outcomes at matched target backward budgets."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    styles = dict(zip(['hybrid','base','wrong_digit','donor','other'],
                     [('#286956',''),('#ededed',''),('#c2b3c8','..'),('#785481','//'),('#777777','xx')]))
    methods = [('clean','Source-field'),('clean_fixed','Fixed translated members'),
               ('clean_swapped','Swapped source function'),('target_gradient','Target gradient'),
               ('target_integrated_gradient','Target integrated gradient'),
               ('fisher_contrast','Target Fisher'),('assignment','Member assignment')]
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
                         'axes.titlesize':10,'axes.linewidth':.6,'pdf.fonttype':42,
                         'svg.fonttype':'none','hatch.linewidth':.3}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0,3.35), sharey=True)
        fig.subplots_adjust(left=.265, right=.985, top=.86, bottom=.25, wspace=.12)
        for ax, op, title in zip(axes,['unit','tens'],['Units replacement','Tens replacement']):
            for y, (method, label) in enumerate(methods):
                c = next(c for c in data['cells'] if c['method']==f'adapt_{method}_u64' and c['operation']==op)
                values = dict(hybrid=c['outcomes'][op], base=c['outcomes']['base'],
                              wrong_digit=c['outcomes']['tens' if op=='unit' else 'unit'],
                              donor=c['outcomes']['donor'],other=c['outcomes']['other'])
                assert abs(sum(values.values())-1)<1e-10
                left=0
                for key, val in values.items():
                    color, hatch=styles[key]
                    ax.barh(y,val*100,left=left,height=.67,color=color,hatch=hatch,
                            edgecolor='#555555',linewidth=.3)
                    left+=val*100
                ax.text(c['hybrid']*50,y,f"{c['hybrid']*100:.1f}",ha='center',va='center',
                        color='white' if c['hybrid']>.1 else 'black',fontsize=8)
            ax.set(xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Generated answers (%)',title=title,
                   yticks=np.arange(len(methods)),yticklabels=[label for _,label in methods])
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(axis='y',length=0,pad=6)
            ax.tick_params(axis='x',width=.6,length=3,labelsize=8)
        axes[0].invert_yaxis()
        labels=['Correct hybrid','Recipient unchanged','Other digit replaced','Both digits replaced','Other output']
        handles=[Patch(facecolor=color,hatch=hatch,edgecolor='#555555',linewidth=.3,label=lab)
                 for (color,hatch),lab in zip(styles.values(),labels)]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.015),ncol=3,
                   frameon=False,fontsize=8,columnspacing=1.2,handlelength=1.5)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        for ext in ['pdf','svg','png']:
            fig.savefig(args.output.with_suffix('.'+ext),dpi=220,facecolor='white')
        plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        data=str(args.data),sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        width_inches=7,height_inches=3.35,fonts='Times New Roman; STIX math',
        scope='Development means at64total target backward batches, equal over five fixed seeds and two prompt forms. Direct gradient selection consumes16of64batches. No uncertainty or independent seed extrapolation is implied.',
        categories=labels,denominator='All outcomes retained; parsed full answer, no base-correct filtering.'),indent=2)+'\n')


if __name__=='__main__':
    main()
