"""Show the joint modification/preservation behavior of response objectives."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'paper/data/arithmetic_response.json').read_text())
    methods=[('response_field_prior','Field','#696969','s'),
             ('response_response','Two margins, finite','#9d4653','v'),
             ('response_finite_anchor1','Two margins + field','#8c6138','D'),
             ('response_direct_profile','Two margins, direct','#286956','o'),
             ('response_digit_direct','All digits, direct','#775789','P'),
             ('direct_320','Direct, 320 batches','#202020','*')]
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
                         'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(7,3.65),sharex=True,sharey=True)
        fig.subplots_adjust(left=.09,right=.985,bottom=.27,top=.91,wspace=.12)
        for ax,op,title in zip(axes,['unit','tens'],['Units replacement','Tens replacement']):
            for m,label,color,marker in methods:
                c=next(c for c in data['cells'] if c['method']==m and c['operation']==op)
                ax.scatter(c['target_digit_success']*100,c['preserve_digit_success']*100,
                           marker=marker,s=48 if marker=='*' else 31,color=color,
                           linewidths=.6,label=label,zorder=3)
            ax.set(xlim=(0,102),ylim=(0,102),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],title=title,
                   xlabel='Requested digit correct (%)')
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(length=3,width=.6)
            ax.grid(color='#dddddd',linewidth=.4,zorder=0)
        axes[0].set_ylabel('Preserved digit correct (%)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.005),ncol=3,
                   frameon=False,fontsize=8,columnspacing=1.2,handletextpad=.4)
        for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'paper/figures/arithmetic_response.{ext}',dpi=230,facecolor='white')
        plt.close(fig)


if __name__=='__main__':main()
