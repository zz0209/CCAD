"""Compare the same execution choices for two published explanations."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/science_upgrade_20260919'
sys.path.insert(0,str(ROOT/'.aris/plot_runtime_v1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                     'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42})
human=json.loads((OUT/'ROUND03_SHIFT_ANALYSIS.json').read_text())
inf=json.loads((OUT/'ROUND03_ANALYSIS_FINAL.json').read_text())
choices=[('initial','native','Initial fixed relation','#696969','o'),
         ('head_mixed','mixed','Trained fixed relation','#77516f','s'),
         ('input_initial','native_tangent_relation_8','Initial input-dependent relation','#696969','o'),
         ('tangent_gain','tangent_gain','Calibrated source columns','#276f64','^'),
         ('tangent_mixed','tangent_mixed','Trained input-dependent relation','#77516f','s'),
         ('raw_reconstruction','raw_reconstruction','Source-direction readout','#111111','D')]
fig,axes=plt.subplots(1,2,figsize=(7.05,3.30),sharey=True)
for which,(ax,data,title) in enumerate(zip(axes,[human,inf],['Human explanation / later readouts','Infinitive explanation / new contexts'])):
    largest=0
    for i,item in enumerate(choices):
        m=item[which];row=data['summary'][m]['held_requests'];v=row['nrmse'];lo,hi=row['interval'];largest=max(largest,hi)
        ax.plot([lo,hi],[i,i],lw=.9,color=item[3])
        ax.plot(v,i,item[4],ms=4,color=item[3],mfc='white' if i==5 else item[3])
        ax.annotate(f'{v:.3f}',(v,i),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xlim(0,largest*1.08);ax.set_title(title,fontsize=9,fontweight='normal')
    ax.set_xlabel('Held-request response nRMSE')
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#dedede',lw=.4);ax.axhline(1.5,color='#d0d0d0',lw=.5)
    ax.axhline(4.5,color='#d0d0d0',lw=.5)
axes[0].set_yticks(range(len(choices)),[x[2] for x in choices]);axes[0].invert_yaxis()
fig.subplots_adjust(left=.30,right=.99,bottom=.19,top=.84,wspace=.25)
fig.savefig(OUT/'program_transfer_comparison.pdf');fig.savefig(OUT/'program_transfer_comparison.png',dpi=220)
print('Saved program_transfer_comparison.pdf/png')
