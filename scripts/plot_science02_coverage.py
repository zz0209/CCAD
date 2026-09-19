"""One shared comparison for the two frozen request panels."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
sys.path.insert(0,str(ROOT/'.aris/plot_runtime_v1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    a=json.loads((OUT/'ROUND02_COVERAGE_ANALYSIS.json').read_text())
    b=json.loads((OUT/'ROUND02_INFINITIVE_ANALYSIS.json').read_text())
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                         'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42})
    definitions=[('endpoints','nearest','Endpoint calibration'),
                 ('coordinate_coverage','nearest','Coordinate coverage'),
                 ('source_response_coverage','nearest','Source-response coverage'),
                 ('random_mean_200','nearest','Random calibration'),
                 ('coordinate_coverage','global_choice','Coordinate, one adapter'),
                 ('source_response_coverage','global_choice','Response, one adapter')]
    labels=[x[2] for x in definitions]+['Best fixed (retrospective)','Reconstruction readout']
    rows=[]
    for result in [a,b]:
        data=[]
        for design,predictor,_ in definitions:
            data.append(next(r['mean_nrmse'] for r in result['summary'] if r['design']==design and r['predictor']==predictor and r['budget']==3 and r.get('information','full_response')=='full_response'))
        best=min(result['fixed'],key=lambda k:result['fixed'][k]['mean_nrmse'])
        data.append(result['fixed'][best]['mean_nrmse'])
        raw=result['external']['raw_reconstruction']
        data.append(raw['mean_nrmse'] if isinstance(raw,dict) else raw)
        rows.append(data)
    fig,axes=plt.subplots(1,2,figsize=(7.05,3.35),sharey=True)
    for ax,data,title in zip(axes,rows,['Human explanation / later readouts','Infinitive explanation / next token']):
        for i,v in enumerate(data):
            col='#276f64' if i in [1,2] else '#77516f' if i in [4,5] else '#303030'
            ax.plot(v,i,'D' if i>=4 and i<6 else 'o',ms=4,color=col)
            ax.annotate(f'{v:.3f}',(v,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=7)
        ax.set_title(title,fontsize=9,fontweight='normal')
        ax.set_xlim(0,max(max(r) for r in rows)*1.2)
        ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x',color='#dedede',lw=.4)
        ax.tick_params(axis='y',length=0)
        ax.axhline(3.5,color='#cfcfcf',lw=.5)
        ax.axhline(5.5,color='#cfcfcf',lw=.5)
        ax.set_xlabel('Held-request response nRMSE')
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.275,right=.99,bottom=.17,top=.86,wspace=.25)
    fig.savefig(OUT/'request_calibration_decisions.pdf')
    fig.savefig(OUT/'request_calibration_decisions.png',dpi=210)
    print('Saved request_calibration_decisions.pdf/png')

def plot_training():
    result=json.loads((OUT/'ROUND02_MIXED_ANALYSIS.json').read_text())
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                         'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.65),sharex=True,sharey=True)
    for ax,reference,title in zip(axes,['parts','continuous'],['Mixed minus part-only training','Mixed minus continuous-only training']):
        ax.axvline(0,color='#777777',lw=.7)
        for i,family in enumerate(['vertices','interior','boundary']):
            for response,offset,color,marker,label in [('head',-.12,'#276f64','o','Old task response'),('pooled',.12,'#77516f','s','Full hidden response')]:
                r=next(r for r in result['differences'] if r['family']==family and r['method']==response+'_mixed' and r['reference']==response+'_'+reference)
                lo,hi=r['interval'];x=r['delta']
                ax.plot([lo,hi],[i+offset]*2,color=color,lw=1)
                ax.plot(x,i+offset,marker,color=color,ms=4,label=label if i==0 else None)
        ax.set_title(title,fontsize=9,fontweight='normal')
        ax.set_xlim(-.10,.08);ax.set_xticks([-.10,-.05,0,.05])
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#e0e0e0',lw=.4)
        ax.set_xlabel('Change in response nRMSE')
    axes[0].set_yticks([0,1,2],['Deletion endpoints','Interior requests','Boundary requests'])
    axes[0].invert_yaxis()
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc='lower center',bbox_to_anchor=(.6,.015),ncol=2,fontsize=7)
    fig.subplots_adjust(left=.19,right=.99,bottom=.29,top=.83,wspace=.18)
    fig.savefig(OUT/'mixed_request_training.pdf')
    fig.savefig(OUT/'mixed_request_training.png',dpi=210)
    print('Saved mixed_request_training.pdf/png')

if __name__=='__main__':
    plot_training() if '--training' in sys.argv else main()
