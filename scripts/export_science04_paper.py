"""Export the frozen confirmation to the existing manuscript's result sections."""
from pathlib import Path
import json,csv,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/science_upgrade_20260919'
sys.path.insert(0,str(ROOT/'.aris/plot_runtime_v1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

def main():
    data=[json.loads((OUT/f'ROUND04_{s}_ANALYSIS.json').read_text()) for s in ['HUMAN','INFINITIVE']]
    supplement=json.loads((OUT/'ROUND04_INFINITIVE_READOUT_SUPPLEMENT.json').read_text())
    choices=[('initial','native','Initial fixed relation','#6b6b6b','o'),
             ('head_mixed','mixed','Trained fixed relation','#77516f','s'),
             ('input_initial','native_tangent_relation_8','Initial input-dependent relation','#6b6b6b','o'),
             ('tangent_gain','tangent_gain','Calibrated source columns','#276f64','^'),
             ('tangent_mixed','tangent_mixed','Trained input-dependent relation','#77516f','s'),
             ('raw_reconstruction','raw_reconstruction','Source-direction readout','#111111','D')]
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.05,3.1),sharey=True)
    for k,(ax,d,title) in enumerate(zip(axes,data,['Human explanation / later readouts','Grammatical explanation / new contexts'])):
        largest=0;smallest=float('inf')
        for i,ch in enumerate(choices):
            r=d['summary'][ch[k]]['held_requests'];v=r['nrmse'];lo,hi=r['interval'];largest=max(largest,hi);smallest=min(smallest,lo)
            ax.plot([lo,hi],[i,i],color=ch[3],lw=.9)
            ax.plot(v,i,ch[4],ms=4,color=ch[3],mfc='white' if i==5 else ch[3])
            ax.annotate(f'{v:.3f}',(v,i),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
        assert smallest>0
        ax.set_xscale('log');ax.set_xlim(smallest*.77,largest*1.30)
        ax.xaxis.set_major_locator(FixedLocator([.02,.05,.1,.2,.5,1,2,5]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title,fontsize=9,fontweight='normal');ax.set_xlabel('Response error (nRMSE, log scale)')
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',lw=.4,color='#dedede')
        ax.axhline(1.5,lw=.5,color='#d0d0d0');ax.axhline(4.5,lw=.5,color='#d0d0d0')
    axes[0].set_yticks(range(len(choices)),[c[2] for c in choices]);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.30,right=.99,bottom=.20,top=.83,wspace=.23)
    fig.savefig(OUT/'program_confirmation.pdf');fig.savefig(OUT/'program_confirmation.png',dpi=230)
    shutil.copy2(OUT/'program_confirmation.pdf',ROOT/'paper/figures/program_confirmation.pdf')
    table=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'& \multicolumn{3}{c}{Human explanation} & \multicolumn{3}{c}{Grammatical explanation} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Execution & Endpoints & Participation & Members & Endpoints & Participation & Members \\',r'\midrule']
    rows=[]
    for ch in choices:
        values=[]
        for k,d in enumerate(data):
            for f in ['endpoints','held_requests','member_subsets']:
                r=d['summary'][ch[k]][f];values.append(r['nrmse'])
                rows.append(dict(study=d['setting'],method=ch[k],family=f,nrmse=r['nrmse'],lower=r['interval'][0],upper=r['interval'][1],by_seed=json.dumps(r['by_seed'])))
        table.append(ch[2]+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
    extra=supplement['summary']['raw_reconstruction_after_tangent_mixed']
    table.append('Readout from trained dictionary & --- & --- & --- & '+' & '.join(f"{extra[f]['nrmse']:.3f}" for f in ['endpoints','held_requests','member_subsets'])+r' \\')
    for m in ['raw_reconstruction_after_tangent_gain','raw_reconstruction_after_tangent_mixed']:
        for f in ['endpoints','held_requests','member_subsets']:
            r=supplement['summary'][m][f]
            rows.append(dict(study='infinitive',method=m,family=f,nrmse=r['nrmse'],lower=r['interval'][0],upper=r['interval'][1],by_seed=json.dumps(r['by_seed'])))
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/tables/program_confirmation.tex').write_text('\n'.join(table)+'\n')
    with (ROOT/'paper/data/program_confirmation.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (ROOT/'paper/data/program_confirmation.json').write_text(json.dumps(dict(human=data[0],infinitive=data[1],infinitive_readout_supplement=supplement),indent=2)+'\n')
    print('Exported program_confirmation figure, table and complete numerical data.')

if __name__=='__main__':main()
