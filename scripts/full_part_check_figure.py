"""Draw the same-case full/part result and its retained illustrative example."""
from pathlib import Path
import csv
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'


def main():
    path=PAPER/'data/full_part_checks.json'
    data=json.loads(path.read_text())
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
        'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5})
    fig=plt.figure(figsize=(3.375,3.52))
    fig.text(.01,.99,'(a) A full edit and two parts, on one biography',va='top',fontsize=8)
    example=data['human']['examples'][0]
    columns=[(.01,'Delete'),(.54,'Source'),(.82,'Geometry')]
    for x,label in columns:fig.text(x,.902,label,ha='left' if x<.1 else 'center')
    fig.add_artist(plt.Line2D([.01,.99],[.876,.876],transform=fig.transFigure,color='.4',lw=.5))
    for y,q,label in [(.817,'full','All gender cues'),(.755,'pronouns','Pronouns'),(.693,'names','Names')]:
        fig.text(.01,y,label)
        # Published binary classifier: label1 is profession13 (nurse).
        left='Nurse' if example['source'][q]>0 else 'Professor'
        right='Nurse' if example['target']['geometry'][q]>0 else 'Professor'
        fig.text(.54,y,left,ha='center')
        fig.text(.82,y,right,ha='center',color='#9b4c50' if right!=left else '#222222')
    fig.text(.01,.622,'(b) Part errors among cases passing the full-edit check',fontsize=8)
    ax=fig.add_axes([.425,.13,.45,.425])
    rows=[('human','geometry','Geometry',6),('human','geometry_gain','Geometry + gains',5),
          ('human','native','Member relation',4),('human','raw','Source-dir. readout',3),
          ('arithmetic','assignment','Assignment',1),('arithmetic','member','Member relation',0)]
    records=[]
    for study,method,label,y in rows:
        result=data[study]['methods'][method]
        value=100*result['any_part_error'];lo,hi=[100*x for x in result['paired95']]
        color='#216b57' if method in ['native','member'] else '#78517b' if method=='raw' else '#777777'
        ax.barh(y,value,height=.48,color=color,alpha=.82)
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='none',ecolor=color,capsize=2,lw=.8)
        ax.text(hi+1.3,y,f'{value:.1f}',va='center',fontsize=7.2)
        records.append(dict(study=study,method=method,error_percent=value,lower95=lo,upper95=hi))
    ax.set_yticks([r[3] for r in rows],[r[2] for r in rows],fontsize=7.3)
    ax.tick_params(axis='y',length=0,pad=4)
    ax.tick_params(axis='x',labelsize=7,length=3)
    ax.set_xlim(0,60);ax.set_xticks([0,20,40,60]);ax.set_ylim(-.6,6.65)
    ax.set_xlabel('At least one part prediction wrong (%)',fontsize=7.5,labelpad=3)
    ax.spines[['top','right','left']].set_visible(False)
    ax.axhline(2,color='.8',lw=.5,xmin=0,xmax=1)
    fig.text(.01,.574,'Human annotation',fontsize=7.5)
    fig.text(.01,.29,'Arithmetic',fontsize=7.5)
    fig.text(.01,.012,'Same accepted cases within each study; lower is better.',fontsize=7.3)
    target=PAPER/'figures/full_part_checks'
    for ext in ['pdf','svg','png']:fig.savefig(target.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    with (PAPER/'data/full_part_checks_plot.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),outputs=[str(target.with_suffix('.'+s)) for s in ['pdf','svg','png']])))


if __name__=='__main__':main()
