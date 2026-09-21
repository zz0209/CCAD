"""Draw the same-case full/part result and its retained illustrative example."""
from pathlib import Path
import argparse
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--human-only',action='store_true')
    args=parser.parse_args()
    path=PAPER/'data/full_part_checks.json'
    data=json.loads(path.read_text())
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
        'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5})
    fig=plt.figure(figsize=(6.75,2.62))
    fig.text(.01,.98,'(a) One biography, four feature deletions',va='top',fontsize=9)
    example=data['human']['examples'][0]
    display=json.loads((PAPER/'data/full_part_display_example.json').read_text())
    assert display['document_sha256']==example['document_sha256']
    fig.text(.01,.83,'“'+display['excerpt']+'”',fontsize=8.4,style='italic')
    columns=[(.01,'Features deleted'),(.24,'Source'),(.335,'Geometry'),(.435,'Member\nrelation')]
    for x,label in columns:fig.text(x,.72,label,ha='left' if x<.1 else 'center',va='center',fontsize=8)
    fig.add_artist(plt.Line2D([.01,.485],[.65,.65],transform=fig.transFigure,color='.4',lw=.5))
    for y,q,label in [(.56,'full','All gender cues'),(.455,'pronouns','Pronouns'),
                       (.35,'names','Names'),(.245,'associated_words','Associated words')]:
        fig.text(.01,y,label,fontsize=8)
        # Published binary classifier: label1 is profession13 (nurse).
        left='Nurse' if example['source'][q]>0 else 'Professor'
        right='Nurse' if example['target']['geometry'][q]>0 else 'Professor'
        member='Nurse' if example['target']['native'][q]>0 else 'Professor'
        fig.text(.24,y,left,ha='center',fontsize=8)
        fig.text(.335,y,right,ha='center',fontsize=8,color='#9b4c50' if right!=left else '#222222')
        fig.text(.435,y,member,ha='center',fontsize=8,color='#216b57')
    fig.text(.01,.08,'Edits act on SAE features; the input text stays fixed.',fontsize=8)
    fig.text(.52,.98,'(b) Part errors after the full-edit check passes',va='top',fontsize=9)
    ax=fig.add_axes([.705,.21,.235,.58])
    rows=[('human','geometry','Geometry',6),('human','geometry_gain','Geometry + gains',5),
          ('human','native','Member relation',4),('human','raw','Source-dir. readout',3),
          ('arithmetic','assignment','Assignment',1),('arithmetic','member','Member relation',0)]
    if args.human_only:
        rows=[(study,method,label,3-i) for i,(study,method,label,_) in enumerate(rows[:4])]
    records=[]
    for study,method,label,y in rows:
        result=data[study]['methods'][method]
        value=100*result['any_part_error'];lo,hi=[100*x for x in result['paired95']]
        color='#216b57' if method in ['native','member'] else '#78517b' if method=='raw' else '#777777'
        ax.barh(y,value,height=.48,color=color,alpha=.82)
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='none',ecolor=color,capsize=2,lw=.8)
        ax.text(hi+1.3,y,f'{value:.1f}',va='center',fontsize=7.8)
        records.append(dict(study=study,method=method,error_percent=value,lower95=lo,upper95=hi))
    ax.set_yticks([r[3] for r in rows],[r[2] for r in rows],fontsize=8)
    ax.tick_params(axis='y',length=0,pad=4)
    ax.tick_params(axis='x',labelsize=8,length=3)
    ax.set_xlim(0,35 if args.human_only else 60)
    ax.set_xticks([0,10,20,30] if args.human_only else [0,20,40,60])
    ax.set_ylim(-.6,3.65 if args.human_only else 6.65)
    ax.set_xlabel('At least one part wrong (%)',fontsize=8,labelpad=3)
    ax.spines[['top','right','left']].set_visible(False)
    if not args.human_only:
        ax.axhline(2,color='.8',lw=.5,xmin=0,xmax=1)
    fig.text(.52,.84,'Human annotation',fontsize=8,style='italic')
    if not args.human_only:
        fig.text(.52,.393,'Arithmetic',fontsize=8,style='italic')
    fig.text(.52,.035,'Identical accepted cases; paired 95% intervals.' if args.human_only else
             'Same cases within each study; paired 95% intervals.',fontsize=8)
    stem='human_part_check' if args.human_only else 'full_part_checks'
    target=PAPER/'figures'/stem
    for ext in ['pdf','svg','png']:fig.savefig(target.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    with (PAPER/'data'/f'{stem}_plot.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if args.human_only:
        def identity(p):
            return dict(path=p.relative_to(PAPER).as_posix(),bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        manifest_path=PAPER/'figures/FIGURE_MANIFEST.json'
        manifest=json.loads(manifest_path.read_text())
        outputs=[identity(target.with_suffix('.'+ext)) for ext in ['pdf','svg','png']]
        names={r['path'] for r in outputs}
        manifest['outputs']=[r for r in manifest['outputs'] if r['path'] not in names]+outputs
        manifest['human_part_check']=dict(
            sources=[identity(path),identity(PAPER/'data/full_part_display_example.json')],
            generator=dict(path='scripts/full_part_check_figure.py',
                           sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            arguments=['--human-only'],outputs=outputs+[identity(PAPER/'data'/f'{stem}_plot.csv')])
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),outputs=[str(target.with_suffix('.'+s)) for s in ['pdf','svg','png']])))


if __name__=='__main__':main()
