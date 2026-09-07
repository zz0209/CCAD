"""Display the measured role/position screen and its causal-prefix boundary."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/seven_round_rebuild_20260906/r4_l15'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/mplconfig'))
STYLE={'font.family':'DejaVu Sans','font.size':7.2,'axes.titlesize':8,
       'axes.labelsize':7.2,'xtick.labelsize':6.7,'ytick.labelsize':7.0,
       'axes.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}
BLUE='#0072B2'; ORANGE='#D55E00'; GREY='#D8DDE2'


def main():
    records=[];sources=[]
    for role,name in [('Temporal','SEVEN_R4_role_temporal_v2_20260907'),
                      ('Quoted','SEVEN_R4_role_quoted_v3_20260907')]:
        p=ROOT/'runs'/name/'metrics.raw.jsonl'
        sources.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        groups={}
        for line in p.open():
            row=json.loads(line);key=(row.get('layer',-1),row.get('method','baseline'))
            value=groups.setdefault(key,[0,0,0.]);value[0]+=1;value[1]+=int(row['correct']);value[2]+=row.get('kl_to_full_donor',0)
        for (layer,method),(n,correct,kl) in groups.items():
            records.append(dict(role=role,layer=layer,method=method,n=n,correct=correct,accuracy=correct/n,kl=kl/n))
    lookup={(r['role'],r['layer'],r['method']):r for r in records}
    data=OUT/'role_material_figure_data.json';data.write_text(json.dumps(dict(rows=records,sources=sources),indent=2)+'\n')
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(7.1,3.95));gs=fig.add_gridspec(2,2,left=.07,right=.975,bottom=.13,top=.925,wspace=.30,hspace=.56,height_ratios=[1.1,1])
        ax=fig.add_subplot(gs[0,:]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.text(-.025,1.05,'a',weight='bold',fontsize=10)
        ax.text(.01,1.05,'A shared cue can play different roles',fontsize=8.2)
        ax.text(.01,.77,'Temporal',color=BLUE,weight='bold');ax.text(.14,.77,'Right now, the engineer near the chef',fontsize=8.2)
        ax.text(.01,.51,'Quoted',color=ORANGE,weight='bold');ax.text(.14,.51,'The title is “Right now”. Right now, the engineer near the chef',fontsize=8.2)
        ax.text(.14,.16,'Time change: Right now → Back then',fontsize=7)
        ax.text(.63,.16,'Number change: engineer → engineers',fontsize=7)
        ax.plot([.14,.95],[.37,.37],color=GREY,lw=.6)
        ax.text(.14,-.11,'At cue end: no access to later nouns or syntax',fontsize=7,color='#444')
        ax.text(.65,-.11,'At final token: complete prefix',fontsize=7,color='#444')
        for j,method in enumerate(['joint_separate_add','joint_final_add']):
            bx=fig.add_subplot(gs[1,j]);bx.set_ylim(0,1.14);bx.set_xlim(-.6,2.7);bx.set_xticks(range(3),['Layer 3','Layer 11','Layer 15']);bx.set_yticks([0,.5,1],['0','50','100']);bx.set_axisbelow(True);bx.grid(axis='y',color='#e7e7e7',lw=.45);bx.spines[['top','right']].set_visible(False)
            if j==0:bx.set_ylabel('Correct joint answer (%)')
            for k,(role,color) in enumerate([('Temporal',BLUE),('Quoted',ORANGE)]):
                for i,layer in enumerate([3,11,15]):
                    r=lookup[role,layer,method];x=i+(-.18 if k==0 else .18)
                    bx.bar(x,r['accuracy'],.29,color=color,zorder=2)
                    bx.text(x,max(.03,r['accuracy']+.035),f"{100*r['accuracy']:.0f}",ha='center',va='bottom',fontsize=6.5,color=color)
            bx.set_title('Separate cue / subject sites' if j==0 else 'Both changes at final token',loc='left',pad=12)
            bx.text(-.11,1.20,chr(ord('b')+j),transform=bx.transAxes,weight='bold',fontsize=10)
        files=[]
        for ext in ['png','pdf','svg']:
            p=OUT/('figure_role_positions.'+ext);fig.savefig(p,dpi=300);files.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
        plt.close(fig)
        (OUT/'figure_role_positions_manifest.json').write_text(json.dumps(dict(sources=sources,data_path=str(data.relative_to(ROOT)),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),files=files,description='Raw LM screen; eachbar256allrow developmentinputs, not seedreplicates. Sum of two separately obtained donor changes. Fulljointdonorreplacement excluded because identity at lastlayer. Sentence illustration uses firstnounblock/familiarcue byindex; quotedexpectedtimefixedpresent. No SAEorFCC result claimed.'),indent=2)+'\n')


if __name__=='__main__':main()
