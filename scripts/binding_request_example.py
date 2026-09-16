"""Show actual member-choice consequences; aggregate comparisons stay in the text."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'paper/data/binding_member_selection.json').read_text())
example=data['example']
assert example['row']['names']==['Henry','Emma']
font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
                    'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(3.375,2.10))
fig.text(.02,.95,'Henry: Poland (Warsaw)     Emma: Chile (Santiago)',va='top')
for x,label in [(.02,'Member selection'),(.63,'Henry'),(.89,'Emma')]:
    fig.text(x,.80,label,ha='left' if x<.1 else 'center')
fig.add_artist(plt.Line2D([.02,.99],[.755,.755],transform=fig.transFigure,color='.4',lw=.5))
for base,label,key in [(.66,'Swap people','entity'),(.32,'Also swap attributes','both')]:
    fig.text(.02,base,label,fontsize=8,fontstyle='italic')
    for offset,method,name in [(.095,'geometry','Magnitude'),(.185,'source_path','Saved source responses')]:
        y=base-offset
        fig.text(.04,y,name,fontsize=7.7)
        correct=example['decoded_outputs']['source_path'][key]
        for x,answer,expected in zip([.63,.89],example['decoded_outputs'][method][key],correct):
            fig.text(x,y,answer,ha='center',color='#9b4c50' if answer!=expected else '#222222')
fig.text(.02,.015,'Both selections retain 16 of the same 64 target contributions.',fontsize=7.2)
for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'paper/figures/binding_request_example.{ext}',dpi=220)
plt.close(fig)
