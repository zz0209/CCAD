"""Show the retained binding requests; quantitative comparisons stay in Table3."""
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
fig=plt.figure(figsize=(3.375,1.40))
fig.text(.02,.95,'Henry: Poland (Warsaw)     Emma: Chile (Santiago)',va='top')
for x,label in [(.02,'Request'),(.63,'Henry'),(.89,'Emma')]:
    fig.text(x,.71,label,ha='left' if x<.1 else 'center')
fig.add_artist(plt.Line2D([.02,.99],[.64,.64],transform=fig.transFigure,color='.4',lw=.5))
for y,label,key in [(.43,'Swap people','entity'),(.22,'Also swap attributes','both')]:
    fig.text(.02,y,label)
    for x,answer in zip([.63,.89],example['decoded_outputs']['source_path'][key]):
        fig.text(x,y,answer,ha='center',color='#216b57')
fig.text(.02,.025,'Target SAE: 16 members chosen using saved source responses.',fontsize=7.2)
for ext in ['pdf','svg','png']:fig.savefig(ROOT/f'paper/figures/binding_request_example.{ext}',dpi=220)
plt.close(fig)
