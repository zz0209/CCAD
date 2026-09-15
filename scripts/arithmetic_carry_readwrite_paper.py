"""Export source read/write experiments and the frozen operator crossing."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'


def main():
    analyses={};cells={};runs=set()
    for tag in ['source','write_support','cone','crossed']:
        for arity in [2,3]:
            p=ART/f'r52_{tag}_{arity}.json'
            d=json.loads(p.read_text());analyses[f'{tag}_{arity}']=d;runs.add(d['run'])
            for cell in d['cells']:
                key=(arity,cell['method'],cell['condition'])
                if key in cells:
                    assert np.isclose(cells[key],cell['success']['mean'],atol=1e-12)
                cells[key]=cell['success']['mean']
    conditions=['same_answer_opposite_carry','same_carry_different_answer']
    methods=[('raw_carry_direction','Raw projection'),('code_scalar_64','Original diagonal'),
             ('diagonal_refit_64','Continued diagonal'),('readwrite_code_64','Code read/write'),
             ('readwrite_raw','Raw read/write'),('readwrite_rescaled_64','Code read/write, RMS'),
             ('readwrite_sparse_64','Reselected signed writer'),('unclipped_code_64','Signed writer, unclipped'),
             ('cone_fixed_64','Positive writers, fixed'),('cone_sparse_64','Positive writers, reselected')]
    allrows=[]
    for method,label in methods+[('code_read_raw_write','Code read, raw write'),('raw_read_code_write','Raw read, code write')]:
        row={'method':method,'label':label}
        for arity in [2,3]:
            for condition,suffix in zip(conditions,['change','preserve']):
                row[f'arity{arity}_{suffix}']=100*cells[arity,method,condition]
        allrows.append(row)
    with (ROOT/'paper/data/arithmetic_carry_readwrite.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    tex=['\\begin{table}[t]','\\centering\\small','\\begin{tabular}{lrrrr}','\\toprule',
         '& \\multicolumn{2}{c}{Two operands} & \\multicolumn{2}{c}{Three operands} \\\\',
         'Source operation & Change & Preserve & Change & Preserve \\\\','\\midrule']
    for row in allrows[:len(methods)]:
        values=[row[f'arity{arity}_{suffix}'] for arity in [2,3] for suffix in ['change','preserve']]
        tex.append(row['label']+' & '+' & '.join(f'{v:.2f}' for v in values)+' \\\\')
    tex += ['\\bottomrule','\\end{tabular}',
      '\\caption{Source read/write development on the common two- and three-operand requests. New learners use the same 405 mixed source-fit questions and 256 additional backward batches. The original raw projection uses the earlier two-operand fit set. Code reading uses the original 64 members; reselected writing uses a separate 64-member allowance. The unclipped diagnostic may produce negative codes. Raw read/write uses 1,536 hidden coordinates.}',
      '\\label{tab:carry_readwrite}','\\end{table}']
    (ROOT/'paper/tables/arithmetic_carry_readwrite.tex').write_text('\n'.join(tex)+'\n')
    costs=[]
    for run in sorted(runs):
        p=ROOT/run;d=json.loads((p/'metrics.summary.json').read_text())
        costs.append({'run':run,'status':json.loads((p/'status.json').read_text()),
                      **{k:d[k] for k in ['wall_seconds','process_cpu_seconds','peak_allocated_bytes','rows','sequence_forwards','token_forwards']}})
    combined={'written_at_utc':datetime.now(timezone.utc).isoformat(),'analyses':analyses,'table':allrows,'costs':costs,
      'total_driver_seconds':sum(c['wall_seconds'] for c in costs),
      'total_process_cpu_seconds':sum(c['process_cpu_seconds'] for c in costs),
      'scope':'All behavior is exposed source development; no new target relation or independent confirmation. Shared source group is conditioned on earlier learning.'}
    (ART/'R52_COMBINED_RESULTS.json').write_text(json.dumps(combined,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix',
                         'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.55))
    cmap=LinearSegmentedColormap.from_list('function',['#f7f7f3','#b5cfc3','#24634c'])
    layout=[['readwrite_code_64','code_read_raw_write'],['raw_read_code_write','readwrite_raw']]
    for ax,arity in zip(axes,[2,3]):
        values=np.array([[100*cells[arity,m,conditions[0]] for m in row] for row in layout])
        ax.imshow(values,cmap=cmap,vmin=0,vmax=100,aspect='auto')
        for i,row in enumerate(layout):
            for j,m in enumerate(row):
                change,preserve=[100*cells[arity,m,c] for c in conditions]
                ax.text(j,i,f'{change:.1f} / {preserve:.1f}',ha='center',va='center',color='white' if change>75 else '#172820')
        ax.set_xticks([0,1],['Native codes\n64 writers','Raw hidden\n1,536 coordinates'])
        ax.set_yticks([0,1],['Code reader\n64 members','Raw reader\n1,536 coordinates'])
        ax.set_title('Two operands' if arity==2 else 'Three operands',pad=9)
        ax.tick_params(length=0,pad=6)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.13,right=.99,bottom=.23,top=.84,wspace=.6)
    for suffix in ['pdf','svg','png']:
        fig.savefig(ROOT/f'paper/figures/arithmetic_carry_readwrite.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'runs':len(costs),'driver_seconds':combined['total_driver_seconds'],'rows':allrows[-2:]}))


if __name__=='__main__':
    main()
