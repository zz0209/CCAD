from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.aris/plot_runtime_v1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


METHODS=[('Local action','initial_inverse_budget'),('Recorded action','initial_trajectory_action'),
         ('State feedback','initial_trajectory_feedback'),('Trained program','task_adapted_tangent'),
         ('Source readout','initial_raw_readout')]


def number_part_effects(studies, output):
    records=[]
    for study in studies:
        run=Path(study['run'])
        design=json.loads((run/'source_design.json').read_text())
        with np.load(run/'responses.npz') as arrays:
            chosen=np.array([row['target_number']=='singular' for row in design['rows']])
            source_matched=np.where(chosen,arrays['singular'],arrays['plural'])
            source_other=np.where(chosen,arrays['plural'],arrays['singular'])
            clean=arrays['none']
            for method in ['source','local_action','recorded_action','state_feedback','raw_readout']:
                prefix='' if method=='source' else method+'__'
                matched=np.where(chosen,arrays[prefix+'singular'],arrays[prefix+'plural'])
                other=np.where(chosen,arrays[prefix+'plural'],arrays[prefix+'singular'])
                records.append(dict(run=str(run),method=method,documents=len(chosen),
                    matched_margin_change=float((matched-clean).mean()),
                    opposite_margin_change=float((other-clean).mean()),
                    part_contrast=float((other-matched).mean()),
                    source_order_agreement=float(((matched<other)==(source_matched<source_other)).mean())))
    output.write_text(json.dumps(dict(rows=records,scope='Descriptive reading of all frozen number cases. Matching number follows the original clean/patch task. Target sets, source and request family are fixed; primary inference remains in NUMBER_CONFIRMATION_ANALYSIS.json.'),indent=2)+'\n')
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',
           r'Execution & Matched & Opposite & Order agreement (\%) \\',r'\midrule']
    for label,method in [('Source program','source'),('Local action','local_action'),
                         ('Recorded action','recorded_action'),('State feedback','state_feedback'),('Source readout','raw_readout')]:
        rows=[row for row in records if row['method']==method]
        values=[np.mean([row[key] for row in rows]) for key in ['matched_margin_change','opposite_margin_change','source_order_agreement']]
        lines.append(f'{label} & {values[0]:.3f} & {values[1]:.3f} & {100*values[2]:.2f}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper/tables/trajectory_number_parts.tex').write_text('\n'.join(lines)+'\n')


def main():
    out=ROOT/'artifacts/reuse_generalization_20260921_round03'
    path=out/'TRAJECTORY_CONFIRMATION_ANALYSIS.json'
    number_path=out/'NUMBER_CONFIRMATION_ANALYSIS.json'
    result=json.loads(path.read_text());number=json.loads(number_path.read_text());records=[]
    for study in result['studies']:
        target=int(study['run'].split('_T')[-1].split('_')[0])
        for method,metrics in study['metrics'].items():
            for endpoint,families in metrics.items():
                for family,value in families.items():
                    records.append(dict(study='Human',target=target,method=method,endpoint=endpoint,family=family,nrmse=value))
    number_methods={'initial_inverse_budget':'local_action','initial_trajectory_action':'recorded_action',
                    'initial_trajectory_feedback':'state_feedback','initial_raw_readout':'raw_readout'}
    for study in number['studies']:
        target=int(study['run'].split('_T')[-1].split('_')[0])
        for method,queries in study['requests'].items():
            for query,value in queries.items():
                records.append(dict(study='Number',target=target,method=method,endpoint='answer_margin',family=query,nrmse=value))
    data=ROOT/'paper/data/trajectory_confirmation.csv'
    with data.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r' & \multicolumn{3}{c}{Human explanation} & Number \\',
           r'Execution & Full & Parts & New requests & All \\',r'\midrule']
    for label,method in METHODS:
        values=[np.mean([s['metrics'][method]['later'][f] for s in result['studies']]) for f in ['full','parts','participation']]
        number_value=(f"{np.mean([s['mean_nrmse'][number_methods[method]] for s in number['studies']]):.3f}"
                      if method in number_methods else r'\textemdash')
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+' & '+number_value+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    table=ROOT/'paper/tables/trajectory_confirmation.tex';table.write_text('\n'.join(lines)+'\n')
    number_part_effects(number['studies'],out/'NUMBER_PART_EFFECTS.json')
    font=Path('C:/Windows/Fonts/times.ttf')
    if not font.is_file():raise FileNotFoundError(font)
    font_manager.fontManager.addfont(str(font))
    with plt.rc_context({'font.family':'Times New Roman','font.size':8,'mathtext.fontset':'stix',
                         'pdf.fonttype':42,'axes.linewidth':.5,'xtick.major.width':.5,'ytick.major.width':.5}):
        fig,axes=plt.subplots(1,3,figsize=(6.4,2.25),sharey=True,layout='constrained')
        limits=[]
        for axis,family,title in zip(axes,['full','participation','number'],
                                     ['Human, complete','Human, new requests','Number, all requests']):
            for i,(label,method) in enumerate(METHODS):
                if family=='number':
                    if method not in number_methods:
                        axis.text(.02,4-i,'Not fitted',fontsize=7,color='.4',va='center')
                        continue
                    values=[s['mean_nrmse'][number_methods[method]] for s in number['studies']]
                else:
                    values=[s['metrics'][method]['later'][family] for s in result['studies']]
                color='#167568' if method=='initial_trajectory_feedback' else '#646464'
                for offset,marker,value in zip([-.13,0,.13],['o','s','^'],values):
                    axis.plot(value,4-i+offset,marker=marker,markersize=4,markeredgewidth=.65,
                              markerfacecolor='white',markeredgecolor=color,linestyle='none')
                axis.plot(np.mean(values),4-i,marker='|',markersize=12,color=color,linestyle='none')
                limits.extend(values)
            axis.set_title(title,fontsize=9,fontweight='normal',pad=8)
            axis.set_xlabel('Response nRMSE')
            axis.grid(axis='x',color='.90',linewidth=.5)
            axis.spines[['top','right']].set_visible(False)
            axis.set_yticks(range(5),[m[0] for m in METHODS][::-1]);axis.tick_params(axis='y',length=0)
        for axis in axes:axis.set_xlim(0,max(limits)*1.07);axis.set_ylim(-.45,4.45)
        for marker,label in zip(['o','s','^'],['Target 3','Target 4','Target 5']):
            axes[2].plot([],[],marker=marker,markerfacecolor='white',color='.4',markersize=4,linestyle='none',label=label)
        axes[2].legend(frameon=False,loc='lower right',fontsize=7)
        fig.savefig(ROOT/'paper/figures/trajectory_confirmation.pdf')
        fig.savefig(out/'trajectory_confirmation.png',dpi=170)
        plt.close(fig)
    index_path=ROOT/'paper/EVIDENCE_INDEX.json';index=json.loads(index_path.read_text())
    index['trajectory_confirmation']=dict(analysis=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        freeze=str((out/'TRAJECTORY_CONFIRMATION_FREEZE.json').relative_to(ROOT)),
        human_replay_freeze=str((out/'TRAJECTORY_REPLAY_FREEZE.json').relative_to(ROOT)),
        number_analysis=str(number_path.relative_to(ROOT)),number_sha256=hashlib.sha256(number_path.read_bytes()).hexdigest(),
        number_freeze=str((out/'NUMBER_REPORTING_REPLAY_FREEZE.json').relative_to(ROOT)),
        number_original_freeze=str((out/'NUMBER_CONFIRMATION_FREEZE.json').relative_to(ROOT)),
        data=str(data.relative_to(ROOT)),table=str(table.relative_to(ROOT)),figure='paper/figures/trajectory_confirmation.pdf',
        evidence='64 new biographies,12 new participation vectors,7 semantic endpoints;64 official-test number prefixes,3 named requests. Three fixed target dictionaries per program. Source remains available at inference.')
    index_path.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(rows=len(records),table=str(table),figure='paper/figures/trajectory_confirmation.pdf')))


if __name__=='__main__':main()
