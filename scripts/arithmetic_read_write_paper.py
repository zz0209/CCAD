"""Export read/write separation evidence into the existing arithmetic appendix."""
from pathlib import Path
import json,csv,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]


def export_confirmation(a,p):
    """Use the frozen analysis; add descriptive full-function intervals for plotting."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    x=json.loads((a/'r45_read_write_confirmation.json').read_text())
    supplement=json.loads((a/'r45_bank_ridge_control.json').read_text())
    x['supplementary_estimator_control']=supplement
    x['fidelity'] += [r for r in supplement['fidelity'] if r['method']=='bank_ridge_readout']
    x['cells'] += [r for r in supplement['cells'] if r['method']=='bank_ridge_readout']
    methods=['raw_readout','activation_readout','bank_ridge_readout','full_activation_readout',
             'response_code','full_native','full_response','member','assignment']
    labels=['Raw ridge','64-input sparse','64-input ridge*','Full-code ridge',
            '64-input response','Full-code Euclidean','Full-code response','Member relation','Assignment']
    scores={r['method']:r for r in x['fidelity']}
    data_runs=list(dict.fromkeys(x['runs']+supplement['runs']))
    rows=[json.loads(line) for run in data_runs for line in (ROOT/run/'metrics.raw.jsonl').read_text().splitlines()]
    lookup={(r['seed'],r['method'],r['operation'],r['row_id']):r for r in rows if r['kind']=='source_patch'}
    nq=json.loads((ROOT/x['run']/'config.resolved.json').read_text())['pairs_per_template']
    weights=np.random.default_rng(x['freeze']['bootstrap_seed']).multinomial(nq,np.full(nq,1/nq),10000)/nq
    full=[]
    for m in ['source']+methods:
        for op in ['unit','tens']:
            values=np.array([np.mean([lookup[s if m=='source' else s%5+1,m+'_full',op,form*nq+q]['exact_hybrid']
                for form in range(2) for s in range(1,6)]) for q in range(nq)])
            value=float(100*values.mean())
            cell=next(r for r in x['cells'] if r['method']==m and r['query']=='full' and r['operation']==op)
            assert abs(value-cell['H'])<1e-10
            full.append(dict(method=m,operation=op,H=value,interval=(100*np.quantile(weights@values,[.025,.975])).tolist()))
    full_lookup={(r['method'],r['operation']):r for r in full}
    colors=['#555555']*4+['#286956']*3+['#79506b','#888888']
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    with plt.rc_context({'font.family':'Times New Roman','font.size':8.5,'mathtext.fontset':'stix',
                         'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(6.8,2.85))
        axes=[fig.add_axes([.215+i*.258,.21,.232,.65]) for i in range(3)]
        for col,ax in enumerate(axes):
            for row,(m,color) in enumerate(zip(methods,colors)):
                r=scores[m] if col==0 else full_lookup[m,['unit','tens'][col-1]]
                v=r['balanced'] if col==0 else r['H'];lo,hi=r['interval']
                ax.plot([lo,hi],[row,row],color=color,lw=1)
                ax.plot(v,row,'s' if m=='full_response' else 'o',ms=4.2,color=color,
                        mfc='white' if m=='full_native' else color)
            ax.set(ylim=(8.6,-.6),yticks=range(9),yticklabels=labels if col==0 else ['']*9,
                   xlim=(40,103) if col==0 else (-3,103),xticks=[40,70,100] if col==0 else [0,50,100],
                   xlabel='Balanced agreement (%)' if col==0 else 'Complete success (%)',
                   title=['New part requests','Full units function','Full tens function'][col])
            if col:
                source=full_lookup['source',['unit','tens'][col-1]]['H']
                ax.axvline(source,color='#333333',lw=.8,ls=(0,(3,2)),zorder=0)
            ax.axhline(3.5,color='#dddddd',lw=.5)
            ax.axhline(6.5,color='#dddddd',lw=.5)
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(axis='y',length=0,pad=4)
            ax.grid(axis='x',color='#e2e2e2',lw=.5);ax.set_axisbelow(True)
        fig.text(.215,.055,'24 paired questions; five-SAE cycle; 95% question-cluster intervals',fontsize=8)
        fig.text(.215,.008,'* Supplementary estimator control on the same panel',fontsize=7.5)
        fig.text(.99,.94,'Dashed: source function',ha='right',fontsize=8)
        for ext in ['pdf','svg','png']:
            fig.savefig(p/f'figures/arithmetic_read_write_confirmation.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    table=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',
           r'Reading and execution & Part agreement & Full units & Full tens \\',r'\midrule']
    table.append('Source & --- & '+' & '.join(f"{full_lookup['source',op]['H']:.2f}" for op in ['unit','tens'])+r' \\')
    for m,label in zip(methods,labels):
        r=scores[m]
        table.append(label+f" & {r['balanced']:.2f} [{r['interval'][0]:.2f},{r['interval'][1]:.2f}] & "+
                     ' & '.join(f"{full_lookup[m,op]['H']:.2f}" for op in ['unit','tens'])+r' \\')
    table += [r'\bottomrule\end{tabular}',
        r'\caption{Read/write comparison on 24 question clusters, two new exact contexts and five dependent SAE directions. The frozen primary is full-code response minus 64-input response on original-strength part requests. The starred ridge row is a supplementary estimator control specified after confirmation; it uses the same input coordinates as the sparse reader. Full-function success requires both modification and preservation. All target-code realizations change at most 64 members in the same bank.}',
        r'\label{tab:read_write_confirmation}',r'\end{anchoredtable}']
    (p/'tables/arithmetic_read_write_confirmation.tex').write_text('\n'.join(table)+'\n')
    profile=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',
             r'Reading and execution & Request & Units H & T & P & Tens H & T & P \\',r'\midrule']
    cells={(r['method'],r['query'],r['operation']):r for r in x['cells']}
    for m,label in zip(['source']+methods,['Source']+labels):
        for q,qn in [('part0_bank0','A'),('part1_bank0','B'),('full','A+B')]:
            profile.append(label+' & '+qn+' & '+' & '.join(f"{cells[m,q,op][k]:.2f}" for op in ['unit','tens'] for k in ['H','T','P'])+r' \\')
    profile += [r'\bottomrule\end{tabular}',r'\caption{All frozen read/write requests. H is complete hybrid accuracy, T is requested-digit success and P is preserved-digit success. Every row contains 240 generations per digit function.}',
                r'\label{tab:read_write_profiles}',r'\end{anchoredtable}']
    (p/'tables/arithmetic_read_write_profiles.tex').write_text('\n'.join(profile)+'\n')
    x['full_function_intervals']=full
    (p/'data/arithmetic_read_write_confirmation.json').write_text(json.dumps(x,indent=2)+'\n')
    with (p/'data/arithmetic_read_write_confirmation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x['cells'][0]));w.writeheader();w.writerows(x['cells'])
    evidence=[run+'/'+name for run in x['runs'] for name in ['config.resolved.json','status.json','inputs.json','code_hashes.json','metrics.raw.jsonl','RESPONSE_EXECUTION.json']]
    evidence += ['artifacts/correspondence_reform_20260913/R45_READ_WRITE_FREEZE.json',
                 'artifacts/correspondence_reform_20260913/r45_read_write_confirmation.json',
                 'scripts/arithmetic_read_write_paper.py','scripts/fit_arithmetic_query_readouts.py','scripts/analyze_arithmetic_response.py']
    evidence += ['artifacts/correspondence_reform_20260913/r45_bank_ridge_control.json',
                 'artifacts/correspondence_reform_20260913/R45_BANK_RIDGE_CONTROL.json',
                 'scripts/fit_arithmetic_bank_ridge.py','scripts/analyze_arithmetic_bank_ridge.py',
                 'runs/REFORM_R45_bank_ridge_control_eval_v1_20260915/metrics.raw.jsonl',
                 'runs/REFORM_R45_bank_ridge_control_fit_v1_20260915/metrics.raw.jsonl']
    old=json.loads((p/'data/arithmetic_read_write_evidence.json').read_text())
    claim=dict(id='read_write_separation',paper='Reading a source request and realizing it in target members',
               result=x['primary'],full_function_contrasts=x['full_function_contrasts'],evidence=evidence,
               development=old)
    (p/'data/arithmetic_read_write_evidence.json').write_text(json.dumps(claim,indent=2)+'\n')
    manifest=p/'figures/FIGURE_MANIFEST.json';f=json.loads(manifest.read_text())
    f['outputs']=[r for r in f['outputs'] if 'arithmetic_read_write_confirmation' not in r['path']]
    for ext in ['pdf','svg','png']:
        out=p/f'figures/arithmetic_read_write_confirmation.{ext}'
        f['outputs'].append(dict(path=out.relative_to(p).as_posix(),bytes=out.stat().st_size,
                                sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    manifest.write_text(json.dumps(f,indent=2)+'\n')
    print(json.dumps(dict(primary=x['primary'],figure='arithmetic_read_write_confirmation',full=full)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--confirmation',action='store_true');args=ap.parse_args()
    a=ROOT/'artifacts/correspondence_reform_20260913';p=ROOT/'paper'
    x=json.loads((a/'r45_full_code_pilot.json').read_text())
    labels={
        'member':'Member relation','assignment':'Assignment','raw_readout':'Raw readout',
        'activation_readout':'64-input readout','reconstruction_readout':'Reconstruction readout',
        'full_activation_readout':'Full-code readout','native_code':'64-input Euclidean',
        'response_code':'64-input response','scalar_code':'64-input scalar response',
        'raw_native':'Raw Euclidean','raw_response':'Raw response',
        'reconstruction_response':'Reconstruction response',
        'full_native':'Full-code Euclidean','full_response':'Full-code response'}
    fidelity={r['method']:r['balanced'] for r in x['fidelity']}
    cells={(r['method'],r['operation']):r for r in x['cells']}
    lines=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Reader and execution & Part agreement & Full units & Full tens \\',r'\midrule']
    lines.append('Source & --- & '+' & '.join(f"{cells['source_full',op]['H']:.2f}" for op in ['unit','tens'])+r' \\')
    for method,label in labels.items():
        values=[fidelity[method]]+[cells[method+'_full',op]['H'] for op in ['unit','tens']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Development comparison of request reading and execution. Eight exposed question clusters, two prompt forms, source SAE1 to target SAE2. Part agreement balances changed and unchanged source answers. Full-function columns require correct modification and preservation. All code realizations use the same 64-member target bank.}',
        r'\label{tab:read_write_development}',r'\end{anchoredtable}']
    (p/'tables/arithmetic_read_write_development.tex').write_text('\n'.join(lines)+'\n')
    (p/'data/arithmetic_read_write_development.json').write_text(json.dumps(x,indent=2)+'\n')
    with (p/'data/arithmetic_read_write_development.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x['cells'][0]));w.writeheader();w.writerows(x['cells'])
    runs=[x['run']]+[r['run'] for r in x['reference_runs']]
    evidence=[r+'/'+name for r in runs for name in ['config.resolved.json','status.json','inputs.json','code_hashes.json','metrics.raw.jsonl']]
    evidence += ['runs/REFORM_R45_full_code_readouts_v1_20260915/config.resolved.json','runs/REFORM_R45_full_code_readouts_v1_20260915/metrics.raw.jsonl',
        'scripts/fit_arithmetic_query_readouts.py','scripts/arithmetic_readout_queries.py','scripts/arithmetic_read_write_paper.py',
        'artifacts/correspondence_reform_20260913/r45_full_code_pilot.json','paper/sections/arithmetic_role_queries_details.tex']
    claim=dict(id='read_write_separation_development',paper='Arithmetic appendix: reading information and writing members',
        result='Eight exposed question clusters, two forms, one source-target pair: full-code readout and response realization82.72% balanced part agreement; full units100% and tens25%. Development evidence; five-direction confirmation pending.',evidence=evidence)
    (p/'data/arithmetic_read_write_evidence.json').write_text(json.dumps(claim,indent=2)+'\n')
    print(json.dumps(dict(development_table='paper/tables/arithmetic_read_write_development.tex',methods=len(labels))))
    if args.confirmation:export_confirmation(a,p)


if __name__=='__main__':main()
