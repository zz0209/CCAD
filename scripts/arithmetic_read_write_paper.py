"""Export read/write separation evidence into the existing arithmetic appendix."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1]


def main():
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


if __name__=='__main__':main()
