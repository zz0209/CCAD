"""Export role-dependent source-profile calibration and its frozen confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'paper';ART=ROOT/'artifacts/correspondence_reform_20260913'
def main():
    c=json.loads((ART/'r36_confirmation/confirmation.json').read_text());dev=json.loads((ART/'r36_development.json').read_text())
    labels=[('role_member','Member weights by answer role'),('role_scalar','Scalar weights by answer role'),('static_member','Constant member weights'),('role_swapped','Wrong-source role weights'),('source_views','Source operation'),('direct_320','Direct gradient (320 batches)')]
    data=dict(confirmation=c,development=dev,labels=labels)
    (PAPER/'data/arithmetic_positions.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=[r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'& \multicolumn{3}{c}{Units replacement} & \multicolumn{3}{c}{Tens replacement} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Method & H & T & P & H & T & P \\',r'\midrule']
    for method,label in labels:
        if method=='source_views':lines.append(r'\midrule')
        vals=[100*next(x for x in c['cells'] if x['initialization']==method and x['operation']==op)[m] for op in ['unit','tens'] for m in ['exact_hybrid','target_digit_success','preserve_digit_success']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\caption{\textbf{Source-profile reuse on new arithmetic questions.} H: complete hybrid answer; T: requested digit; P: preserved digit. Each cell contains 640 interventions sharing 64 question-pair clusters across two prompt forms and five fixed SAEs. The first four methods use the same 64 directly ranked candidates and response bank. Source learning and the 320-batch direct reference retain their original fitting recipes.}',r'\label{tab:arithmetic_positions}',r'\end{anchoredtable}'])
    (PAPER/'tables/arithmetic_positions.tex').write_text('\n'.join(lines)+'\n')
    inventory,evidence=[],[]
    for run in sorted((ROOT/'runs').glob('REFORM_R36_*')):
        status=json.loads((run/'status.json').read_text());assert status['status'] in ['PASS','FAIL','CUT']
        summary=json.loads((run/'metrics.summary.json').read_text());assert hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest()==summary['metrics_raw_sha256']
        inventory.append(dict(run=run.relative_to(ROOT).as_posix(),status=status,summary=summary))
        evidence.extend(p.relative_to(ROOT).as_posix() for p in run.iterdir() if p.is_file() and p.suffix in ['.json','.npz','.jsonl'])
    (ART/'R36_RUNS.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),runs=inventory),indent=2)+'\n')
    evidence += ['artifacts/correspondence_reform_20260913/'+x for x in ['R36_CONFIRMATION_FREEZE.json','R36_RUNS.json','r36_development.json','r36_development_outcomes.npz','r36_confirmation/confirmation.json','r36_confirmation/cluster_outcomes.npz','r36_peap_reading/identity.json']]
    evidence += ['scripts/'+x for x in ['arithmetic_position_relation.py','analyze_arithmetic_positions.py','arithmetic_position_paper.py','plot_arithmetic_positions.py','run_arithmetic_digit_components.py','analyze_arithmetic_reuse_confirmation.py']]
    evidence += ['paper/'+x for x in ['data/arithmetic_positions.json','tables/arithmetic_positions.tex','sections/arithmetic_response_results.tex','sections/arithmetic_position_findings.tex','figures/arithmetic_positions.pdf']]
    (PAPER/'data/arithmetic_positions_evidence.json').write_text(json.dumps(dict(id='arithmetic_role_memberships',paper='Functional source profiles and role-dependent member participation',result=c['primary'],scope=c['scope'],evidence=evidence),indent=2)+'\n')
    fp=PAPER/'figures/FIGURE_MANIFEST.json';fm=json.loads(fp.read_text());paths=['figures/arithmetic_positions.'+x for x in ['pdf','svg','png']]
    fm['outputs']=[x for x in fm['outputs'] if x['path'] not in paths]
    fm['outputs'].extend(dict(path=p,bytes=(PAPER/p).stat().st_size,sha256=hashlib.sha256((PAPER/p).read_bytes()).hexdigest()) for p in paths)
    fm.setdefault('additional_sources',{})['arithmetic_positions']=dict(path='data/arithmetic_positions.json',sha256=hashlib.sha256((PAPER/'data/arithmetic_positions.json').read_bytes()).hexdigest(),generator='scripts/plot_arithmetic_positions.py')
    fp.write_text(json.dumps(fm,indent=2)+'\n')
    print(json.dumps(dict(primary=c['primary'],runs=len(inventory))))
if __name__=='__main__':main()
