"""Export retained R33 results and provenance into the editable manuscript."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PAPER = ROOT / 'paper'
RUNS = {name: ROOT / f'runs/REFORM_R33_qwen_{name}_v1_20260914' for name in
        ['identity_gradient','functional_path','short_budget','fixed_identity',
         'new_range_confirmation','source_range_diagnostic']}


def main():
    now = datetime.now(timezone.utc).isoformat()
    records, inventory, evidence = {}, [], []
    for name, run in RUNS.items():
        status = json.loads((run/'status.json').read_text())
        assert status['status']=='PASS'
        summary = json.loads((run/'metrics.summary.json').read_text())
        raw = run/'metrics.raw.jsonl'
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        assert digest==summary['metrics_raw_sha256']
        records[name] = [json.loads(s) for s in raw.read_text().splitlines()]
        files = ['config.resolved.json','status.json','manifest.json','code_hashes.json',
                 'inputs.json','metrics.raw.jsonl','metrics.summary.json','environment.json',
                 'SOURCE_FREEZE.json','contract_validation.json']
        evidence.extend((run/f).relative_to(ROOT).as_posix() for f in files)
        inventory.append(dict(run=run.relative_to(ROOT).as_posix(),status=status,
                              summary=summary,raw_sha256=digest))

    def row(run_name, method):
        result={}
        for op in ['unit','tens']:
            rr=[r for r in records[run_name] if r['kind']=='source_patch'
                and r['method']==method and r['operation']==op]
            assert len(rr)==640
            result[op]={m:sum(float(r[m]) for r in rr)/len(rr) for m in
                        ['exact_hybrid','target_digit_success','preserve_digit_success']}
        return result

    def table(filename, specifications):
        lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',
               r'& & \multicolumn{3}{c}{Units replacement} & \multicolumn{3}{c}{Tens replacement} \\',
               r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
               r'Method & Budget & H & T & P & H & T & P \\',r'\midrule']
        payload=[]
        for run,method,label,budget in specifications:
            values=row(run,method)
            numbers=[100*values[op][m] for op in ['unit','tens'] for m in
                     ['exact_hybrid','target_digit_success','preserve_digit_success']]
            lines.append(label+' & '+str(budget)+' & '+' & '.join(f'{v:.2f}' for v in numbers)+r' \\')
            payload.append(dict(run=run,method=method,label=label,budget=budget,values=values))
        lines += [r'\bottomrule',r'\end{tabular}']
        (PAPER/'tables'/f'{filename}.tex').write_text('\n'.join(lines)+'\n')
        (PAPER/'data'/f'{filename}.json').write_text(json.dumps(payload,indent=2)+'\n')
        evidence.extend([f'paper/tables/{filename}.tex',f'paper/data/{filename}.json'])

    table('arithmetic_short_budget',[
        ('identity_gradient','adapt_clean_fixed_u16','Fixed translated members','16'),
        ('fixed_identity','adapt_clean_swapped_fixed_u16','Exchanged source function','16'),
        ('short_budget','adapt_target_gradient_s2_u16','Direct gradient','2+14'),
        ('short_budget','adapt_target_gradient_s4_u16','Direct gradient','4+12'),
        ('short_budget','adapt_target_gradient_s8_u16','Direct gradient','8+8')])
    table('arithmetic_path_cost',[
        ('functional_path','adapt_source_path_gradient_u64','Source-functional path','16+48'),
        ('functional_path','adapt_source_swapped_path_gradient_u64','Exchanged source path','16+48'),
        ('functional_path','adapt_target_gradient_u64','Direct gradient','16+48'),
        ('functional_path','adapt_target_gradient_u320','Direct gradient','16+304')])
    table('arithmetic_fixed_confirmation',[
        ('new_range_confirmation','adapt_clean_fixed_u16','Fixed translated members','16'),
        ('new_range_confirmation','adapt_target_gradient_s2_u16','Direct gradient','2+14'),
        ('new_range_confirmation','adapt_clean_swapped_fixed_u16','Exchanged source function','16')])
    table('arithmetic_source_range',[
        ('source_range_diagnostic','adapt_source_original_u0','Original source mask','256 prior'),
        ('source_range_diagnostic','adapt_source_improved_u0','Improved source mask','320 prior')])
    for src,dst in [('r33_identity_analysis/identity.json','arithmetic_identity.json'),
                    ('r33_new_range_confirmation_analysis/confirmation.json','arithmetic_fixed_confirmation_analysis.json')]:
        (PAPER/'data'/dst).write_bytes((ART/src).read_bytes())
        evidence.append('paper/data/'+dst)
    proof=json.loads((ART/'r33_new_range_confirmation_analysis/confirmation.json').read_text())
    evidence.extend(['scripts/arithmetic_identity_paper.py','scripts/analyze_arithmetic_identity.py',
                     'scripts/analyze_arithmetic_reuse_confirmation.py','scripts/plot_arithmetic_identity.py',
                     'scripts/arithmetic_counterfactual_fit.py','scripts/run_arithmetic_digit_components.py',
                     'paper/sections/arithmetic_component_details.tex','paper/sections/main_text.tex',
                     'paper/figures/arithmetic_identity.pdf',
                     'artifacts/correspondence_reform_20260913/r33_new_range_confirmation_analysis/cluster_outcomes.npz',
                     'artifacts/correspondence_reform_20260913/r33_functional_path_analysis/prefix_check.json'])
    claim=dict(id='arithmetic_fixed_membership_identity',paper='Fixed memberships retain source-functional identity on new operand questions',
               scope=proof['scope'],result=proof['primary'],
               both_primary_comparisons_positive=proof['both_primary_comparisons_positive'],evidence=evidence)
    (PAPER/'data/arithmetic_identity_evidence.json').write_text(json.dumps(claim,indent=2)+'\n')
    (ART/'R33_RUNS.json').write_text(json.dumps(dict(written_at_utc=now,runs=inventory),indent=2)+'\n')
    index=json.loads((PAPER/'EVIDENCE_INDEX.json').read_text())
    index['claims']=[c for c in index['claims'] if c['id']!=claim['id']]
    verified={**claim,'evidence':[dict(path=p,bytes=(ROOT/p).stat().st_size,
                                    sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in evidence]}
    index['claims'].append(verified)
    if claim['id'] not in index['current_manuscript_claim_ids']:
        index['current_manuscript_claim_ids'].append(claim['id'])
    # Refresh file identities already present in older entries; original producing
    # code is retained in each run's immutable source_snapshot and code_hashes.
    mutable={p for p in evidence if p.startswith(('paper/','scripts/'))}
    for c in index['claims']:
        for entry in c['evidence']:
            if entry['path'] not in mutable:
                continue
            p=ROOT/entry['path']
            entry.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (PAPER/'EVIDENCE_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(runs=len(inventory),primary=proof['primary'])))


if __name__=='__main__':
    main()
