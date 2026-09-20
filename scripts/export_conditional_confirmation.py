from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/final_science_20260920_round02'
PAPER=ROOT/'paper'
METHODS=[('Global profile','initial_refined_cached_source_metric'),
         ('Conditional profile','initial_refined_conditional_source_metric'),
         ('Program-trained execution','task_adapted_tangent')]


def identity(path):
    return dict(path=str(path.relative_to(ROOT)).replace('\\','/'),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    path=ART/'CONDITIONAL_CONFIRMATION_ANALYSIS.json'
    data=json.loads(path.read_text())
    assert data['all_targets']['later_heads']['seeds']==[1,2,3,4,5]
    rows=[]
    for cohort in ['all_targets','trained_comparison']:
        for endpoint,result in data[cohort].items():
            for method,families in result['summary'].items():
                for family,value in families.items():
                    rows.append(dict(cohort=cohort,endpoint=endpoint,method=method,family=family,
                        nrmse=value['nrmse'],lower=value['interval'][0],upper=value['interval'][1],
                        seeds=json.dumps(result['seeds']),by_seed=json.dumps(value['by_seed'])))
    csv_path=PAPER/'data/conditional_profile_confirmation.csv'
    with csv_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    shutil.copyfile(path,PAPER/'data/conditional_profile_confirmation.json')
    table=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r'& \multicolumn{2}{c}{Participation} & \multicolumn{2}{c}{Later classifiers} \\',
           r'Execution & Later & Original & Members & Endpoints \\',r'\midrule']
    for cohort,heading in [('all_targets','Five unchanged targets'),('trained_comparison','Matched four-target cohort')]:
        table.append(r'\multicolumn{5}{l}{'+heading+r'} \\')
        for label,method in METHODS:
            if method not in data[cohort]['later_heads']['summary']: continue
            values=[data[cohort][ep]['summary'][method][family]['nrmse'] for ep,family in
                    [('later_heads','participation'),('original_head','participation'),('later_heads','members'),('later_heads','endpoints')]]
            table.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r' \\')
        table.append(r'\midrule' if cohort=='all_targets' else r'\bottomrule')
    table.append(r'\end{tabular}')
    table_path=PAPER/'tables/conditional_profile_confirmation.tex'
    table_path.write_text('\n'.join(table)+'\n')
    macros=[]
    for suffix,(_,method) in zip(['Global','Weighted'],METHODS[:2]):
        value=data['all_targets']['later_heads']['summary'][method]['participation']['nrmse']
        macros.append('\\newcommand{\\Conditional'+suffix+'}{'+f'{value:.3f}'+'}')
    contrast=next(c for c in data['all_targets']['later_heads']['contrasts']
                  if c['family']=='participation' and c['reference']==METHODS[0][1])
    for suffix,value in [('Reduction',contrast['reduction']),('Lower',contrast['interval'][0]),('Upper',contrast['interval'][1])]:
        macros.append('\\newcommand{\\Conditional'+suffix+'}{'+f'{value:.3f}'+'}')
    (PAPER/'tables/conditional_profile_values.tex').write_text('\n'.join(macros)+'\n')
    provenance=dict(sources=[identity(path),identity(ART/'CONDITIONAL_CONFIRMATION_FREEZE.json')],
        generator=identity(Path(__file__).resolve()),outputs=[identity(csv_path),identity(table_path),identity(PAPER/'data/conditional_profile_confirmation.json')])
    manifest_path=PAPER/'data/DATA_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text()); manifest['conditional_profile_confirmation']=provenance
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    index_path=PAPER/'EVIDENCE_INDEX.json'
    index=json.loads(index_path.read_text()); index['claims']=[c for c in index['claims'] if c['id']!='conditional_profile_confirmation']
    index['claims'].append(dict(id='conditional_profile_confirmation',paper='Section4.1 and AppendixE.5',
        result='Frozen comparison of conditional and global source sensitivity on a complete published human program, five unchanged targets, new biographies and new requests',
        evidence=provenance['sources']+provenance['outputs'],statistics=data['inference']))
    index_path.write_text(json.dumps(index,indent=2)+'\n')
    (ART/'CONDITIONAL_EXPORT.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),**provenance),indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),primary=contrast)))


if __name__=='__main__': main()
