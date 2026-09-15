"""Collect frozen write realizations on identical exposed carry requests."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
CONDITIONS=['same_answer_opposite_carry','same_carry_different_answer']


def main():
    analyses={}; cells={}; runs={}; raw={}
    for tag in ['teacher','contextual','capacity']:
        for arity in [2,3]:
            d=json.loads((ART/f'r53_{tag}_{arity}.json').read_text())
            analyses[f'{tag}_{arity}']=d
            p=ROOT/d['run']
            assert hashlib.sha256((p/'metrics.raw.jsonl').read_bytes()).hexdigest()==d['raw_sha256']
            runs[tag]=p
            for c in d['cells']:
                key=(arity,c['method'],c['condition'])
                if key in cells: assert cells[key]['success']==c['success']
                cells[key]=c
        raw[tag]=[json.loads(x) for x in (p/'metrics.raw.jsonl').read_text().splitlines()]
    # Verify exact requests and discrete baseline outcomes before pooling analyses.
    fields=['recipient_question','donor_question','expected','answer','correct','unit_preserved','retained_original']
    def records(tag,method):
        return {(r['operation'],r['row_id']):[r[k] for k in fields] for r in raw[tag]
                if r['kind']=='rule_intervention' and r['method']==method}
    assert records('teacher','code_read_raw_write')==records('contextual','code_read_raw_write')==records('capacity','code_read_raw_write')
    assert len(records('capacity','code_read_raw_write'))==512
    panel=json.loads((runs['capacity']/'RULE_PANEL.json').read_text())
    meta={p['component']:p for p in panel['canonical_pairs']}
    paired=[]
    for arity in [2,3]:
        for reference,ref_tag in [('adaptive_native_64','contextual'),('adaptive_native_128','contextual'),('readwrite_code_64','teacher')]:
            for condition in CONDITIONS:
                def grouped(tag,method):
                    rr=[r for r in raw[tag] if r['kind']=='rule_intervention' and r['method']==method
                        and r['operation']==condition and len(r['recipient_question'])==arity]
                    return {cid:np.mean([r['correct'] for r in rr if r['component']==cid]) for cid in sorted({r['component'] for r in rr})}
                a=grouped('capacity','adaptive_native_512'); b=grouped(ref_tag,reference)
                assert list(a)==list(b)
                ids=list(a); diff=100*np.array([a[i]-b[i] for i in ids])
                labels=[meta[i].get('carry_group',meta[i].get('stratum','all')) for i in ids]
                rng=np.random.default_rng(9491502); weights=np.zeros((10000,len(ids)))
                for label in sorted(set(labels)):
                    jj=[i for i,x in enumerate(labels) if x==label];n=len(jj)
                    weights[:,jj]=rng.multinomial(n,np.full(n,1/n),10000)/len(ids)
                paired.append(dict(arity=arity,method='adaptive_native_512',reference=reference,condition=condition,
                    difference_points=float(diff.mean()),interval_points=np.quantile(weights@diff,[.025,.975]).tolist()))
    diagnostics=[]; costs=[]
    for tag,p in runs.items():
        d=json.loads((p/'metrics.summary.json').read_text()); assert d['status']=='PASS'
        costs.append(dict(run=p.relative_to(ROOT).as_posix(),status=json.loads((p/'status.json').read_text()),
                          **{k:d[k] for k in ['wall_seconds','process_cpu_seconds','peak_allocated_bytes','rows','sequence_forwards','token_forwards']}))
        for f in p.glob('*_native_execution.json'):
            rr=json.loads(f.read_text())['rows'];assert len(rr)==512
            assert min(r['minimum_final_code'] for r in rr)>=-1e-6
            kkt=[r['relative_projected_gradient'] for r in rr if 'relative_projected_gradient' in r]
            if kkt: assert max(kkt)<=1e-5
            diagnostics.append(dict(method=f.name.removesuffix('_native_execution.json'),rows=len(rr),
                actual_changed_counts=sorted(set(sum(float(x)!=0 for x in r['increments']) for r in rr)),
                union_members=len({i for r in rr for i,x in zip(r['members'],r['increments']) if float(x)!=0}),
                mean_field_error=float(np.mean([r['relative_field_error'] for r in rr])),
                max_projected_gradient=max(kkt) if kkt else None,
                solve_seconds=sum(r['batch_solve_seconds']/r['batch_size'] for r in rr),
                minimum_final_code=min(r['minimum_final_code'] for r in rr)))
    old=json.loads((ART/'R52_COMBINED_RESULTS.json').read_text())
    rows=old['table'][:10]
    names=[('teacher_signed_64','Teacher signed, 64'),('teacher_signed_128','Teacher signed, 128'),
           ('teacher_unclipped_64','Teacher signed, unclipped 64'),
           ('teacher_positive_64','Teacher positive, 64'),('teacher_positive_128','Teacher positive, 128'),
           ('adaptive_native_64','Contextual projection, 64'),('adaptive_native_128','Contextual projection, 128'),
           ('adaptive_native_512','Contextual projection, 512'),('reencode_native','Re-encoding')]
    for method,label in names:
        row=dict(method=method,label=label)
        for arity in [2,3]:
            for condition,suffix in zip(CONDITIONS,['change','preserve']):
                row[f'arity{arity}_{suffix}']=100*cells[arity,method,condition]['success']['mean']
        rows.append(row)
    with (ROOT/'paper/data/arithmetic_write_realization.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tex=['\\begin{table}[t]','\\centering\\small','\\begin{tabular}{lrrrr}','\\toprule',
         '& \\multicolumn{2}{c}{Two operands} & \\multicolumn{2}{c}{Three operands} \\\\',
         'Source operation & Change & Preserve & Change & Preserve \\\\','\\midrule']
    for i,row in enumerate(rows):
        if i in [10,15]:tex.append('\\midrule')
        values=[row[f'arity{a}_{s}'] for a in [2,3] for s in ['change','preserve']]
        tex.append(row['label']+' & '+' & '.join(f'{v:.2f}' for v in values)+' \\\\')
    tex += ['\\bottomrule','\\end{tabular}',
      '\\caption{Source writing on identical exposed requests. Upper block: prior source-fit comparisons; refitted writers use 405 mixed source-fit questions and 256 extra backward batches. The original raw projection uses the earlier two-operand fit set. Middle: fixed supports approximate the frozen raw writers, with a shared 64-code reader. Bottom: input-dependent native realizations of the same raw request, with full code access and the stated changed-code allowance. Re-encoding changes at most 128 codes. New realizations use no output fitting.}',
      '\\label{tab:carry_readwrite}','\\end{table}']
    (ROOT/'paper/tables/arithmetic_write_realization.tex').write_text('\n'.join(tex)+'\n')
    prep=json.loads((ART/'r53_teacher_writers_v1/TEACHER_WRITERS.json').read_text())
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),analyses=analyses,table=rows,
        paired_capacity_contrasts=paired,native_diagnostics=diagnostics,costs=costs,teacher_preparation=prep,
        total_driver_seconds=sum(c['wall_seconds'] for c in costs),total_process_cpu_seconds=sum(c['process_cpu_seconds'] for c in costs),
        replay_checks='All 512 code-reader/raw-writer reference request/outcome records identical across three runs; source-native reference replays too.',
        scope='Exposed source1 development, adaptively chosen capacities; no new FCC result or independent confirmation. Fixed-bank and per-input budgets differ.')
    assert records('teacher','readwrite_code_64')==records('contextual','readwrite_code_64')
    for p in [ART/'R53_COMBINED_RESULTS.json',ROOT/'paper/data/arithmetic_write_realization.json']:
        p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(costs=costs,diagnostics=diagnostics,paired=[x for x in paired if x['arity']==3])))


if __name__=='__main__':main()
