"""Export the response objective study from retained arithmetic runs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/correspondence_reform_20260913'
PAPER = ROOT/'paper'


def main():
    files = ['r35_response_original.json','r35_symmetric_original.json',
             'r35_profile_identity_original.json','r35_digit_original.json']
    summaries = [json.loads((ART/f).read_text()) for f in files]
    cells = [c for s in summaries for c in s['cells']]
    labels = [('response_field_prior','Trajectory field'),
              ('response_field_teacher','Answer-prefix field'),
              ('source_views','Source operation'),('direct_320','Direct gradient, 320 batches'),
              ('response_response','Two margins: finite'),
              ('response_response_anchor','Two margins: finite + field 0.1'),
              ('response_finite_anchor1','Two margins: finite + field 1'),
              ('response_linear_response','Two margins: local source'),
              ('response_linear_anchor','Two margins: local + field 0.1'),
              ('response_direct_profile','Two margins: direct candidates'),
              ('response_direct_scalar','Two margins: scalar weights'),
              ('response_direct_swapped','Two margins: swapped source'),
              ('response_digit_anchor','All digits: finite + field 0.1'),
              ('response_digit_anchor1','All digits: finite + field 1'),
              ('response_digit_linear','All digits: local + field 0.1'),
              ('response_digit_direct','All digits: direct candidates'),
              ('response_digit_scalar','All digits: scalar weights'),
              ('response_digit_swapped','All digits: swapped source')]
    raw_paths = sorted(set(s['run'].replace('\\','/')+'/metrics.raw.jsonl' for s in summaries))
    data = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),cells=cells,
                labels=labels,raw_paths=raw_paths,
                scope='Adaptive development on the previously exposed original arithmetic panel. Equal averages over requests, two prompt forms and five fixed SAEs; no fresh confirmation.')
    import numpy as np
    methods=[m for m,_ in labels]
    panel=json.loads((ROOT/raw_paths[0]).with_name('panel.json').read_text())
    def cluster(p):return tuple(panel['rows'][i][k] for i in [p['recipient'],p['donor']] for k in ['a','b'])
    ids=sorted({cluster(p) for p in panel['pairs']});assert len(ids)==64
    out=np.full((len(methods),64,2,2,5,3),np.nan)
    for file in raw_paths:
        raw=ROOT/file
        assert json.loads(raw.with_name('panel.json').read_text())['pairs']==panel['pairs']
        for line in raw.read_text().splitlines():
            r=json.loads(line)
            if r['kind']!='source_patch' or r['seed']==0:continue
            m=r['method'];m='response_'+m.split('_',2)[2] if m.startswith('response_s') else m
            if m not in methods:continue
            p=panel['pairs'][r['row_id']]
            ix=(methods.index(m),ids.index(cluster(p)),['unit','tens'].index(r['operation']),p['template'],r['seed']-1)
            assert np.isnan(out[ix]).all()
            out[ix]=[r[k] for k in ['exact_hybrid','target_digit_success','preserve_digit_success']]
    assert np.isfinite(out).all()
    rng=np.random.default_rng(9350914);draws=rng.integers(64,size=(10000,64))
    comparisons=[('response_direct_profile','response_field_prior'),('response_direct_profile','response_direct_scalar'),
                 ('response_direct_profile','response_direct_swapped'),('response_finite_anchor1','response_field_prior'),
                 ('response_digit_direct','response_direct_profile'),('response_digit_direct','response_digit_scalar'),
                 ('response_digit_direct','response_digit_swapped')]
    contrasts=[]
    for a,b in comparisons:
        delta=(out[methods.index(a)]-out[methods.index(b)]).mean((1,2,3))
        ci=np.quantile(delta[draws].mean(1),[.025,.975],axis=0)*100
        contrasts.append(dict(reference=a,comparator=b,metrics={m:dict(difference_points=float(delta[:,k].mean()*100),
             interval_points=ci[:,k].tolist()) for k,m in enumerate(['exact_hybrid','target_digit_success','preserve_digit_success'])}))
    data['descriptive_paired_contrasts']=contrasts
    np.savez_compressed(ART/'r35_cluster_outcomes.npz',outcomes=out,methods=np.array(methods),operand_pairs=np.array(ids))
    (PAPER/'data/arithmetic_response.json').write_text(json.dumps(data,indent=2)+'\n')
    lines = [r'\begin{anchoredtable}\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',
             r'& \multicolumn{3}{c}{Units replacement} & \multicolumn{3}{c}{Tens replacement} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Method & H & T & P & H & T & P \\',r'\midrule']
    for i,(method,label) in enumerate(labels):
        if i in [4,12]:lines.append(r'\midrule')
        values=[100*next(c for c in cells if c['method']==method and c['operation']==op)[m]
                for op in ['unit','tens'] for m in ['exact_hybrid','target_digit_success','preserve_digit_success']]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\caption{\textbf{Functional response objectives and actual digit replacement.} H: complete hybrid answer; T: requested digit; P: preserved digit. All methods allow 64 members. Each cell includes 640 interventions, sharing 64 question-pair clusters across the two prompt forms and five fixed SAEs. The two-margin bank uses 32 shared backward batches; the all-digit bank uses 320. The source and direct-320 references retain their earlier fitting budgets. This is a development comparison.}',
              r'\label{tab:arithmetic_response}',r'\end{anchoredtable}']
    (PAPER/'tables/arithmetic_response_results.tex').write_text('\n'.join(lines)+'\n')
    inventory,evidence=[],[]
    for run in sorted((ROOT/'runs').glob('REFORM_R35_*')):
        status=json.loads((run/'status.json').read_text());assert status['status'] in ['PASS','FAIL']
        summary=json.loads((run/'metrics.summary.json').read_text())
        raw=run/'metrics.raw.jsonl';assert hashlib.sha256(raw.read_bytes()).hexdigest()==summary['metrics_raw_sha256']
        inventory.append(dict(run=run.relative_to(ROOT).as_posix(),status=status,summary=summary))
        evidence.extend(p.relative_to(ROOT).as_posix() for p in run.iterdir() if p.is_file() and
                        (p.suffix in ['.json','.npz','.jsonl'] or p.name=='traceback.log'))
    (ART/'R35_RUNS.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),runs=inventory),indent=2)+'\n')
    evidence += ['artifacts/correspondence_reform_20260913/'+f for f in files+['r35_cluster_outcomes.npz']]
    evidence += ['scripts/arithmetic_response_relation.py','scripts/arithmetic_response_paper.py',
                 'scripts/plot_arithmetic_response.py','scripts/run_arithmetic_digit_components.py',
                 'scripts/analyze_arithmetic_source_learning.py','paper/data/arithmetic_response.json',
                 'paper/tables/arithmetic_response_results.tex','paper/sections/arithmetic_response_results.tex',
                 'paper/figures/arithmetic_response.pdf','artifacts/correspondence_reform_20260913/R35_RUNS.json']
    claim=dict(id='arithmetic_response_objectives',paper='Functional response representation and source calibration',
               result='Response-objective and source-identity comparisons with the same fit pairs, fixed SAE cohorts and candidate budgets; actual requested and preserved digits are separately observed.',
               scope=data['scope'],evidence=evidence)
    (PAPER/'data/arithmetic_response_evidence.json').write_text(json.dumps(claim,indent=2)+'\n')
    print(json.dumps({m:sum(c['exact_hybrid'] for c in cells if c['method']==m)/2 for m,_ in labels}))


if __name__=='__main__':main()
