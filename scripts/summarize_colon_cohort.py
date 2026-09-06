"""Frozen four-target, document-pair-dependent natural confirmation summary."""
from pathlib import Path
import hashlib,json,statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/colon_cohort_20260906'
OPS=['field_component','clause_component','sum','difference']
SEEDS=[1,2,4,5]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cells=[];runs=[];allrows={};source_ref=None
    for seed in SEEDS:
        run=ROOT/'runs'/f'F4_colon_cohort_apply_s{seed}_v1_20260906';summary=json.loads((run/'metrics.summary.json').read_text());assert summary['status']=='PASS'
        cfg=json.loads((run/'config.resolved.json').read_text());parent=ROOT/cfg['frozen_map']['path']
        rows=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()];allrows[seed]={(r['case_id'],r['operator'],r['method']):r for r in rows}
        with np.load(run/'coefficients.npz') as a,np.load(parent/'coefficients.npz') as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
            supports={m:int(np.count_nonzero(np.linalg.norm(a[m],axis=1))) for m in a.files if m!='source_decoder'}
        ref={(r['case_id'],r['operator']):{k:r[k] for k in ['source_activation','source_difference','common_dose','source_kl','candidate_kl','normalized_kl_error']} for r in rows if r['method']=='raw'}
        if source_ref is None:source_ref=ref
        else:assert ref==source_ref
        for op in OPS:
            cell=dict(seed=seed,operator=op,methods={})
            for m in supports:
                group=[r for r in rows if r['operator']==op and r['method']==m];v=[r['normalized_kl_error'] for r in group if r['normalized_kl_error'] is not None]
                cell['methods'][m]=dict(n=len(group),defined_ratios=len(v),median_ratio=st.median(v) if v else None,median_absolute_kl=st.median(r['candidate_kl'] for r in group),mean_absolute_kl=st.mean(r['candidate_kl'] for r in group),support=supports[m],weak_source_kl_below_1e_minus6=sum(r['source_kl']<1e-6 for r in group),mean_vector_error=st.mean(r['vector_squared_error'] for r in group))
            cell['shared_case_wins']={m:sum(r['candidate_kl']<allrows[seed][r['case_id'],op,m]['candidate_kl'] for r in rows if r['operator']==op and r['method']=='shared16') for m in supports if m!='shared16'};cells.append(cell)
        runs.append(dict(run=run.name,wall_seconds=summary['wall_seconds'],forwards=summary['forwards'],rows=summary['rows'],raw_sha256=H(run/'metrics.raw.jsonl'),coefficients_exact=True,supports=supports))
    cases=json.loads((OUT/'prepared_inputs.json').read_text())['cases'];npairs=len(cases)//2
    assert len(set(c['document_id'] for c in cases))==len(cases)
    source=[]
    for i in range(npairs):
        field=allrows[1][2*i,'clause_component','raw'];clause=allrows[1][2*i+1,'clause_component','raw']
        delta=np.array(clause['source_activation'])-field['source_activation']
        source.append(dict(pair=i,documents=[field['document_id'],clause['document_id']],field_text=field['text'],clause_text=clause['text'],field_activation=field['source_activation'],clause_activation=clause['source_activation'],clause_minus_field=delta.tolist(),predictions=[bool(delta[0]<0),bool(delta[1]>0)]))
    boot={};rng=np.random.default_rng(20260906);sample=rng.integers(0,npairs,size=(10000,npairs))
    methods=list(runs[0]['supports'])
    for m in methods:
        if m=='shared16':continue
        delta=np.array([st.mean(allrows[s][2*i+j,'clause_component','shared16']['candidate_kl']-allrows[s][2*i+j,'clause_component',m]['candidate_kl'] for s in SEEDS for j in [0,1]) for i in range(npairs)])
        means=delta[sample].mean(1);boot[m]=dict(pair_differences=delta.tolist(),mean_absolute_kl_difference=float(delta.mean()),percentile95=np.quantile(means,[.025,.975]).tolist(),pairs_shared_better=int(np.sum(delta<0)))
    result=dict(cells=cells,runs=runs,source_pairs=source,source_prediction_successes=[sum(p['predictions'][j] for p in source) for j in [0,1]],source_zero_differences=[sum(p['clause_minus_field'][j]==0 for p in source) for j in [0,1]],bootstrap_component2=boot,bootstrap_unit='Document pair: average reciprocal recipients and all four dependent targets first;10000resamples seed20260906. Descriptive interval conditional on one fixed source/four targets, not seed-population coverage.',all_source_and_raw_identical=True,total_forwards=sum(r['forwards'] for r in runs),total_rows=sum(r['rows'] for r in runs),total_wall_seconds=sum(r['wall_seconds'] for r in runs),scope='Prospective fresh-context confirmation after selecting component2 on earlier development. Five controlled SAE assets, one source pair/four dependent targets; all four operations and all ten frozen methods retained.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# Four-target fresh natural confirmation','',result['scope'],'','| Target | Operation | Shared | Dynamic pair | Separate own | Separate union | Full | Raw |','|---|---|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        vals=[c['methods'][m]['median_ratio'] for m in ['shared16','dynamic_pair_ridge','separate8_ridge','union16_ridge','full','raw']]
        lines.append('| '+str(c['seed'])+' | '+c['operator']+' | '+' | '.join('undefined' if v is None else f'{v:.6f}' for v in vals)+' |')
        if c['operator']=='clause_component':print(c['seed'],dict(zip(['shared','dynamic','separate','union','full','raw'],vals)),c['shared_case_wins'])
    lines+=['','Medians use the original candidate/source KL ratio. Absolute medians, absolute means, all10methods, paired wins and weak-source counts are retained in summary.json. No input exclusion follows source/target outcomes.','',f"Source lexical predictions: {result['source_prediction_successes']} successes out of{npairs}pairs; zero activation differences {result['source_zero_differences']}. These are unvalidated short-field/long-clause proxies, not two isolated semantic mechanisms.",'','| Control | Mean shared-minus-control absolute KL | Descriptive95% interval | Pairs shared better |','|---|---:|---|---:|']
    for m,d in boot.items():lines.append(f"| {m} | {d['mean_absolute_kl_difference']:.8g} | [{d['percentile95'][0]:.8g}, {d['percentile95'][1]:.8g}] | {d['pairs_shared_better']}/{npairs} |")
    lines+=['',result['bootstrap_unit'],'',f"Fresh application totals: {result['total_forwards']}LM/{result['total_rows']}rows/{result['total_wall_seconds']:.6f}s. Every coefficient equals its frozen parent; every raw/source row is identical across targets, no-op/Gram and run contracts pass. These establish application consistency, not independent validation. Missing target4/5 strong-control fits and old-data replay are separate preparation costs."]
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(source_successes=result['source_prediction_successes'],total_seconds=result['total_wall_seconds'],bootstrap=boot)))
if __name__=='__main__':main()
