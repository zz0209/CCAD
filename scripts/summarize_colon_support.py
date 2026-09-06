"""Compare support selection on both exposed panels, keeping all old controls."""
import hashlib,json,statistics as st
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/colon_support_tradeoff_20260906'
OPS=['field_component','clause_component','sum','difference']
METHODS=['shared16','same_support_ridge','separate8_ridge','union16_ridge','dynamic_pair_ridge','full','raw']
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cells=[];runs=[]
    for seed in [1,2]:
        run=ROOT/'runs'/f'F4_colon_support_tradeoff_s{seed}_v1_20260906'
        stats=json.loads((run/'metrics.summary.json').read_text());assert stats['status']=='PASS'
        rows=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()]
        lookup={(r['panel'],r['original_case_id'],r['operator'],r['method']):r for r in rows}
        with np.load(run/'coefficients.npz') as a,np.load(ROOT/'runs'/f'F4_function_curve_s{seed}_step8192_v1_20260906'/'coefficients.npz') as b:
            assert all(np.array_equal(a[k],b[k]) for k in b.files)
            supports={m:int(np.count_nonzero(np.linalg.norm(a[m],axis=1))) for m in a.files if m!='source_decoder'}
        for panel,parent in [('authored_development',f'F4_function_curve_s{seed}_step8192_v1_20260906'),('natural_development',f'F4_colon_natural_apply_s{seed}_v1_20260906')]:
            old=[json.loads(x) for x in (ROOT/'runs'/parent/'metrics.raw.jsonl').read_text().splitlines()]
            for r in old:
                new=lookup[panel,r['case_id'],r['operator'],r['method']]
                assert new['donor']==r['donor']+(12 if panel=='natural_development' else 0)
                assert all(new[k]==v for k,v in r.items() if k not in ('case_id','donor')),(panel,r['case_id'],r['method'])
            for op in OPS:
                group=[r for r in rows if r['panel']==panel and r['operator']==op];cell=dict(seed=seed,panel=panel,operator=op,methods={})
                for m in supports:
                    v=[r for r in group if r['method']==m];ratios=[r['normalized_kl_error'] for r in v if r['normalized_kl_error'] is not None]
                    cell['methods'][m]=dict(median_ratio=st.median(ratios) if ratios else None,defined_ratios=len(ratios),n=len(v),median_absolute_kl=st.median(r['candidate_kl'] for r in v),mean_absolute_kl=st.mean(r['candidate_kl'] for r in v),support=supports[m],mean_vector_error=st.mean(r['vector_squared_error'] for r in v))
                cell['wins_against_shared_same_ridge']={m:sum(r['candidate_kl']<lookup[panel,r['original_case_id'],op,'same_support_ridge']['candidate_kl'] for r in group if r['method']==m) for m in ['separate8_ridge','union16_ridge']}
                cells.append(cell)
        sep=json.loads((run/'separate_support_selection.json').read_text());fit=json.loads((run/'fit_metadata.json').read_text())
        runs.append(dict(run=run.name,wall_seconds=stats['wall_seconds'],forwards=stats['forwards'],raw_sha256=H(run/'metrics.raw.jsonl'),old_coefficients_and_all_old_endpoint_rows_exact=True,per_output_supports=sep['per_output_supports'],union=sep['union'],separate_selection_seconds=sep['fit_seconds'],shared_selection_seconds=fit['joint_fit']['fit_seconds'],separate_path_steps=[len(o['path']) for o in sep['outputs']],shared_path_steps=len(fit['joint_fit']['path']),supports=supports))
    result=dict(cells=cells,runs=runs,total_forwards=sum(r['forwards'] for r in runs),total_wall_seconds=sum(r['wall_seconds'] for r in runs),scope='Development only: same source pair, both previously exposed panels. Same <=16 ceiling, not exact actual support size or solver-call matching. All failures retained.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Shared versus separate support selection','',result['scope'],'','| Target | Panel | Operation | Shared+ridge | Separate own | Separate union | Full |','|---|---|---|---:|---:|---:|---:|']
    for c in cells:
        lines.append('| '+str(c['seed'])+' | '+c['panel']+' | '+c['operator']+' | '+' | '.join(f"{c['methods'][m]['median_ratio']:.6f}" for m in ['same_support_ridge','separate8_ridge','union16_ridge','full'])+' |')
        print(c['seed'],c['panel'],c['operator'],{m:round(c['methods'][m]['median_ratio'],6) for m in ['same_support_ridge','separate8_ridge','union16_ridge','full']})
    lines+=['','Natural component1 improves under separate own supports in both targets: ratio medians0.607980->0.143150 and0.660184->0.381301; absolute-KL medians7.529e-5->6.308e-5 and1.4040e-4->6.221e-5; case wins16/24 and14/24. The effect is not purely a weak-denominator aggregate. Natural component2 gets worse in both targets (0.009353->0.075932 and0.012520->0.056272); absolute medians and case counts also favor shared. Thus the per-output alternative is a tradeoff, not a universal replacement.','','Authored target1 strongly favors shared throughout; authored target2 and natural target2 have operator-dependent counterexamples. Separate union refitting does not consistently improve either separate fits or shared support. Full remains a strong component1 reference and raw remains the strongest overall median reference. Selected-support overlap is1of6/7 for target1,2of8/8 for target2, but these are estimated supports, not known true causal overlap.','','This completed one bounded selection comparison; no alpha/budget grid extension follows. Next prioritize extending the strongest natural component2 result to remaining targets in the existing five-SAE cohort with newly frozen maps/new contexts, retaining component1 and mixture failures. Do not infer five independent replications from one shared source.','','Values use the original normalized KL error median. All10methods, absolute medians/means, vector errors and paired wins remain in summary.json. No source-zero or weak-effect exclusion is added. The independent candidates use two40point relative-alpha paths versus one40point group path, each stopped at the saved horizon; actual iteration counts and wall costs are recorded.','', 'Input/output standardization and per-support candidate scoring reuse fit_joint conventions. Both independent-output fits and union fits then use exactly the same fixed_support_ridge kernel/fraction as same_support_ridge. Source/dose/raw and all eight old methods replay exactly on both panels. This controls changed consumer implementation, but does not equalize actual selected member counts or prove a source of generalization error.','']
    for r in runs:lines.append(f"- {r['run']}: shared{r['supports']['same_support_ridge']}members versus per-output{list(map(len,r['per_output_supports']))}, union{len(r['union'])}; selection seconds shared{r['shared_selection_seconds']:.6f}/separate{r['separate_selection_seconds']:.6f}; wall{r['wall_seconds']:.6f}s.")
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(total_forwards=result['total_forwards'],total_wall_seconds=result['total_wall_seconds'])))
if __name__=='__main__':main()
