"""All-case frozen natural-pair summary; no outcome filtering or refitting."""
from pathlib import Path
import hashlib,json,statistics as st
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/colon_natural_20260906'
OPS=['field_component','clause_component','sum','difference']
METHODS=['best_atom','geometric_atom','dynamic_pair_ridge','geometric_pair_ridge','shared16','same_support_ridge','full','raw']
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rows_by_seed={};runs=[];cells=[];source_ref=None
    for seed in [1,2]:
        run=ROOT/'runs'/f'F4_colon_natural_apply_s{seed}_v1_20260906'
        status=json.loads((run/'metrics.summary.json').read_text());assert status['status']=='PASS'
        rows=[json.loads(x) for x in (run/'metrics.raw.jsonl').read_text().splitlines()];rows_by_seed[seed]=rows
        cfg=json.loads((run/'config.resolved.json').read_text());parent=ROOT/cfg['frozen_map']['path']
        with np.load(run/'coefficients.npz') as a,np.load(parent/'coefficients.npz') as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
            supports={m:int(np.count_nonzero(np.linalg.norm(a[m],axis=1))) for m in METHODS}
        cur={(r['case_id'],r['operator']):{k:r[k] for k in ['source_activation','source_difference','common_dose','source_kl','normalized_kl_error']} for r in rows if r['method']=='raw'}
        if source_ref is None:source_ref=cur
        else:assert cur==source_ref
        lookup={(r['case_id'],r['operator'],r['method']):r for r in rows}
        for op in OPS:
            cell=dict(seed=seed,operator=op,methods={})
            for m in METHODS:
                group=[r for r in rows if r['operator']==op and r['method']==m]
                ratios=[r['normalized_kl_error'] for r in group if r['normalized_kl_error'] is not None]
                cell['methods'][m]=dict(n=len(group),defined_ratios=len(ratios),median_ratio=st.median(ratios) if ratios else None,median_candidate_kl=st.median(r['candidate_kl'] for r in group),mean_candidate_kl=st.mean(r['candidate_kl'] for r in group),mean_vector_error=st.mean(r['vector_squared_error'] for r in group),support=supports[m],median_source_kl=st.median(r['source_kl'] for r in group),source_kl_below_1e_minus6=sum(r['source_kl']<1e-6 for r in group))
            cell['shared_case_wins']={m:sum(r['candidate_kl']<lookup[r['case_id'],op,m]['candidate_kl'] for r in rows if r['operator']==op and r['method']=='shared16') for m in METHODS if m!='shared16'}
            cells.append(cell)
        runs.append(dict(run=run.name,wall_seconds=status['wall_seconds'],forwards=status['forwards'],rows=status['rows'],raw_sha256=H(run/'metrics.raw.jsonl'),frozen_coefficients_exact=True))
    r0={r['case_id']:r for r in rows_by_seed[1]};pairs=[]
    for i in range(12):
        field,clause=r0[2*i],r0[2*i+1]
        d=np.array(clause['source_activation'])-field['source_activation']
        pairs.append(dict(pair=i,field_document=field['document_id'],clause_document=clause['document_id'],field_text=field['text'],clause_text=clause['text'],field_activation=field['source_activation'],clause_activation=clause['source_activation'],clause_minus_field=d.tolist(),predictions=[bool(d[0]<0),bool(d[1]>0)]))
    payload=dict(cells=cells,runs=runs,pairs=pairs,source_hypothesis_successes=[sum(p['predictions'][j] for p in pairs) for j in range(2)],source_zero_differences=[sum(p['clause_minus_field'][j]==0 for p in pairs) for j in range(2)],source_and_raw_exact_between_targets=True,total_forwards=sum(r['forwards'] for r in runs),total_wall_seconds=sum(r['wall_seconds'] for r in runs),scope='All24prefixes/12cross-document pairs, reciprocal directions and shared seeds dependent; no semantic/native identification; lexical proxies include time/citation counterexamples, all retained.')
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for c in cells:print(c['seed'],c['operator'],{m:round(c['methods'][m]['median_ratio'],6) if c['methods'][m]['median_ratio'] is not None else None for m in METHODS})
    print(json.dumps({k:payload[k] for k in ['source_hypothesis_successes','source_zero_differences','total_forwards','total_wall_seconds']}))
    lines=['# Frozen two-component FCC on new natural contexts','', 'The source-only fixed sampler used204new documents/1024sequences, with149short-field and131long-clause eligible positions; deterministic hash order selected12pairs/24different documents. Selection used no source activations or target outcomes. All cases, including lexical time/citation proxy failures, remain.','', '| Target | Operator | Shared | Dynamic pair | Geometric pair | Full | Raw |','|---|---|---:|---:|---:|---:|---:|']
    for c in cells:lines.append('| '+str(c['seed'])+' | '+c['operator']+' | '+' | '.join(f"{c['methods'][m]['median_ratio']:.6f}" for m in ['shared16','dynamic_pair_ridge','geometric_pair_ridge','full','raw'])+' |')
    lines+=['','The repeatable positive scope is component2: shared normalized medians0.009636/0.013113 versus dynamic pair0.255609/0.486929 and full0.060812/0.046704. Shared beats dynamic pair in20/24 and22/24recipient cases, full in14/24 and16/24. Both absolute-KL medians also favor shared for this component. Raw is stronger on every operation median.','','Component1 normalized medians favor dynamic pair/full in both targets; four recipient source KL values are below1e-6 and none is discarded. On target1 component1, shared wins19/24case comparisons versus dynamic pair despite the higher median ratio: different aggregates answer different questions. Sum/difference favor shared over dynamic pair in both targets, but target1 apparent normalized-median gains over full do not survive absolute-KL medians or majority casewise comparisons. Do not assert broad shared dominance.','','Same-support ridge closely follows shared, so shrinkage alone is not a supported explanation; joint support selection and changed donor distribution are next hypotheses, not established causes. A fixed total16budget comparison of shared versus per-output support selection can test the tradeoff on development data; any tuned result needs another fresh confirmation.','','Values are medians of candidate KL/source KL, lower is better. summary.json includes all eight methods, absolute KL, weak-source counts and casewise wins; no weak source excluded. Two methods with shared support distinguish support selection from ridge details.','',f"Source input-condition hypotheses succeed{payload['source_hypothesis_successes'][0]}/12 and{payload['source_hypothesis_successes'][1]}/12; exact-zero source differences{payload['source_zero_differences']}. Old authored6/6patterns do not universally transfer. These are lexical format/length proxies, not identified semantic mechanisms.",'','All coefficients exactly equal frozen terminal parents; zero refits. Source quantities and raw endpoint rows exactly agree across targets. No-op and pointwise error-Gram checks pass. Old-panel frozen interface replay reproduces384rows exactly. These checks establish application consistency, not independent scientific review.','',f"New target runs total{payload['total_forwards']}LM forwards/{sum(r['rows'] for r in runs)}rows/{payload['total_wall_seconds']:.6f}s; corpus25,237,678network bytes, no training/new weights/audit. Fixed source pair and two target seeds are not the full five-seed source suite."]
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
