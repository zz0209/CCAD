"""Complete natural-prefix endpoint table, paired controls and numerical replay."""
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'artifacts/and_natural_confirmation_20260906'
    sp=ROOT/'runs/F4_and_natural_source_v1_20260906';tp=ROOT/'runs/F4_and_natural_targets_v1_20260906'
    source=[json.loads(s) for s in (sp/'metrics.raw.jsonl').read_text().splitlines()]
    target=[json.loads(s) for s in (tp/'metrics.raw.jsonl').read_text().splitlines()]
    tables=[];cases=[];paired=[]
    for family in ['ordinary','constrained']:
        for conj in [' and',' or','both']:
            for m in ['fcc','atom','raw']:
                rr=[r for r in target if r['family']==family and r['method']==m and (conj=='both' or r['conjunction']==conj)]
                grouped={}
                for r in rr:grouped.setdefault((r['pair_index'],r['conjunction']),[]).append(r)
                cc=[dict(pair_index=k[0],conjunction=k[1],family=family,method=m,source_contrast=rs[0]['source_contrast'],median_target_contrast=statistics.median(r['candidate_contrast'] for r in rs),median_absolute_error=statistics.median(abs(r['contrast_error']) for r in rs),median_relative_kl=statistics.median(r['normalized_kl_error'] for r in rs),direction_agreement=sum(r['candidate_contrast']*r['source_contrast']>0 for r in rs)) for k,rs in grouped.items()]
                if conj=='both':cases.extend(cc)
                tables.append(dict(family=family,conjunction=conj,method=m,target_cases=len(rr),source_cases=len(cc),direction_agreement=sum(r['candidate_contrast']*r['source_contrast']>0 for r in rr),zero_effects=sum(r['candidate_contrast']==0 for r in rr),pooled_median_absolute_error=statistics.median(abs(r['contrast_error']) for r in rr),pooled_median_relative_kl=statistics.median(r['normalized_kl_error'] for r in rr),case_balanced_median_absolute_error=statistics.median(r['median_absolute_error'] for r in cc),case_balanced_median_relative_kl=statistics.median(r['median_relative_kl'] for r in cc)))
            grouped={}
            for r in target:
                if r['family']==family and (conj=='both' or r['conjunction']==conj):grouped.setdefault((r['pair_index'],r['conjunction'],r['target_seed']),{})[r['method']]=r
            for m in ['atom','raw']:
                paired.append(dict(family=family,conjunction=conj,baseline=m,rows=len(grouped),fcc_lower_absolute_error=sum(abs(v['fcc']['contrast_error'])<abs(v[m]['contrast_error']) for v in grouped.values()),fcc_lower_relative_kl=sum(v['fcc']['normalized_kl_error']<v[m]['normalized_kl_error'] for v in grouped.values())))
    # Independent probability formulas, without importing the shared endpoint helper.
    arr=np.load(tp/'probabilities.npz');auth=json.loads((sp/'authored_inputs.json').read_text());ids=auth['clause_token_ids'];errors=[];klerrors=[]
    def p(key):
        v=arr[key].astype(float);return v/v.sum()
    def lo(v):
        mass=sum(float(v[k]) for k in ids);return np.log(mass)-np.log1p(-mass)
    def kl(a,b):return float(np.sum(a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(b,1e-300)))))
    for i,c in enumerate(auth['cases']):
        base=p(f'base_{i}');src=p(f'source_{i}')
        for r in [r for r in target if r['pair_index']==c['pair_index'] and r['conjunction']==c['conjunction']]:
            candidate=p(f"candidate_{i}_{r['target_seed']}_{r['method']}")
            errors.append(abs(lo(candidate)-lo(base)-r['candidate_contrast']))
            kr=kl(src,candidate)/kl(src,base);klerrors.append(abs(kr-r['normalized_kl_error']))
    assert max(errors)<2e-6 and max(klerrors)<1e-3
    source_rows=[dict(pair_index=r['pair_index'],family=r['family'],conjunction=r['conjunction'],document_id=r['document_id'],sequence_index=r['sequence_index'],token_index=r['token_index'],coordinate=r['coordinate'],coordinate_difference=r['source_difference_coordinate'],dose=r['dose_scale'],source_logodds_delta=r['effects']['source_swap']['contrast_delta'],baseline_pronoun_mass=r['effects']['baseline']['clause_mass'],source_pronoun_mass=r['effects']['source_swap']['clause_mass'],source_kl=r['effects']['source_swap']['kl_to_baseline'],random_logodds_delta=r['effects']['orthogonal_random_same_norm']['contrast_delta']) for r in source]
    for name,data in [('TARGET_ALL_ROWS.csv',target),('TARGET_CASES.csv',cases),('SOURCE_ALL_ROWS.csv',source_rows)]:
        fields=[k for k in data[0] if k not in ['token_ids']]
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(data)
    result=dict(source=source_rows,target=tables,paired=paired,verification=dict(max_effect_error=max(errors),max_relative_kl_error=max(klerrors),scope='Saved float32 probability numerical replay, not independent scientific replication'),inputs=[dict(path=str(p.relative_to(ROOT)),sha256=H(p)) for p in [sp/'metrics.raw.jsonl',tp/'metrics.raw.jsonl',tp/'probabilities.npz',Path(__file__)]],scope='Primary ordinary and reported separately; all or and constrained cases retained. Eight documents, one source query, four dependent target seeds. Full FCC unequal atom capacity; raw strong control. No source-effect rejection.')
    (out/'CHECKPOINT_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(target=tables,paired=paired,verification=result['verification'])))
    for r in source_rows:
        if r['conjunction']==' and':print(json.dumps(r))


if __name__=='__main__':main()
