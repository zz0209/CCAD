"""Descriptive complete-scope report for the fixed and/or explanation probes."""
import csv,json,statistics,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def raw(name):
    p=ROOT/'runs'/name/'metrics.raw.jsonl'
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()],dict(path=str(p),sha256=digest(p))

def main():
    initial,hi=raw('F4_and_source_v1_20260906');repl,hr=raw('F4_and_source_lexical_replication_v1_20260906');target,ht=raw('F4_and_targets_v1_20260906')
    out=ROOT/'artifacts/and_source_development_20260906';rows=[];case_rows=[]
    for family in ['ordinary','constrained']:
        for method in ['fcc','atom','raw']:
            rr=[r for r in target if r['family']==family and r['method']==method]
            for topic in sorted({r['topic'] for r in rr}):
                for conj in [' and',' or']:
                    cc=[r for r in rr if r['topic']==topic and r['conjunction']==conj]
                    assert len(cc)==4
                    case_rows.append(dict(family=family,topic=topic,conjunction=conj,method=method,source_contrast=cc[0]['source_contrast'],
                        median_target_contrast=statistics.median(r['candidate_contrast'] for r in cc),median_absolute_error=statistics.median(abs(r['contrast_error']) for r in cc),
                        median_normalized_kl=statistics.median(r['normalized_kl_error'] for r in cc),direction_agreement=sum(r['source_contrast']*r['candidate_contrast']>0 for r in cc),targets=4))
            cr=[r for r in case_rows if r['family']==family and r['method']==method]
            rows.append(dict(family=family,method=method,target_cases=len(rr),direction_agreement=sum(r['source_contrast']*r['candidate_contrast']>0 for r in rr),
                pooled_median_absolute_error=statistics.median(abs(r['contrast_error']) for r in rr),pooled_median_normalized_kl=statistics.median(r['normalized_kl_error'] for r in rr),
                case_balanced_median_absolute_error=statistics.median(r['median_absolute_error'] for r in cr),case_balanced_median_normalized_kl=statistics.median(r['median_normalized_kl'] for r in cr)))
    pairs={}
    for r in target:pairs.setdefault((r['pair_index'],r['conjunction'],r['target_seed']),{})[r['method']]=r
    paired={fam:{m:dict(contrast_error_fcc_lower=sum(abs(v['fcc']['contrast_error'])<abs(v[m]['contrast_error']) for v in pairs.values() if v['fcc']['family']==fam),
                       kl_error_fcc_lower=sum(v['fcc']['normalized_kl_error']<v[m]['normalized_kl_error'] for v in pairs.values() if v['fcc']['family']==fam),target_cases=32)
                 for m in ['atom','raw']} for fam in ['ordinary','constrained']}
    source={}
    for label,rr in [('initial_failed_direction',initial),('revised_new_lexical',repl)]:
        source[label]={}
        for family in ['ordinary','constrained']:
            cc=[r for r in rr if r['family']==family];aa=[r for r in cc if r['conjunction']==' and'];oo=[r for r in cc if r['conjunction']==' or']
            source[label][family]=dict(and_negative_coordinate_difference=sum(r['source_difference_coordinate']<0 for r in aa),and_effects=[r['effects']['source_swap']['contrast_delta'] for r in aa],or_effects=[r['effects']['source_swap']['contrast_delta'] for r in oo],
                and_clause_mass_deltas=[r['effects']['source_swap']['clause_mass']-r['effects']['baseline']['clause_mass'] for r in aa],
                and_item_mass_deltas=[r['effects']['source_swap']['item_mass']-r['effects']['baseline']['item_mass'] for r in aa])
    for filename,data in [('TARGET_CASES.csv',case_rows),('TARGET_ALL_ROWS.csv',target)]:
        fields=[k for k in data[0] if k not in ['token_ids']]
        with (out/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
    summary=dict(source=source,target=rows,paired_comparisons=paired,inputs=[hi,hr,ht],script_sha256=digest(Path(__file__)),
        scope='Initial direction failure and revised lexical replication separate. All 16 target inputs retained. Dependent query/topic/seed cases; pooled and case-balanced medians shown, no significance claim.')
    (out/'CHECKPOINT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(target=rows,paired_comparisons=paired)))

if __name__=='__main__':main()
