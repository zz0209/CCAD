"""Paired native-correspondence evidence with the original hypotheses retained."""
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'artifacts/interpretability_closeout_20260906';data={};arrays={};inputs=[]
    for label in ['source','counterpart']:
        p=ROOT/'runs'/f'NATIVE_we_natural_{label}_v1_20260906'
        data[label]=[json.loads(s) for s in (p/'metrics.raw.jsonl').read_text().splitlines()];arrays[label]=np.load(p/'probabilities.npz')
        inputs.extend(dict(path=str(f.relative_to(ROOT)),sha256=H(f)) for f in [p/'metrics.raw.jsonl',p/'probabilities.npz',p/'contrast_tokens.json'])
    first=json.loads((ROOT/'runs/NATIVE_we_natural_source_v1_20260906/contrast_tokens.json').read_text())['first'];third=json.loads((ROOT/'runs/NATIVE_we_natural_source_v1_20260906/contrast_tokens.json').read_text())['third']
    def normalize(p):p=p.astype(float);return p/p.sum()
    def kl(a,b):return float(np.sum(a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(b,1e-300)))))
    def lc(p):return np.log(p[first].sum()/p[third].sum())
    rows=[];maxbase=maxeffect=0.
    for i,(s,t) in enumerate(zip(data['source'],data['counterpart'])):
        for k in ['token_ids','document_id','subject','template','position']:assert s[k]==t[k],k
        base=normalize(arrays['source'][f'case_{i}_baseline']);otherbase=normalize(arrays['counterpart'][f'case_{i}_baseline']);maxbase=max(maxbase,float(np.max(abs(base-otherbase))))
        for op in ['native_remove','native_add']:
            ps=normalize(arrays['source'][f'case_{i}_{op}']);pt=normalize(arrays['counterpart'][f'case_{i}_{op}']);se=s['effects'][op]['contrast_delta'];te=t['effects'][op]['contrast_delta']
            maxeffect=max(maxeffect,abs(lc(ps)-lc(base)-se),abs(lc(pt)-lc(otherbase)-te));den=kl(ps,base);err=kl(ps,pt)
            rows.append(dict(template=s['template'],document_id=s['document_id'],subject=s['subject'],verb=s['verb'],operation=op,source_activation=s['activation'],counterpart_activation=t['activation'],source_contrast=se,counterpart_contrast=te,absolute_contrast_error=abs(te-se),source_first_mass=s['effects'][op]['first_mass'],counterpart_first_mass=t['effects'][op]['first_mass'],source_to_baseline_kl=den,source_to_counterpart_kl=err,relative_kl=err/den if den>1e-12 else None,source_dose=s['dose_scale'],counterpart_dose=t['dose_scale'],direction_agreement=se*te>0,both_zero=se==0 and te==0))
    assert maxbase==0 and maxeffect<2e-6
    summary={}
    for label,rr in data.items():
        we=[r for r in rr if r['subject']=='we'];comparisons=[w['activation']>c['activation'] for w in we for c in rr if c['template']==w['template'] and c['subject']!='we']
        summary[label]=dict(we_activation_higher=sum(comparisons),matched_comparisons=len(comparisons),we_activation_range=[min(r['activation'] for r in we),max(r['activation'] for r in we)],original_remove_hypothesis_success=sum(r['effects']['native_remove']['contrast_delta']<0 for r in we),original_add_hypothesis_success=sum(r['effects']['native_add']['contrast_delta']>0 for r in we),we_cases=len(we),control_nonzero_activations=sum(r['activation']>0 for r in rr if r['subject']!='we'),control_cases=sum(r['subject']!='we' for r in rr))
    correspondence={}
    for subject in ['we','controls','all']:
        rr=[r for r in rows if (r['subject']=='we' if subject=='we' else r['subject']!='we' if subject=='controls' else True)]
        for op in ['native_remove','native_add']:
            cc=[r for r in rr if r['operation']==op];nz=[r for r in cc if r['relative_kl'] is not None]
            correspondence[subject+'/'+op]=dict(cases=len(cc),nonzero_source_cases=len(nz),same_direction=sum(r['direction_agreement'] for r in cc),both_zero=sum(r['both_zero'] for r in cc),median_absolute_contrast_error=statistics.median(r['absolute_contrast_error'] for r in cc),median_relative_kl=statistics.median(r['relative_kl'] for r in nz),relative_contrast_squared_error=sum((r['source_contrast']-r['counterpart_contrast'])**2 for r in cc)/sum(r['source_contrast']**2 for r in cc))
    with (out/'NATIVE_ALL_ROWS.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    result=dict(original_hypotheses=summary,correspondence=correspondence,rows=rows,verification=dict(max_cross_seed_baseline_probability_error=maxbase,max_saved_probability_contrast_error=maxeffect),inputs=inputs+[dict(path=str(Path(__file__).relative_to(ROOT)),sha256=H(Path(__file__)))],scope='Eight new source documents, fixed native pair and all32subject inputs; original constant-sign hypotheses not relabeled. Cross-seed native effects use own amplitudes under common cap, not equal-dose proof. Controls/nulls retained, not independent64replications or FCC claim.')
    (out/'NATIVE_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(hypotheses=summary,correspondence=correspondence,verification=result['verification'])))


if __name__=='__main__':main()
