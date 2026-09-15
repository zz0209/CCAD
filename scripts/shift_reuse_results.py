"""Analyze human-explanation reuse on the retained biography development panel.

Uncertainty is a paired, group-stratified document bootstrap, conditional on
the fixed source head and one target training trajectory. It is not seed-level
or post-selection confirmatory inference. Original run records remain intact.
"""
from pathlib import Path
import json, hashlib, datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
RUNS={
    'field_1m':'REFORM_R58_shift_transfer_dev_v1_20260915',
    'token_1m':'REFORM_R58_shift_functional_transfer_dev_v1_20260915',
    'context_1m':'REFORM_R58_shift_context_transfer_dev_v1_20260915',
    **{f'curve_{step}':f'REFORM_R58_shift_context_curve_s{step}_v1_20260915' for step in (1024,4096,8192)},
}

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def identity(p): return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    source_run=ROOT/'runs/REFORM_R58_shift_source_development_v2_20260915'
    panel=[r for r in read(source_run/'panel.json')['rows'] if r['split']=='dev']
    ids=[r['document_sha256'] for r in panel]
    assert len(ids)==len(set(ids))==695
    label=np.array([r['label'] for r in panel]);gender=np.array([r['gender'] for r in panel])
    group=[np.flatnonzero((label==y)&(gender==g)) for y in (0,1) for g in (0,1)]
    rng=np.random.default_rng(5815)
    boot=[rng.choice(ix,size=(4000,len(ix)),replace=True) for ix in group]
    order={k:i for i,k in enumerate(ids)}
    values={};metrics={};sources=[];checks={}
    for key,name in {'source':source_run.name,**RUNS}.items():
        run=ROOT/'runs'/name;assert read(run/'status.json')['status']=='PASS'
        assert read(run/'contract_validation.json')['ok']
        rows=[json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
        cells={}
        for r in rows:
            if r['kind']!='classification':continue
            op=r.get('operation','full');method=r['method']
            if key=='source':op,method=method,('none' if method=='none' else 'source')
            cell=cells.setdefault(method+'/'+op,np.full(695,np.nan))
            cell[order[r['component']]]=r['logit']
        values[key]=cells;metrics[key]={}
        for cell,logits in cells.items():
            assert np.isfinite(logits).all(),(key,cell)
            correct=(logits>0)==label
            samples=np.stack([correct[b].mean(1) for b in boot],axis=1)
            groupmean=np.array([correct[ix].mean() for ix in group])
            overall=samples@np.array([len(g) for g in group])/695
            metrics[key][cell]=dict(profession=float(correct.mean()),worst_group=float(groupmean.min()),gender=float(((logits>0)==gender).mean()),groups=groupmean.tolist(),group_ci=np.quantile(samples,[.025,.975],axis=0).T.tolist(),profession_ci=np.quantile(overall,[.025,.975]).tolist(),worst_group_ci=np.quantile(samples.min(1),[.025,.975]).tolist())
        sources.append(identity(run/'metrics.raw.jsonl'))
    reference=values['field_1m']
    for key,cells in values.items():
        if key=='source':continue
        checks[key+'_source_predictions_equal']=all(np.array_equal(cells['source/'+q]>0,reference['source/'+q]>0) for q in ('full','pronouns','names','associated_words'))
        checks[key+'_source_max_logit_error']=max(float(np.max(np.abs(cells['source/'+q]-reference['source/'+q]))) for q in ('full','pronouns','names','associated_words'))
    checks['context_runtime_refactor_geometry_max_logit_error']=max(float(np.max(np.abs(values['context_1m']['geometry/'+q]-reference['geometry/'+q]))) for q in ('full','pronouns','names','associated_words'))
    contrasts={}
    def contrast(a,b):
        ca=(values[a[0]][a[1]]>0)==label;cb=(values[b[0]][b[1]]>0)==label
        sa=np.stack([ca[ix].mean(1) for ix in boot],axis=1);sb=np.stack([cb[ix].mean(1) for ix in boot],axis=1)
        score=(sa-sb)@np.array([len(g) for g in group])/695
        worst=sa.min(1)-sb.min(1)
        return dict(profession=float(ca.mean()-cb.mean()),profession_ci=np.quantile(score,[.025,.975]).tolist(),worst_group=metrics[a[0]][a[1]]['worst_group']-metrics[b[0]][b[1]]['worst_group'],worst_group_ci=np.quantile(worst,[.025,.975]).tolist())
    for key in RUNS:
        for q in ('full','pronouns','names','associated_words'):
            for control in ('geometry','geometry_gain','raw'):
                contrasts[key+'/'+q+'/native-'+control]=contrast((key,'native/'+q),(key,control+'/'+q))
        contrasts[key+'/full/native-unedited']=contrast((key,'native/full'),('source','none/none'))
    for control in ('field_1m','token_1m','curve_1024','curve_4096'):
        current='context_1m' if control.endswith('_1m') else 'curve_8192'
        contrasts[current+'-'+control]=contrast((current,'native/full'),(control,'native/full'))
    quality=[]
    training=ROOT/'runs/REFORM_R58_shift_dictionaries_curve_v1_20260915'
    for line in (training/'metrics.raw.jsonl').read_text().splitlines():quality.append(json.loads(line))
    write(ART/'r58_shift_reuse_analysis.json',dict(written_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=metrics,contrasts=contrasts,checks=checks,quality=quality,sources=sources,
        scope='Development, fixed 695 unique biographies, one source head and one target seed. 4000 paired document bootstrap draws stratified by profession/gender, seed5815. Repeated development comparisons are descriptive, not confirmatory. Public source ReLU and independently trained target TopK differ in training conditions. All 11 sites replaced. Official BiB test unexposed.'))
    for key,rows in metrics.items():
        print(key,{k:[round(100*v['profession'],2),round(100*v['worst_group'],2)] for k,v in rows.items() if k.endswith('/full')})
    print('source_identity',checks)

if __name__=='__main__':main()
