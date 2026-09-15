"""Development-only, matched-calibration forecasts from saved binding diagnostics."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913'
def main():
    p=ROOT/'runs/REFORM_R43_binding_judgment_diagnostics_v1_20260915/metrics.raw.jsonl'
    groups=defaultdict(list)
    for line in p.read_text().splitlines():
        r=json.loads(line)
        groups[r['method'],r['component'],r['task'],r['seed'],r['operation']].append(r)
    contexts=sorted({k[1] for k in groups});cal=set(contexts[:16]);output=[]
    specs={'source':[0,1],'code':[2,3],'source_code':[0,1,2,3],
           'source_code_residual':[0,1,2,3,4,5,6]}
    for method in sorted({k[0] for k in groups}):
        x=[];y=[];train=[]
        for key,rr in sorted(groups.items()):
            if key[0]!=method:continue
            assert len(rr)==2 and {r['query'] for r in rr}=={0,1}
            src=[r['source_log_probability'] for r in rr];code=[r['code_log_probability'] for r in rr]
            x.append([min(src),sum(src),min(code),sum(code),rr[0]['relative_prediction_residual'],rr[0]['cosine'],rr[0]['norm_ratio'],float(key[-1]=='first'),float(key[-1]=='second')])
            y.append(all(r['target_correct'] for r in rr));train.append(key[1] in cal)
        x=np.array(x);y=np.array(y);train=np.array(train)
        for name,cols in specs.items():
            features=x[:,cols+[7,8]]
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))
            model.fit(features[train],y[train]);score=model.predict_proba(features[~train])[:,1]
            order=np.argsort(-score,kind='stable');n=len(score)//2
            output.append(dict(method=method,score=name,auc=roc_auc_score(y[~train],score),
                top_half_success=float(y[~train][order[:n]].mean()),all_success=float(y[~train].mean()),
                calibration_complete_requests=int(train.sum()),test_complete_requests=int((~train).sum())))
    out=dict(rows=output,calibration_contexts=sorted(cal),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        notes='Exposed development: first16contexts calibrate, remaining112 form a simulated holdout. All forms, operations and dependent SAE directions stay with their context. Predictors use min/sum source or code log-probability, then optional residual/cosine/norm ratio; operation indicators are shared. No independent confirmation or efficiency claim.')
    (ART/'r43_binding_judgment_reproducible.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([r for r in output if r['method']=='synthesized_code_bank']))
if __name__=='__main__':main()
