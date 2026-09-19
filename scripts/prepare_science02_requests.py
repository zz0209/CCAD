"""Freeze calibration candidates and unobserved request coordinates before execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np

root=Path(__file__).resolve().parents[1]
out=root/'artifacts/science_upgrade_20260919/ROUND02_REQUESTS.json'
if out.exists(): raise FileExistsError(out)
groups=['pronouns','names','associated_words']
cal={}
for mask in range(1,8):
    q=[float((mask>>i)&1) for i in range(3)]
    name='full' if mask==7 else '+'.join(g for i,g in enumerate(groups) if q[i])
    cal[name]=q
cal['center']=[.5,.5,.5]
for i,g in enumerate(groups):
    for v in [0.,1.]:
        q=[.5,.5,.5];q[i]=v;cal[f'face_{g}_{int(v)}']=q
# A seeded stratified design covers the interior; explicit faces and edges
# prevent a successful center-point policy from being judged only nearby.
rng=np.random.default_rng(2026091902)
interior=np.stack([(rng.permutation(12)+rng.random(12))/12 for _ in range(3)],axis=1)
test={f'interior_{i:02d}':q.tolist() for i,q in enumerate(interior)}
for i in range(12):
    q=rng.uniform(.08,.92,3);fixed=i%3;q[fixed]=float((i//3)%2)
    if i>=6: q[(fixed+1)%3]=float(i%2)
    test[f'boundary_{i:02d}']=q.tolist()
assert len({tuple(q) for q in list(cal.values())+list(test.values())})==38
methods=['initial','head_parts','pooled_parts','white_parts','pooled_whole','head_continuous','pooled_continuous']
panel={'written_at_utc':datetime.now(timezone.utc).isoformat(),'groups':groups,
       'calibration':cal,'evaluation':test,'seed':2026091902,'methods':methods,
       'budgets':[1,3,7],'primary_budget':3,
       'policy':'All request coordinates and analysis rules fixed before any ROUND02 responses. Existing one target seed and exposed contexts remain development.',
       'designs':['endpoints','coordinate_coverage','source_response_coverage','random_mean_200'],
       'prediction':'Nearest calibrated request, using coordinate or source-response distance. Secondary RBF kernel interpolation uses a fixed median distance and ridge 1e-4. No held-out target responses select a rule.',
       'primary':'Mean later-task response nRMSE of the selected native adapter across all24 evaluation requests, each head on its profession pair. Report best-fixed native adapter and raw reconstruction separately.',
       'secondary':['selection regret','adapter ranking','response-error prediction','interior versus boundary'],
       'resampling':'Paired documents within profession/gender; queries kept paired and grouped by interior/boundary. One target seed; no seed-population inference.'}
out.write_text(json.dumps(panel,indent=2)+'\n')
base=json.loads((root/'configs/science01_response_continuous_tasks_v2.json').read_text())
base['queries']=list(cal)+list(test)
base['dose_queries']={name:dict(zip(groups,q)) for name,q in (cal|test).items()}
base['run_parent']='SCIENCE_UPGRADE_02';base['request_panel']=str(out)
base['request_panel_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
base['variants']=methods[1:];base['evaluate_baselines']=['raw_reconstruction']
base['evaluate_checkpoints']={}
for m in methods[1:]:
    run='SCIENCE01_response_continuous_dev_v1_20260919' if m.endswith('_continuous') else 'SCIENCE01_response_space_dev_v1_20260919'
    base['evaluate_checkpoints'][m]=f'D:/CCAD_Storage/training_curves/{run}/{m}'
base['budget_seconds']=1200
base['budget']='At most1200driver seconds per panel; existing frozen candidates,38 fixed requests,64 or256 documents. Based on SCIENCE01 measured inference. No fitting/download.'
base['evidence_level']='frozen_request_development'
base['scope']='New fixed request coordinates on exposed contexts and existing one-seed adapted dictionaries. Source-side query design and budget-matched adapter selection; not independent multi-seed confirmation.'
for suffix in ['cal','tasks']:
    c=base.copy();c['run_id']=f'SCIENCE02_coverage_{suffix}_v1_20260919'
    c['purpose']='Measure request coverage and future-query reliability of explanation correspondence'
    if suffix=='cal':
        c.pop('evaluation_panel',None);c.pop('evaluation_per_cell',None)
        c['development_per_group']=16
        c['program_evaluation_exclude_per_group']=32
    else:c['evaluation_per_cell']=32
    path=root/'configs'/f'science02_coverage_{suffix}_v1.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(c,indent=2)+'\n')
print(json.dumps({'panel':str(out),'calibration':len(cal),'evaluation':len(test),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
