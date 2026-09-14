"""Compare role-conditioned source functions and their cross-seed reuse."""
from pathlib import Path
from datetime import datetime, timezone
import collections, hashlib, json, re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'


def main():
    groups, identities, inputs = {}, None, []
    run_specs = [
        ('REFORM_R34_qwen_valid_equivalent_views_v2_20260914', {'gradient_views':'static_gradient'}),
        ('REFORM_R37_qwen_source_roles_v1_20260914', {'gradient_roles':'role_gradient'}),
        ('REFORM_R37_qwen_roles_remaining_v1_20260914', {'gradient_roles':'role_gradient','gradient_views':'static_gradient'}),
        ('REFORM_R37_qwen_role_transfer_v1_20260914', {'transfer_clean':'role_field','transfer_assignment':'role_assignment'}),
    ]
    for run, methods in run_specs:
        p = ROOT / 'runs' / run
        if not (p/'metrics.summary.json').exists():
            continue
        assert json.loads((p/'status.json').read_text())['status'] == 'PASS'
        raw = p/'metrics.raw.jsonl'; digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        assert digest == json.loads((p/'metrics.summary.json').read_text())['metrics_raw_sha256']
        inputs.append(dict(run=str(p.relative_to(ROOT)),raw_sha256=digest))
        panel = json.loads((p/'panel.json').read_text())
        ids = [[panel['rows'][i][k] for i in [q['recipient'],q['donor']] for k in ['a','b']] for q in panel['pairs'][:64]]
        if identities is None: identities = ids
        assert ids == identities
        for line in raw.read_text().splitlines():
            r=json.loads(line)
            if r['kind']!='source_patch' or not r['seed']:continue
            method = re.sub(r'^transfer_s\d+_', 'transfer_', r['method'])
            if method not in methods:continue
            name = methods[method]
            if name not in groups:groups[name]=np.full((64,2,2,5,3),np.nan)
            q=panel['pairs'][r['row_id']]
            cluster=[panel['rows'][i][k] for i in [q['recipient'],q['donor']] for k in ['a','b']]
            ix=(identities.index(cluster),['unit','tens'].index(r['operation']),q['template'],r['seed']-1)
            assert np.isnan(groups[name][ix]).all()
            groups[name][ix]=[r[k] for k in ['exact_hybrid','target_digit_success','preserve_digit_success']]
    assert all(np.isfinite(a).all() for a in groups.values())
    cells=[]
    for name,a in groups.items():
        for oi,op in enumerate(['unit','tens']):
            cells.append(dict(method=name,operation=op,n=640,
                              **{m:float(a[:,oi,...,k].mean()) for k,m in enumerate(['H','T','P'])},
                              per_seed_H=a[:,oi,:,:,0].mean((0,1)).tolist()))
    draws=np.random.default_rng(9370914).integers(64,size=(10000,64)); contrasts=[]
    for name,other in [('role_gradient','static_gradient'),('role_field','role_assignment'),('role_field','role_gradient')]:
        if name not in groups or other not in groups:continue
        for op,sel in [('both',slice(None)),('unit',0),('tens',1)]:
            diff=groups[name][:,sel]-groups[other][:,sel]
            delta=diff.mean(tuple(range(1,diff.ndim-1)))
            ci=np.quantile(delta[draws].mean(1),[.025,.975],axis=0)*100
            contrasts.append(dict(reference=name,comparator=other,operation=op,
                                  metrics={m:dict(difference_points=float(delta[:,k].mean()*100),interval_points=ci[:,k].tolist()) for k,m in enumerate(['H','T','P'])}))
    schedules=[];profiles=[]
    for seed in range(1,6):
        role_run='REFORM_R37_qwen_'+('source_roles' if seed<3 else 'roles_remaining')+'_v1_20260914'
        static_run='REFORM_R34_qwen_valid_equivalent_views_v2_20260914' if seed<3 else role_run
        with np.load(ROOT/'runs'/role_run/f'counterfactual_seed{seed}_k64_gradient_roles.npz') as r,np.load(ROOT/'runs'/static_run/f'counterfactual_seed{seed}_k64_gradient_views.npz') as s:
            assert np.array_equal(r['fit_schedule'],s['fit_schedule'])
            g=r['gates'];assert g.shape==(6,8192,2) and np.all((g>=0)&(g<=1))
            assert np.all((g>0).any(0).sum(0)<=64)
            schedules.append(hashlib.sha256(r['fit_schedule'].tobytes()).hexdigest())
            profiles.append(dict(seed=seed,union=(g>0).any(0).sum(0).tolist(),role_mass=g.sum(1).tolist(),role_active=(g>0).sum(1).tolist()))
    assert len(set(schedules))==1
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),scope='Adaptive original-panel development. Question-cluster intervals condition on the fixed five-SAEs.',
             cells=cells,contrasts=contrasts,profiles=profiles,training_schedule_sha256=schedules[0],inputs=inputs)
    (ART/'r37_development.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ART/'r37_development_outcomes.npz',outcomes=np.stack(list(groups.values())),methods=np.array(list(groups)),operand_pairs=np.array(identities))
    print(json.dumps(dict(means={n:float(a[...,0].mean()) for n,a in groups.items()},contrasts=contrasts)))

if __name__=='__main__':main()
