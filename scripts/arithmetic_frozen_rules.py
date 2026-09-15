"""Evaluate immutable source and target memberships on a new carry-rule panel."""
import hashlib
import json
import numpy as np
import torch
from run_causalgym_multisite import ROOT, write


def evaluate_frozen_rules(w, cfg, rows, hidden, codes, saes, generate, budget, base, model_context=None):
    spec=cfg['frozen_rule_evaluation']
    freeze=json.loads((ROOT/spec['freeze']).read_text())
    for item in freeze['inputs']:
        p=w.checked(ROOT/item['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
    panel=json.loads((ROOT/spec['panel']).read_text())
    assert panel['rows']==rows and not any(r['split']=='fit' for r in rows)
    pairs=panel['pairs'];write(w.run/'RULE_PANEL.json',panel)
    for seed,ae in saes.items():
        definition=spec['memberships'][str(seed)]
        parent=ROOT/definition['run']
        pcfg=json.loads((parent/'config.resolved.json').read_text())
        assert all(pcfg[k]==cfg[k] for k in ['training_run','checkpoint_step','model_revision'])
        with np.load(parent/f'rule_members_seed{seed}.npz') as data:
            stored={k:data[k] for k in data.files}
        if cfg.get('source_function_refit'):
            assert seed == 1 and len(saes) == 1 and model_context is not None
            from arithmetic_carry_source_fit import fit_source
            stored=fit_source(w,cfg,*model_context,ae,stored,budget)
        v=torch.tensor(stored['raw_direction'],device=w.device)
        assert abs(float(v.norm())-1)<1e-5
        decoder=ae.decoder.weight.T;z=codes[seed]
        for name in definition['methods']:
            indices=None if name in ['noop','whole_state','raw_carry_direction'] else torch.tensor(stored[name],device=w.device)
            weights=torch.tensor(stored[name+'_weights'],device=w.device) if name+'_weights' in stored else None
            if indices is not None:
                assert len(indices)<=64 and len(indices.unique())==len(indices)
            if weights is not None:assert bool(((weights>=0)&(weights<=1)).all())
            records=[]
            for off in range(0,len(pairs),cfg['batch_size']):
                pp=pairs[off:off+cfg['batch_size']];ii=[p['recipient'] for p in pp];di=[p['donor'] for p in pp]
                tens_site=torch.tensor([cfg['tens_sites'][p['template']] for p in pp],device=w.device)
                def patch(current,step):
                    mask=(torch.arange(step+1,device=w.device)[None,:]==tens_site[:,None])
                    if name=='whole_state':change=hidden[di,:step+1]-current
                    elif name=='raw_carry_direction':
                        delta=hidden[di,:step+1]-current;change=(delta@v)[...,None]*v
                    else:
                        current_z=ae.encode(current.flatten(0,1)).reshape(len(ii),step+1,-1)
                        coeff=z[di,:step+1][...,indices]-current_z[...,indices]
                        if weights is not None:coeff=coeff*weights
                        change=coeff@decoder[indices]
                    return change*mask[...,None]
                ans,text,replay,editnorm=generate(ii,patch=None if name=='noop' else patch)
                if name=='noop':assert ans==[base[i] for i in ii]
                for j,p in enumerate(pp):
                    r,d=rows[p['recipient']],rows[p['donor']];numeric=ans[j] is not None and 10<=ans[j]<100
                    record=dict(kind='rule_intervention',task=f"template{p['template']}",row_id=p['recipient'],
                        component=p['component'],mode='tens_role',method=name,seed=seed,target_seed=None,
                        operation=p['condition'],stratum=p['stratum'],split=cfg.get('evaluation_split','confirmation'),donor_row=p['donor'],
                        recipient_question=[r[k] for k in ['a','b','c'] if k in r],donor_question=[d[k] for k in ['a','b','c'] if k in d],
                        recipient_carry=r['carry'],donor_carry=d['carry'],original_answer=r['total'],
                        donor_original_answer=d['total'],expected=p['expected'],answer=ans[j],generated_text=text[j],
                        correct=ans[j]==p['expected'],retained_original=ans[j]==r['total'],
                        unit_preserved=bool(numeric and ans[j]%10==r['unit']),
                        signed_tens_change=((ans[j]//10-r['tens'])*(d['carry']-r['carry']) if numeric else None),
                        edit_norm=float(editnorm[j]))
                    w.record(**record);records.append(record)
                budget()
            profile={condition:dict(success=float(np.mean([r['correct'] for r in records if r['operation']==condition])),
                        unit_preserved=float(np.mean([r['unit_preserved'] for r in records if r['operation']==condition])))
                     for condition in ['same_answer_opposite_carry','same_carry_different_answer']}
            w.progress('FROZEN_RULE_TESTED',seed=seed,method=name,profile=profile)
    write(w.run/'RULE_RESULTS.json',dict(scope=cfg['scope'],baseline_accuracy=float(np.mean([a==r['total'] for a,r in zip(base,rows)])),
          freeze=spec['freeze'],fit_updates=cfg.get('counterfactual_fit',{}).get('steps',0),
          source_definition=('Direct carry-answer CE on the original two-operand source fit split.' if cfg.get('source_function_refit') else 'Frozen R49 source1 binary/weighted/association members, target memberships and raw direction.'),
          primary='Complete carry-rule success pooled equally over three carry-pair strata; same-carry preservation separately. All predetermined questions retained regardless of baseline correctness.'))
    w.checks['frozen_artifact_hashes']=True
    if not cfg.get('source_function_refit'):w.checks['no_new_fitting']=True
    w.checks['cached_baseline_replay']=True
