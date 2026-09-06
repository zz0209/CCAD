"""Source-only conditional contribution fitting and frozen cross-seed transfer.

The kernel is ordinary weighted least squares for actual decoder contributions.
The scientific question is conditional function and transfer, not solver novelty.
"""
import json
from pathlib import Path
import numpy as np
from run_r011s1_raw_hook_asset import write_json as write


def run_composition(cfg,run,rows,pairs,raw,base,candidate_ids,forward,progress):
    import torch
    spec=cfg['composition']
    tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float64,device='cuda:0')
    host=lambda x:x.detach().cpu().numpy()
    slots=np.array([r['slot'] for r in rows])
    h=raw[np.arange(len(rows)),slots]
    ri=np.array([p['recipient'] for p in pairs])
    di=np.array([p['donor'] for p in pairs])
    phase=np.array([rows[i]['phase'] for i in ri])
    role=np.array([rows[i]['role'] for i in ri])
    fit=np.flatnonzero(phase=='fit')
    test=np.flatnonzero(phase=='test')
    positive=role=='calendar'
    dh=h[di]-h[ri]
    z,d={},{}
    for seed in range(1,6):
        ar=np.load(run/f'codes_seed{seed}.npz')
        z[seed]=ar['codes'][np.arange(len(rows)),slots].astype(float)
        d[seed]=ar['decoder'].astype(float)
    dz={s:z[s][di]-z[s][ri] for s in z}
    weights=np.where(positive[fit],1.,spec['negative_weight'])
    target=np.where(positive[fit,None],dh[fit],0.)
    tx={s:tensor(x[fit]) for s,x in dz.items()}
    td={s:tensor(v) for s,v in d.items()}
    tw=tensor(weights)
    tt=tensor(target)
    ridge=spec['ridge_fraction']

    def native(x,decoder,teacher,weight,cap,ids=None):
        n=len(x)
        diag=torch.mean(x*x*weight[:,None],dim=0)*torch.sum(decoder*decoder,dim=1)
        rhs=torch.sum((x.T@(teacher*weight[:,None])/n)*decoder,dim=1)
        active=torch.where(diag>1e-12)[0]
        score=rhs*rhs/torch.clamp(diag,min=1e-30)
        if ids is None:
            ids=active[torch.argsort(score[active],descending=True,stable=True)[:cap]]
        else:
            ids=torch.as_tensor(ids,device='cuda:0',dtype=torch.int64)
        if len(ids)!=cap:
            raise ValueError('Insufficient active native columns for configured support')
        a=x[:,ids]
        dec=decoder[ids]
        gram=(a.T@(a*weight[:,None])/n)*(dec@dec.T)
        scale=torch.sqrt(diag[ids])
        normalized=gram/scale[:,None]/scale[None,:]
        normal=normalized+ridge*torch.eye(cap,dtype=torch.float64,device='cuda:0')
        rhsn=rhs[ids]/scale
        beta=torch.linalg.solve(normal,rhsn)
        coef=beta/scale
        residual=float(torch.linalg.norm(normal@beta-rhsn)/torch.clamp(torch.linalg.norm(rhsn),min=1e-30))
        if residual>1e-8:
            raise ValueError('Native normalized ridge solve residual too large')
        return host(ids),host(coef),dict(support=cap,negative_coefficients=int((coef<0).sum()),
            max_abs_coefficient=float(torch.max(torch.abs(coef))),normal_equation_relative_residual=residual,
            selection='best one-atom weighted reduction followed by joint normalized ridge' if ids is None else 'supplied or ranked support; joint normalized ridge',
            active_count=len(active),weighted_training_mse=float(torch.mean(torch.sum(((a*coef)@dec-teacher)**2,dim=1)*weight)))

    def raw_linear(x,teacher,weight):
        # Thin SVD implements all-input normalized ridge, not a tuned PCA rank.
        xx=tensor(x)
        scale=torch.sqrt(torch.mean(xx*xx*weight[:,None],dim=0))
        active=torch.where(scale>1e-12)[0]
        xn=xx[:,active]/scale[active]
        root=torch.sqrt(weight[:,None])
        u,s,v=torch.linalg.svd(xn*root,full_matrices=False)
        beta=(v.T*(s/(s*s+len(xx)*ridge)))@(u.T@(teacher*root))
        result=torch.zeros((xx.shape[1],teacher.shape[1]),dtype=torch.float64,device='cuda:0')
        result[active]=beta/scale[active,None]
        return host(result),dict(rank_at_relative_1e_8=int((s>s[0]*1e-8).sum()),
             singular_values=host(s).tolist(),ridge=ridge,kind='full raw normalized linear ridge via thin SVD')

    def native_predictions(s,ids,coef):
        return (dz[s][:,ids]*coef)@d[s][ids]

    sources={}
    fitted={}
    rng=np.random.default_rng(spec['random_seed'])
    # This entire loop sees no target states or behavior in each source fit.
    # All source groups are frozen to disk before any cross-seed fit starts.
    for s in range(1,6):
        ids,a,info=native(tx[s],td[s],tt,tw,spec['source_support'])
        atom,aa,ai=native(tx[s],td[s],tt,tw,1)
        active=torch.where(torch.mean(tx[s]*tx[s],dim=0)>1e-12)[0]
        random_ids=rng.choice(host(active),spec['source_support'],replace=False)
        rand,ra,rinfo=native(tx[s],td[s],tt,tw,spec['source_support'],random_ids)
        pos=tensor(positive[fit].astype(float))
        positive_ids,pa,pinfo=native(tx[s],td[s],tensor(dh[fit]),pos,spec['source_support'],ids)
        rb,rbinfo=raw_linear(dh[fit],tt,tw)
        conditional=native_predictions(s,ids,a)
        positive_only=native_predictions(s,positive_ids,pa)
        pfit=fit[positive[fit]]
        amplitude=float(np.sqrt(np.sum(conditional[pfit]**2)/max(np.sum(positive_only[pfit]**2),1e-30)))
        predictions=dict(conditional64=native_predictions(s,ids,a),
            source_full=dz[s]@d[s],conditional_atom=native_predictions(s,atom,aa),
            random64_refit=native_predictions(s,rand,ra),
            positive_only_same64=positive_only,positive_only_energy_matched=positive_only*amplitude,
            raw_conditional=dh@rb)
        sources[s]=predictions
        fitted[s]=dict(source_seed=s,source_ids=ids.tolist(),source_coefficients=a.tolist(),source_info=info,
            atom_ids=atom.tolist(),atom_coefficients=aa.tolist(),atom_info=ai,
            random_ids=rand.tolist(),random_coefficients=ra.tolist(),random_info=rinfo,
            positive_only_coefficients=pa.tolist(),positive_only_info=pinfo,raw_info=rbinfo,
            positive_only_energy_matched_scale=amplitude,
            scale_rule='match source conditional64 positive-fit vector energy; frozen before all test outputs',
            selection_used_source_codes_and_raw_fit_only=True,test_rows_or_model_outputs_used_for_fit=False)
        np.savez_compressed(run/f'composition_source_s{s}.npz',ids=ids,coef=a,raw_beta=rb,
                            **{name:p for name,p in predictions.items()})
    write(run/'composition_sources.json',dict(sources=list(fitted.values()),specification=spec,
        source_freeze='all five source groups fixed before cross-seed fitting',
        fit_pairs=fit.tolist(),test_pairs=test.tolist(),fit_prompt_ids=sorted(map(int,set(ri[fit])|set(di[fit])))))
    progress('conditional_sources_frozen',source_groups=5,fit_pairs=len(fit),test_pairs=len(test))

    maps={}
    maps_meta=[]
    ones=torch.ones(len(fit),dtype=torch.float64,device='cuda:0')
    for s in range(1,6):
        teacher=tensor(sources[s]['conditional64'][fit])
        rb,rbinfo=raw_linear(dh[fit],teacher,ones)
        for t in range(1,6):
            if t==s:
                continue
            ids,g,info=native(tx[t],td[t],teacher,ones,spec['target_support'])
            ai,ag,ainfo=native(tx[t],td[t],teacher,ones,1)
            maps[s,t]=dict(native64=native_predictions(t,ids,g),
                best_native_atom=native_predictions(t,ai,ag),raw_linear=dh@rb,
                target_full=dz[t]@d[t])
            maps_meta.append(dict(source=s,target=t,ids=ids.tolist(),coefficients=g.tolist(),info=info,
                atom_ids=ai.tolist(),atom_coefficients=ag.tolist(),atom_info=ainfo,raw_info=rbinfo,
                fixed_before_test_interventions=True))
            np.savez_compressed(run/f'composition_map_s{s}_t{t}.npz',ids=ids,coef=g,atom_ids=ai,
                atom_coef=ag,raw_beta=rb,**{name:p for name,p in maps[s,t].items()})
    write(run/'composition_maps.json',dict(maps=maps_meta,role='target masks fit to actual frozen source contribution; no target outcome selection'))
    progress('conditional_target_maps_frozen',directions=len(maps))
    del tx,td,tt,tw

    # Only after freezing source definitions and maps, evaluate new template
    # interventions. Each seed shares the same authored cases and is dependent.
    metrics=[]
    rawlp=np.load(run/'raw_layer5_slot_logprobs.npy',mmap_mode='r')
    checks={}
    batches=[test[i:i+cfg['batch_size']] for i in range(0,len(test),cfg['batch_size'])]
    width=raw.shape[1]
    identity={p['pair_id']:p for p in pairs}

    def metric(pi,lp,reference,prediction,source,target_seed,method,operation):
        p=identity[int(pi)]
        r=rows[p['recipient']]
        donor=rows[p['donor']]
        ids=candidate_ids[r['family']]
        i,j=ids[r['expected_index']],ids[donor['expected_index']]
        ideal=rawlp[pi] if r['role']=='calendar' else base[p['recipient']]
        kl=lambda a,b:max(0.,float(np.sum(np.exp(a)*(a-b))))
        sourcepred=sources[source]['conditional64'][pi]
        denominator=float(sourcepred@sourcepred)
        return dict(record_kind='conditional_composition',pair_id=int(pi),source_seed=source,target_seed=target_seed,
            template=r['template'],phase=r['phase'],role=r['role'],recipient_value=r['value'],donor_value=donor['value'],
            method=method,operation=operation,
            primary_conditional_kl=kl(ideal,lp),conditional_noop_kl=kl(ideal,base[p['recipient']]),
            reference_kl=kl(reference,lp),reference_noop_kl=kl(reference,base[p['recipient']]),
            intervention_kl=kl(lp,base[p['recipient']]),raw_intervention_kl=kl(rawlp[pi],base[p['recipient']]),
            donor_margin_change=float((lp[j]-lp[i])-(base[p['recipient'],j]-base[p['recipient'],i])),
            raw_margin_change=float((rawlp[pi,j]-rawlp[pi,i])-(base[p['recipient'],j]-base[p['recipient'],i])),
            expected_donor_probability=float(np.exp(lp[j])),expected_recipient_probability=float(np.exp(lp[i])),
            donor_family_correct=bool(np.argmax(lp[ids])==donor['expected_index']),
            full_prediction_token=int(np.argmax(lp)),candidate_probabilities=np.exp(lp[ids]).tolist(),
            vector_error_to_raw=float(np.sum((prediction-dh[pi])**2)),raw_vector_energy=float(dh[pi]@dh[pi]),
            vector_energy=float(prediction@prediction),source_vector_energy=denominator,
            vector_error_to_source=float(np.sum((prediction-sourcepred)**2)))

    def evaluate(predictions,source,target_seed,method,operation,reference_lp=None,save_reference=False):
        stored=[]
        for offset,ix in enumerate(batches):
            delta=np.zeros((len(ix),width,768),dtype=np.float64)
            delta[np.arange(len(ix)),slots[ri[ix]]]=predictions[ix]
            lp,_=forward(ri[ix],layer=5,deltas=delta)
            if save_reference:
                stored.append(lp)
            for k,pi in enumerate(ix):
                # Source controls compare to their intended conditional raw/zero
                # objective; transferred maps compare to actual source forward.
                reference=(rawlp[pi] if positive[pi] else base[ri[pi]]) if reference_lp is None else reference_lp[offset*cfg['batch_size']+k]
                m=metric(pi,lp[k],reference,predictions[pi],source,target_seed,method,operation)
                metrics.append(m)
                with (run/'metrics.raw.jsonl').open('a') as f:
                    f.write(json.dumps(m)+'\n')
        return np.concatenate(stored) if stored else None

    for s in range(1,6):
        ref=evaluate(sources[s]['conditional64'],s,None,'conditional64','source',save_reference=True)
        np.save(run/f'composition_source_s{s}_test_logprobs.npy',ref)
        for method,pred in sources[s].items():
            if method!='conditional64':
                evaluate(pred,s,None,method,'source')
        for t in range(1,6):
            if t==s:
                continue
            for method,pred in maps[s,t].items():
                evaluate(pred,s,t,method,'cross_seed',reference_lp=ref)
        progress('conditional_behavior',source=s,rows=len(metrics))
    expected=len(test)*(5*7+20*4)
    checks.update(conditional_complete_panel=len(metrics)==expected,
        conditional_only_test_interventions=all(m['phase']=='test' for m in metrics),
        conditional_finite=all(np.isfinite(m['primary_conditional_kl']) for m in metrics),
        source_and_maps_frozen_before_test_intervention=True,
        all_twenty_directions=len(maps)==20)
    write(run/'composition_summary.json',dict(rows=len(metrics),expected_rows=expected,checks=checks,
        fit_pairs=len(fit),test_pairs=len(test),specification=spec,
        scope='controlled new-prefix application; all12values; no natural confirmation or solver novelty'))
    return dict(metrics=metrics,checks=checks)
