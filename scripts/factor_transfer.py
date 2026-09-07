"""End-to-end FCC evaluation using common compact source-native teachers."""
import json
import numpy as np
from ccad.factor_correspondence import compact_source,fit_controls


def run_transfer(cfg,run,pair_cache,cache,measure,progress):
    seeds=cfg['transfer']['seeds'];transfer=cfg['transfer'];source_records=[]
    for source_seed in seeds:
        zs,ds=cache[source_seed]
        for factor,(pp,ri,di,dh,discovery,sign,slot) in pair_cache.items():
            xs=zs[di,slot]-zs[ri,slot];source=compact_source(xs,ds,discovery,sign,transfer['source_budget']);basis=source['basis'];teacher=source['coordinates']@basis.T
            completed={target for target in seeds if (run/f'transfer_s{source_seed}_t{target}_{factor}.json').exists()} if cfg.get('resume_parent') else set()
            source_exists=(run/f'source{source_seed}_{factor}_teacher.npz').exists() if cfg.get('resume_parent') else False
            if source_exists:
                saved=np.load(run/f'source{source_seed}_{factor}_teacher.npz');assert np.array_equal(source['support'],saved['support']);np.testing.assert_allclose(teacher,saved['native_delta'],rtol=1e-12,atol=1e-12)
                source.update(basis=saved['basis'],coefficients=saved['coefficients'],coordinates=saved['coordinates']);basis=source['basis'];teacher=saved['native_delta']
            source_record=dict(source_seed=source_seed,factor=factor,support=source['support'].tolist(),scores=source['score'].tolist(),rank=source['rank'],span_error=source['span_error'],discovery_singular_values=source['discovery_singular_values'].tolist());source_records.append(source_record)
            if len(completed)==len(seeds)-1:continue
            identity=dict(checkpoint='transfer',source_seed=source_seed,target_seed=source_seed,source_members=transfer['source_budget'],source_rank=source['rank'])
            ref=measure(factor,'source_native_teacher',teacher,identity,None,teacher,persist=not source_exists)
            if not source_exists:
                measure(factor,'noop_teacher_reference',np.zeros_like(teacher),identity,ref,teacher)
                _,_,task_vt=np.linalg.svd(source['coordinates'][discovery],full_matrices=False)
                for rank in [1,4]:
                    projector=task_vt[:rank].T@task_vt[:rank];compressed=source['coordinates']@projector@basis.T
                    measure(factor,'source_pca_rank'+str(rank),compressed,identity,ref,teacher)
                np.savez_compressed(run/f'source{source_seed}_{factor}_teacher.npz',support=source['support'],basis=basis,coefficients=source['coefficients'],coordinates=source['coordinates'],native_delta=teacher)
            fit=np.array([p['block'] in transfer['fit_blocks'] for p in pp]);cal=np.array([p['block'] in transfer['calibration_blocks'] for p in pp]);assert np.array_equal(fit|cal,discovery) and not np.any(fit&cal)
            for target_seed in seeds:
                if source_seed==target_seed or target_seed in completed:continue
                zt,dt=cache[target_seed];xt=zt[di,slot]-zt[ri,slot]
                predictions,diagnostics=fit_controls(xt,dh,xs,source,fit,cal,discovery,transfer['target_budgets'],transfer['ridge_fractions'],random_seed=source_seed*100+target_seed)
                identity=dict(checkpoint='transfer',source_seed=source_seed,target_seed=target_seed,source_members=transfer['source_budget'],source_rank=source['rank']);arrays={}
                for name,(coordinates,w) in predictions.items():
                    singular=np.linalg.svd(coordinates[discovery],compute_uv=False);effective_rank=int(np.sum(singular>max(float(singular[0])*1e-6,1e-12)))
                    identity=dict(identity,target_members=int(np.sum(np.linalg.norm(w,axis=1)>0)) if w is not None else dh.shape[1],prediction_rank_discovery=effective_rank)
                    delta=coordinates@basis.T;measure(factor,name,delta,identity,ref,teacher)
                    if w is not None:
                        arrays[name+'_coefficients']=w;arrays[name+'_predicted_coordinates']=coordinates
                    if name.startswith('fcc_'):
                        support=np.flatnonzero(np.linalg.norm(w,axis=1)>0);native=xt[:,support]@dt[support]
                        measure(factor,name+'_same_support_native',native,identity,ref,teacher)
                        measure(factor,name+'_same_support_projection',native@basis@basis.T,identity,ref,teacher)
                        gain=float(np.sum(native[discovery]*teacher[discovery])/max(np.sum(native[discovery]**2),1e-12))
                        measure(factor,name+'_same_support_native_gain',gain*native,identity,ref,teacher)
                        diagnostics[name]['native_same_support_discovery_gain']=gain
                        # Wrong input factor at the same recipient site: retain
                        # the fitted map but feed the other variable's change.
                        other='distractor' if factor=='subject' else 'subject';otherpairs={p['base']:p['donor'] for p in pair_cache[other][0]};otherdi=np.array([otherpairs[int(i)] for i in ri]);wrong=(zt[otherdi,slot]-zt[ri,slot])@w@basis.T
                        measure(factor,name+'_wrong_factor',wrong,identity,ref,teacher)
                        # Exact signed per-input group relations, one unchosen
                        # example from each held lexical block; all IDs saved.
                        example=[k for k,p in enumerate(pp) if not discovery[k] and p['template']=='pp' and not p['base']%4]
                        left=xs[example][:,source['support'],None]*source['coefficients'][None]
                        right=xt[example][:,support,None]*w[support][None]
                        relation=np.einsum('nir,njr->nij',left,right)
                        arrays[name+'_relation_examples']=relation;arrays[name+'_example_pair_ids']=np.array([pp[k]['id'] for k in example]);arrays[name+'_target_support']=support
                stem=f'transfer_s{source_seed}_t{target_seed}_{factor}';np.savez_compressed(run/(stem+'.npz'),**arrays)
                (run/(stem+'.json')).write_text(json.dumps(dict(source=source_record,controls=diagnostics,selection='all source and target choices use discovery only; calibration internal to discovery; held blocks remain development'),indent=2)+'\n')
                progress('TRANSFER_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor)
    (run/'source_teachers.json').write_text(json.dumps(dict(rows=source_records),indent=2)+'\n')
