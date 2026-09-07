"""Test whether a contrast map can also reuse a complete source-group removal.

Fits consume only the old temporal development panel. Retrospective evaluation
uses the already exposed role panel and is not a new independent confirmation.
Complete removal predicts uncentered Dz, not a target-native intervention or a
semantic ground-truth deletion. Original contrast maps are never overwritten.
"""
import argparse
import json
from pathlib import Path
import traceback

from composition_runtime import CompositionRun, ROOT, write, np
from ccad.pair_complete_correspondence import pair_parts, pair_ridge, balanced_common_weight


def closed_subset(donors, mask):
    ids=np.flatnonzero(mask);inverse=np.full(len(mask),-1,dtype=int);inverse[ids]=np.arange(len(ids))
    selected=inverse[donors[ids]]
    if np.any(selected<0):raise ValueError('A fit/calibration split cuts a reciprocal pair')
    return selected


def choose_fit(x,y,donors,fit,cal,alphas,eta):
    local=closed_subset(donors,fit);cal_local=closed_subset(donors,cal);errors=[]
    for alpha in alphas:
        w,_=pair_ridge(x[fit],y[fit],local,alpha,eta)
        em,ep=pair_parts(x[cal]@w-y[cal],cal_local)
        errors.append(float(np.mean(np.sum(em**2,axis=1))+eta*np.mean(np.sum(ep**2,axis=1))))
    alpha=alphas[int(np.argmin(errors))]
    w,diag=pair_ridge(x,y,donors,alpha,eta)
    return w,dict(diag,alpha=alpha,calibration_losses=errors,fit_rows=int(fit.sum()),calibration_rows=int(cal.sum()),refit_rows=len(x))


def evaluate(work,name,factor,consumer,vectors,reference,source_seed,target_seed=None):
    delta=np.zeros_like(work.h,dtype=np.float64)
    delta[:,0]=vectors
    lp=work.evaluate(delta)
    reference=lp if reference is None else reference
    for i,(values,ref) in enumerate(zip(lp,reference)):
        row=work.rows[i];small=values[work.labels];refsmall=ref[work.labels]
        work.record(dict(kind='intervention',method=name,factor=factor,consumer=consumer,
            source_seed=source_seed,target_seed=target_seed,row_id=i,block=row['block'],cue_id=row['cue_id'],
            template=row['template'],role=row['cue_role'],
            number=row['number'],past=row['past'],distractor=row['distractor'],
            label=int(np.argmax(small)),source_label=int(np.argmax(refsmall)),source_label_agreement=bool(np.argmax(small)==np.argmax(refsmall)),
            label_logprobs=small.tolist(),source_label_logprobs=refsmall.tolist(),
            number_logodds=float(np.logaddexp(small[1],small[3])-np.logaddexp(small[0],small[2])),
            past_logodds=float(np.logaddexp(small[2],small[3])-np.logaddexp(small[0],small[1])),
            kl_reference=max(0.,float(np.sum(np.exp(ref)*(ref-values)))),
            kl_to_baseline=max(0.,float(np.sum(np.exp(work.base[i])*(work.base[i]-values)))),
            delta_norm=float(np.linalg.norm(vectors[i])),
            semantic_expected_label=None,reference_kind='same source group and same consumer',
            scope='retrospective exposed role panel; no semantic accuracy assigned to deletion'))
    return lp


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    path=parser.parse_args().config
    work=CompositionRun(path,'scripts/run_pair_complete_correspondence.py',['src/ccad/pair_complete_correspondence.py']);error=None
    manifest=json.loads((work.run/'manifest.json').read_text())
    manifest.update(milestone='contrast-to-complete-group-consumer',candidate_family_frozen=False,
        mean_constants_source_split='Uncentered source-group Dz. Legacy mean correction uses old 384-row development panel only; no independent mean-split fit claimed.',
        statistics_unit='Five shared controlled seeds, lexical blocks and reciprocal cue pairs; exposed role panel is retrospective development')
    write(work.run/'manifest.json',manifest)
    try:
        cfg=work.cfg;work.load()
        work.checks['final_common_position']=bool(np.array_equal(work.positions[:,0],work.positions[:,1]))
        if not work.checks['final_common_position']:raise ValueError('This consumer requires common final-token factor positions')
        dev=ROOT/cfg['development_material_run'];panel=json.loads(work.checked(dev/'panel.json').read_text())
        rows=panel['rows'];n=len(rows);fit=np.array([r['block'] in cfg['fit_blocks'] for r in rows]);cal=~fit
        donors={f:np.array([p[f] for p in panel['pairs']]) for f in ['number','time']}
        devcache=np.load(work.checked(dev/'raw_cache.npz'));raw_dev=devcache['layer15'][:,0].astype(np.float64)
        raw_eval=work.h[:,0].astype(np.float64)
        assets={}
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];a=np.load(work.checked(dev/f'seed{seed}_codes.npz'))
            b=np.load(work.checked(ROOT/cfg['evaluation_code_run']/f'seed{seed}_codes.npz'))
            work.checks[f'codes_same_final_slot_{seed}']=bool(np.array_equal(a['number_z'],a['time_z']) and np.array_equal(b['number_z'],b['time_z']))
            assets[seed]=dict(dev=a['number_z'].astype(np.float64),evaluation=b['number_z'].astype(np.float64),
                coordinates=np.load(work.checked(ROOT/cfg['source_run']/f'seed{seed}_source_coordinates.npz')),
                checkpoint=spec['path'])
        diagnostics=[];geometry=[];examples=[]
        from safetensors import safe_open
        for source_seed,target_seed in cfg['seed_pairs']:
            source=assets[source_seed];target=assets[target_seed]
            maps=np.load(work.checked(ROOT/cfg['correspondence_run']/f'maps_s{source_seed}_t{target_seed}.npz'))
            with safe_open(work.checked(Path(target['checkpoint'])/'sae.safetensors'),framework='numpy') as weights:
                decoder=weights.get_tensor('W_dec').astype(np.float64)
            work.checked(Path(target['checkpoint'])/'cfg.json')
            for factor in ['number','time']:
                prefix=f"{factor}_{cfg['source_budgets'][factor]}_";coord=source['coordinates']
                support=coord[prefix+'support'];coef=coord[prefix+'coefficients'];basis=coord[prefix+'basis']
                y=source['dev'][:,support]@coef;ye=source['evaluation'][:,support]@coef
                members=maps[factor+'_fcc_members'];x=target['dev'][:,members];xe=target['evaluation'][:,members]
                oldw=maps[factor+'_fcc_group_coefficients'][members].astype(np.float64)
                oldpred=xe@oldw;bias=np.mean(y-x@oldw,axis=0)
                common_weight=balanced_common_weight(y[fit],closed_subset(donors[factor],fit))
                candidates={'legacy_contrast':oldpred,'legacy_plus_mean':oldpred+bias}
                saved={'source_support':support,'basis':basis,'source_coefficients':coef,'target_members':members,'legacy_weights':oldw,'mean_correction':bias}
                fit_specs=[('paired_complete',x,xe,1.),('paired_balanced',x,xe,common_weight),
                           ('full_code_complete',target['dev'],target['evaluation'],1.),('raw_complete',raw_dev,raw_eval,1.)]
                if cfg.get('pair_anchor_followup'):
                    # Both inputs are required: retain the frozen odd prediction and
                    # fit only the pair-common contribution. This is a donor-conditioned
                    # removal, not an unpaired context-to-contribution model.
                    candidates={'legacy_contrast':oldpred}
                    old_odd,_=pair_parts(oldpred,work.donors[factor])
                    _,y_common=pair_parts(y,donors[factor])
                    for name,train,test in [('anchor_compact',x,xe),('anchor_full_code',target['dev'],target['evaluation']),('anchor_raw',raw_dev,raw_eval)]:
                        _,train_common=pair_parts(train,donors[factor]);_,test_common=pair_parts(test,work.donors[factor])
                        w,diag=choose_fit(train_common,y_common,donors[factor],fit,cal,cfg['alphas'],1.)
                        candidates[name]=old_odd+test_common@w;saved[name+'_common_weights']=w
                        work.checks[f'anchor_contrast_s{source_seed}_t{target_seed}_{factor}_{name}']=bool(np.allclose(
                            candidates[name][work.donors[factor]]-candidates[name],oldpred[work.donors[factor]]-oldpred,rtol=1e-11,atol=1e-11))
                        diagnostics.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,**diag))
                    fit_specs=[]
                for name,train,test,eta in fit_specs:
                    w,diag=choose_fit(train,y,donors[factor],fit,cal,cfg['alphas'],eta)
                    candidates[name]=test@w;saved[name+'_weights']=w
                    diagnostics.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,members=members.tolist() if name.startswith('paired_') else None,**diag))
                ym,yp=pair_parts(y,donors[factor]);em,ep=pair_parts(x@oldw-y,donors[factor])
                decomposition_error=abs(float(np.sum((x@oldw-y)**2)-np.sum(em**2)-np.sum(ep**2)))
                work.checks[f'energy_identity_s{source_seed}_t{target_seed}_{factor}']=decomposition_error<1e-8*max(1.,float(np.sum((x@oldw-y)**2)))
                diagnostics.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method='legacy_decomposition',
                    source_odd_energy=float(np.sum(ym**2)),source_common_energy=float(np.sum(yp**2)),
                    legacy_odd_error=float(np.sum(em**2)),legacy_common_error=float(np.sum(ep**2)),
                    unobserved_common_error_fraction=float(np.sum(ep**2)/max(np.sum(em**2)+np.sum(ep**2),1e-20)),
                    energy_identity_error=decomposition_error,balanced_common_weight=common_weight))
                np.savez_compressed(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.npz',**saved)
                true=ye@basis.T
                physical={name:prediction@basis.T for name,prediction in candidates.items()}
                if not cfg.get('pair_anchor_followup'):
                    physical['same_members_native']=target['evaluation'][:,members]@decoder[members]
                for consumer in ['contrast','complete_removal']:
                    teacher_vectors=true[work.donors[factor]]-true if consumer=='contrast' else -true
                    teacher=evaluate(work,'source_teacher',factor,consumer,teacher_vectors,None,source_seed)
                    for name,values in physical.items():
                        vectors=values[work.donors[factor]]-values if consumer=='contrast' else -values
                        evaluate(work,name,factor,consumer,vectors,teacher,source_seed,target_seed)
                        for role in ['temporal','quoted']:
                            mask=np.array([r['cue_role']==role for r in work.rows])
                            residual=vectors[mask]-teacher_vectors[mask]
                            geometry.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,consumer=consumer,method=name,role=role,n=int(mask.sum()),
                                error_sse=float(np.sum(residual**2)),reference_energy=float(np.sum(teacher_vectors[mask]**2)),candidate_energy=float(np.sum(vectors[mask]**2))))
                # Mean correction changes no reciprocal contrast, including new contexts.
                if 'legacy_plus_mean' in physical:
                    work.checks[f'mean_correction_contrast_s{source_seed}_t{target_seed}_{factor}']=bool(np.allclose(
                        physical['legacy_contrast'][work.donors[factor]]-physical['legacy_contrast'],
                        physical['legacy_plus_mean'][work.donors[factor]]-physical['legacy_plus_mean'],rtol=1e-12,atol=1e-12))
                work.progress('FACTOR_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor,methods=len(physical))
            del decoder
        write(work.run/'fit_diagnostics.json',dict(rows=diagnostics))
        write(work.run/'geometry.json',dict(rows=geometry))
        work.checks['all_rows']=len(work.metrics)==len(cfg['seed_pairs'])*2*2*work.n*(5 if cfg.get('pair_anchor_followup') else 8)
        work.checks['unique_rows']=len(work.metrics)==len({(r['source_seed'],r['target_seed'],r['factor'],r['consumer'],r['method'],r['row_id']) for r in work.metrics})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
