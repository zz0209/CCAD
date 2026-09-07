"""Fit source allocations for reusable component operations; evaluate real LM edits.

All fit/selection/calibration uses the original development panel. A saved fit
can be replayed on another material run without fitting. Full-vocabulary LM
outcomes are never used by the shared OLS or ridge coefficient estimators.
"""
import argparse,hashlib,json,traceback
from pathlib import Path
from composition_runtime import CompositionRun,ROOT,write,np
from run_pair_complete_correspondence import choose_fit,closed_subset
from ccad.complete_support import pair_design,orthogonal_least_squares_support
from ccad.pair_complete_correspondence import balanced_common_weight
from ccad.component_correspondence import component_metric,metric_factor,component_update,predefined_masks


def measure(work,name,factor,mask_name,consumer,dose,vectors,reference,source_seed,target_seed):
    ids=work.evaluation_ids;delta=np.zeros_like(work.h,dtype=np.float64);delta[:,0]=vectors
    lp=work.evaluate(delta,ids);reference=lp if reference is None else reference
    for j,values,ref in zip(ids,lp,reference):
        j=int(j);row=work.rows[j];small=values[work.labels];rs=ref[work.labels];base=work.base[j,work.labels]
        num=float(np.logaddexp(small[1],small[3])-np.logaddexp(small[0],small[2]));past=float(np.logaddexp(small[2],small[3])-np.logaddexp(small[0],small[1]))
        rnum=float(np.logaddexp(rs[1],rs[3])-np.logaddexp(rs[0],rs[2]));rpast=float(np.logaddexp(rs[2],rs[3])-np.logaddexp(rs[0],rs[1]))
        bnum=float(np.logaddexp(base[1],base[3])-np.logaddexp(base[0],base[2]));bpast=float(np.logaddexp(base[2],base[3])-np.logaddexp(base[0],base[1]))
        work.record(dict(kind='component_intervention',method=name,factor=factor,mask=mask_name,consumer=consumer,dose=dose,source_seed=source_seed,target_seed=target_seed,row_id=j,block=row['block'],cue_id=row['cue_id'],template=row['template'],role=row.get('cue_role','temporal'),number=row['number'],past=row['past'],distractor=row['distractor'],kl_reference=max(0.,float(np.sum(np.exp(ref)*(ref-values)))),label=int(np.argmax(small)),source_label=int(np.argmax(rs)),label_logprobs=small.tolist(),source_label_logprobs=rs.tolist(),number_logodds=num,past_logodds=past,source_number_logodds=rnum,source_past_logodds=rpast,number_change=num-bnum,past_change=past-bpast,source_number_change=rnum-bnum,source_past_change=rpast-bpast,delta_norm=float(np.linalg.norm(vectors[j])),scope=work.cfg['evidence_level'],semantic_expected_label=None))
    return lp


def fit_methods(work,x,zs,decoder,donors,fit,cal,old,budget,eta):
    from scipy.optimize import linear_sum_assignment
    cfg=work.cfg;result={};diagnostics=[]
    # Every old aggregate map has a unique source-code lift here only because
    # the observed source decoder has full row rank. Verify that assumption.
    coefficient=old['source_coefficients'];inverse=np.linalg.pinv(coefficient)
    assert np.linalg.matrix_rank(coefficient)==len(decoder)
    for name,prefix in [('aggregate_ols','ols_complete'),('aggregate_full','full_code_balanced'),('aggregate_raw','raw_balanced')]:
        members=old[prefix+'_members'] if name=='aggregate_ols' else np.arange(x.shape[1] if name=='aggregate_full' else work.h.shape[-1])
        result[name]=dict(members=members,weights=old[prefix+'_weights']@inverse,input_kind='raw' if name=='aggregate_raw' else 'codes')
    rawdev=np.load(work.checked(ROOT/cfg['development_material_run']/'raw_cache.npz'))['layer15'][:,0].astype(np.float64)
    for family in ['half','singleton']:
        factor=metric_factor(component_metric(decoder,family));response=zs@factor
        design,target=pair_design(x,response,donors,eta);members,selection=orthogonal_least_squares_support(design,target,budget)
        variants=[('component_'+family,x,members,'codes')]
        if family=='half':
            full,full_diag=choose_fit(x,response,donors,fit,cal,cfg['alphas'],eta)
            rms=np.sqrt(np.mean(x*x,axis=0));dense=np.argsort(-rms*np.linalg.norm(full,axis=1),kind='stable')[:budget]
            variants.extend([('dense_half',x,dense,'codes'),('full_half',x,np.arange(x.shape[1]),'codes'),('raw_half',rawdev,np.arange(rawdev.shape[1]),'raw')])
        for name,values,selected,input_kind in variants:
            weights,fd=choose_fit(values[:,selected],response,donors,fit,cal,cfg['alphas'],eta)
            allocation=np.linalg.solve(factor.T,weights.T).T
            result[name]=dict(members=selected,weights=allocation,input_kind=input_kind)
            diagnostics.append(dict(method=name,family=family,members=selected.tolist(),selection=selection if name.startswith('component_') else None,**fd))
    # Equal-count one-to-one matched atoms, then a separate scalar regression
    # for each source member. Matching sees all development rows, not LM output.
    design,target=pair_design(x,zs,donors,eta)
    norms=np.linalg.norm(design,axis=0);active=np.flatnonzero(norms>max(norms.max()*1e-8,1e-12))
    corr=(design[:,active].T@target)/np.maximum(norms[active,None]*np.linalg.norm(target,axis=0)[None,:],1e-30)
    ti,si=linear_sum_assignment(-corr*corr)
    selected=np.empty(len(decoder),dtype=int);selected[si]=active[ti];allocation=np.zeros((len(selected),len(decoder)))
    atom_fits=[]
    for i,member in enumerate(selected):
        w,fd=choose_fit(x[:,[member]],zs[:,[i]],donors,fit,cal,cfg['alphas'],eta);allocation[i,i]=w[0,0];atom_fits.append(fd)
    result['matched_atoms']=dict(members=selected,weights=allocation,input_kind='codes')
    diagnostics.append(dict(method='matched_atoms',family='one-to-one code match',members=selected.tolist(),scalar_fits=atom_fits))
    return result,diagnostics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_component_correspondence.py',['scripts/run_pair_complete_correspondence.py','src/ccad/component_correspondence.py','src/ccad/complete_support.py','src/ccad/pair_complete_correspondence.py','src/ccad/factor_correspondence.py','src/ccad/nip_baselines.py']);error=None
    manifest=json.loads((work.run/'manifest.json').read_text());manifest.update(milestone='reusable source component operations',candidate_family_frozen=bool(work.cfg.get('frozen_fit_run')),mean_constants_source_split='Uncentered source code; complete contribution uses no centering mean',statistics_unit='Authored lexical/cue pairs and source components, five shared SAE seeds');write(work.run/'manifest.json',manifest)
    try:
        cfg=work.cfg
        if cfg.get('freeze_json'):
            frozen=json.loads(work.checked(ROOT/cfg['freeze_json']).read_text())
            for record in frozen['files']:
                path=work.checked(ROOT/record['path'])
                assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['path']
            work.checks['frozen_input_identity']=True
        work.load();dev=ROOT/cfg['development_material_run'];panel=json.loads(work.checked(dev/'panel.json').read_text())
        if cfg.get('freeze_json'):
            texts=[r['text'] for r in work.rows]
            assert hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()==frozen['evaluation_texts_sha256']
            work.checks['frozen_new_texts']=True
        fit=np.array([r['block'] in cfg['fit_blocks'] for r in panel['rows']]);cal=~fit
        donors={f:np.array([r[f] for r in panel['pairs']]) for f in ['number','time']}
        assets={};geometry=[];allfits=[];allmasks=[];expected_rows=0
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];a=np.load(work.checked(dev/f'seed{seed}_codes.npz'));b=np.load(work.checked(ROOT/cfg['evaluation_code_run']/f'seed{seed}_codes.npz'))
            assets[seed]=dict(dev=a['number_z'].astype(np.float64),evaluation=b['number_z'].astype(np.float64),source=np.load(work.checked(ROOT/cfg['source_run']/f'seed{seed}_source_coordinates.npz')))
        for source_seed,target_seed in cfg['seed_pairs']:
            source=assets[source_seed];target=assets[target_seed]
            for factor,budget in cfg['source_budgets'].items():
                prefix=f'{factor}_{budget}_';coordinates=source['source'];support=coordinates[prefix+'support'];coeff=coordinates[prefix+'coefficients'];basis=coordinates[prefix+'basis'];decoder=coeff@basis.T
                source_dev=source['dev'][:,support];source_eval=source['evaluation'][:,support]
                eta=balanced_common_weight(source_dev[fit]@coeff,closed_subset(donors[factor],fit))
                if cfg.get('frozen_fit_run'):
                    origin=ROOT/cfg['frozen_fit_run'];meta=json.loads(work.checked(origin/f'maps_s{source_seed}_t{target_seed}_{factor}.json').read_text());arrays=np.load(work.checked(origin/f'maps_s{source_seed}_t{target_seed}_{factor}.npz'))
                    assert np.array_equal(arrays['source_members'],support) and np.array_equal(arrays['source_decoder'],decoder)
                    methods={name:dict(members=arrays[name+'_members'],weights=arrays[name+'_weights'],input_kind=kind) for name,kind in meta['input_kinds'].items()};fits=[]
                else:
                    old=np.load(work.checked(ROOT/cfg['aggregate_run']/f'maps_s{source_seed}_t{target_seed}_{factor}.npz'))
                    methods,fits=fit_methods(work,target['dev'],source_dev,decoder,donors[factor],fit,cal,old,budget,eta)
                    replay=target['evaluation'][:,methods['aggregate_ols']['members']]@methods['aggregate_ols']['weights']@decoder
                    oldpred=target['evaluation'][:,old['ols_complete_members']]@old['ols_complete_weights']@basis.T
                    work.checks[f'aggregate_lift_s{source_seed}_{factor}']=bool(np.max(np.abs(replay-oldpred))<1e-8)
                arrays=dict(source_members=support,source_decoder=decoder)
                for name,method in methods.items():arrays.update({name+'_members':method['members'],name+'_weights':method['weights']})
                np.savez_compressed(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.npz',**arrays)
                write(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.json',dict(source_seed=source_seed,target_seed=target_seed,factor=factor,input_kinds={name:m['input_kind'] for name,m in methods.items()},source_common_weight=eta,source_decoder_rank=int(np.linalg.matrix_rank(decoder)),source_decoder_condition=float(np.linalg.cond(decoder)),fit_rows=0 if cfg.get('frozen_fit_run') else len(source_dev)))
                predictions={name:(work.h[:,0] if m['input_kind']=='raw' else target['evaluation'])[:,m['members']]@m['weights'] for name,m in methods.items()}
                np.savez_compressed(work.run/f'predictions_s{source_seed}_t{target_seed}_{factor}.npz',source=source_eval,**predictions)
                masks=predefined_masks(budget,source_seed*100+(0 if factor=='number' else 1));allmasks.extend(dict(source_seed=source_seed,factor=factor,name=name,source_members=support.tolist(),scales=mask.tolist()) for name,mask in masks)
                for mask_name,mask in masks:
                    for consumer in cfg['consumers']:
                        code_change=lambda z:z[work.donors[factor]]-z if consumer=='contrast' else -z
                        for dose in cfg.get('mask_doses',{}).get(mask_name,cfg['doses']):
                            teacher=dose*component_update(code_change(source_eval),decoder,mask)
                            reference=measure(work,'source',factor,mask_name,consumer,dose,teacher,None,source_seed,target_seed)
                            for name,pred in predictions.items():
                                vectors=dose*component_update(code_change(pred),decoder,mask)
                                measure(work,name,factor,mask_name,consumer,dose,vectors,reference,source_seed,target_seed)
                                for role in ['temporal','quoted']:
                                    ids=np.array([i for i in work.evaluation_ids if work.rows[i].get('cue_role','temporal')==role],dtype=int)
                                    if len(ids):geometry.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,mask=mask_name,consumer=consumer,dose=dose,role=role,method=name,n=len(ids),error_sse=float(np.sum((vectors[ids]-teacher[ids])**2)),teacher_energy=float(np.sum(teacher[ids]**2))))
                            expected_rows+=len(work.evaluation_ids)*(1+len(predictions))
                allfits.extend(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,**r) for r in fits)
                work.progress('FACTOR_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor,methods=len(methods),evaluation_rows=len(work.evaluation_ids),fit_rows=0 if cfg.get('frozen_fit_run') else len(source_dev))
        write(work.run/'fit_diagnostics.json',dict(rows=allfits));write(work.run/'geometry.json',dict(rows=geometry));write(work.run/'source_masks.json',dict(rows=allmasks,rule='Source ranking order or fixed RNG; no target/outcome selection'))
        work.checks['all_rows']=len(work.metrics)==expected_rows
        work.checks['unique_rows']=len(work.metrics)==len({(r['source_seed'],r['target_seed'],r['factor'],r['mask'],r['consumer'],r['dose'],r['method'],r['row_id']) for r in work.metrics})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
