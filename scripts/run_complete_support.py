"""Test one-input complete contribution maps with shared multi-consumer support.

Selection and fitting use only the original temporal development codes and
source coordinates. The exposed role panel is retrospective development.
"""
import argparse, json, traceback
from pathlib import Path
from composition_runtime import CompositionRun, ROOT, write, np
from run_pair_complete_correspondence import choose_fit, closed_subset, evaluate
from ccad.complete_support import pair_design, orthogonal_least_squares_support
from ccad.pair_complete_correspondence import pair_parts, balanced_common_weight
from ccad.factor_correspondence import group_support


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_complete_support.py',[
        'scripts/run_pair_complete_correspondence.py','src/ccad/complete_support.py',
        'src/ccad/pair_complete_correspondence.py','src/ccad/factor_correspondence.py','src/ccad/nip_baselines.py'])
    manifest=json.loads((work.run/'manifest.json').read_text())
    manifest.update(milestone='single-input complete contribution support',candidate_family_frozen=False,
        mean_constants_source_split='Uncentered source-group Dz; no independent centering mean or donor input at deployment.',
        statistics_unit='Five shared controlled seeds and authored lexical/cue pairs. Exposed role data are development.')
    write(work.run/'manifest.json',manifest);error=None
    try:
        work.load();cfg=work.cfg;dev=ROOT/cfg['development_material_run']
        panel=json.loads(work.checked(dev/'panel.json').read_text());rows=panel['rows']
        fit=np.array([r['block'] in cfg['fit_blocks'] for r in rows]);cal=~fit
        donors={f:np.array([r[f] for r in panel['pairs']]) for f in ['number','time']}
        rawdev=np.load(work.checked(dev/'raw_cache.npz'))['layer15'][:,0].astype(np.float64)
        assets={};diagnostics=[];geometry=[]
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];a=np.load(work.checked(dev/f'seed{seed}_codes.npz'));b=np.load(work.checked(ROOT/cfg['evaluation_code_run']/f'seed{seed}_codes.npz'))
            work.checks[f'codes_same_slot_{seed}']=bool(np.array_equal(a['number_z'],a['time_z']) and np.array_equal(b['number_z'],b['time_z']))
            assets[seed]=dict(dev=a['number_z'].astype(np.float64),evaluation=b['number_z'].astype(np.float64),
                source=np.load(work.checked(ROOT/cfg['source_run']/f'seed{seed}_source_coordinates.npz')),checkpoint=spec['path'])
        target_assets=assets
        if cfg.get('target_code_run'):
            target_assets={}
            target_root=ROOT/cfg['target_code_run']
            for spec in cfg['target_sae_checkpoints']:
                seed=spec['seed']
                a=np.load(work.checked(target_root/'development'/f'seed{seed}_codes.npz'))
                b=np.load(work.checked(target_root/'evaluation'/f'seed{seed}_codes.npz'))
                assert a['number_z'].shape==assets[seed]['dev'].shape
                assert b['number_z'].shape==assets[seed]['evaluation'].shape
                target_assets[seed]=dict(dev=a['number_z'].astype(np.float64),evaluation=b['number_z'].astype(np.float64),checkpoint=spec['path'])
        from safetensors import safe_open
        for source_seed,target_seed in cfg['seed_pairs']:
            source=assets[source_seed];target=target_assets[target_seed]
            old=np.load(work.checked(ROOT/cfg['correspondence_run']/f'maps_s{source_seed}_t{target_seed}.npz'))
            with safe_open(work.checked(Path(target['checkpoint'])/'sae.safetensors'),framework='numpy') as sf:decoder=sf.get_tensor('W_dec').astype(np.float64)
            x=target['dev'];xe=target['evaluation']
            for factor in ['number','time']:
                budget=cfg['source_budgets'][factor];prefix=f'{factor}_{budget}_';c=source['source']
                source_support=c[prefix+'support'];basis=c[prefix+'basis'];coef=c[prefix+'coefficients']
                y=source['dev'][:,source_support]@coef;ye=source['evaluation'][:,source_support]@coef
                legacy_members=old[factor+'_fcc_members'];legacy_w=old[factor+'_fcc_group_coefficients'][legacy_members]
                eta=balanced_common_weight(y[fit],closed_subset(donors[factor],fit))
                design,response=pair_design(x,y,donors[factor],eta)
                selected,sd=orthogonal_least_squares_support(design,response,budget)
                contrast_design,contrast_response=pair_design(x,y,donors[factor],0)
                contrast_selected,cd=orthogonal_least_squares_support(contrast_design,contrast_response,budget)
                lasso_members,ld=group_support(design,response,budget,scaling='native_units')
                full_w,full_diag=choose_fit(x,y,donors[factor],fit,cal,cfg['alphas'],eta)
                rms=np.sqrt(np.mean(x*x,axis=0));dense_members=np.argsort(-rms*np.linalg.norm(full_w,axis=1),kind='stable')[:budget]
                active=np.flatnonzero(rms>max(float(rms.max())*1e-8,1e-12))
                random_members=np.random.default_rng(source_seed*100+target_seed+(0 if factor=='number' else 10000)).choice(active,budget,replace=False)
                # The legacy reference is always the original target checkpoint
                # and its original map; continued encoders require refitting.
                candidates={'legacy_contrast':assets[target_seed]['evaluation'][:,legacy_members]@legacy_w,'full_code_balanced':xe@full_w}
                saved=dict(source_support=source_support,source_coefficients=coef,basis=basis,
                    legacy_contrast_members=legacy_members,legacy_contrast_weights=legacy_w,full_code_balanced_weights=full_w)
                diag=[dict(method='full_code_balanced',members_count=len(active),**full_diag)]
                specs=[('old_support_balanced',legacy_members,eta,{'selection':'original contrast group-L1'}),
                    ('ols_same_count',selected[:len(legacy_members)],eta,sd),
                    ('ols_complete',selected,eta,sd),('lasso_complete',lasso_members,eta,ld),
                    ('dense_complete',dense_members,eta,{'selection':'balanced full-code RMS-weighted strength'}),
                    ('random_complete',random_members,eta,{'selection':'uniform active target members'}),
                    ('ols_contrast_only',contrast_selected,0,cd)]
                for name,members,weight,selection in specs:
                    w,fd=choose_fit(x[:,members],y,donors[factor],fit,cal,cfg['alphas'],weight)
                    candidates[name]=xe[:,members]@w;saved[name+'_members']=members;saved[name+'_weights']=w
                    diag.append(dict(method=name,members_count=len(members),members=members.tolist(),selection=selection,**fd))
                for name,train,test,weight in [('full_code_complete',x,xe,1.),('raw_balanced',rawdev,work.h[:,0],eta),('raw_complete',rawdev,work.h[:,0],1.)]:
                    w,fd=choose_fit(train,y,donors[factor],fit,cal,cfg['alphas'],weight);candidates[name]=test@w;saved[name+'_weights']=w
                    diag.append(dict(method=name,**fd))
                physical={name:pred@basis.T for name,pred in candidates.items()}
                physical['ols_members_native']=xe[:,selected]@decoder[selected]
                saved['ols_native_members']=selected
                np.savez_compressed(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.npz',**saved)
                for row in diag:diagnostics.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,source_common_weight=eta,**row))
                teacher=ye@basis.T
                for consumer in ['contrast','complete_removal']:
                    actual=teacher[work.donors[factor]]-teacher if consumer=='contrast' else -teacher
                    reference=evaluate(work,'source',factor,consumer,actual,None,source_seed,target_seed)
                    for name,contribution in physical.items():
                        vectors=contribution[work.donors[factor]]-contribution if consumer=='contrast' else -contribution
                        evaluate(work,name,factor,consumer,vectors,reference,source_seed,target_seed)
                        for role in ['temporal','quoted']:
                            mask=np.array([r['cue_role']==role for r in work.rows]);e=vectors[mask]-actual[mask]
                            geometry.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,consumer=consumer,role=role,method=name,n=int(mask.sum()),error_sse=float(np.sum(e*e)),reference_energy=float(np.sum(actual[mask]**2))))
                work.progress('FACTOR_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor,methods=len(physical),old_support=len(legacy_members),new_support=len(selected))
            del decoder
        write(work.run/'fit_diagnostics.json',dict(rows=diagnostics));write(work.run/'geometry.json',dict(rows=geometry))
        work.checks['all_rows']=len(work.metrics)==len(cfg['seed_pairs'])*2*2*work.n*14
        work.checks['unique_rows']=len(work.metrics)==len({(r['source_seed'],r['target_seed'],r['factor'],r['consumer'],r['method'],r['row_id']) for r in work.metrics})
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
