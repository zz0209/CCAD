"""Replay frozen signed maps with absolute-source-state constraints."""
import argparse,hashlib,json,traceback,time
from pathlib import Path
from composition_runtime import CompositionRun,ROOT,write,np
from run_component_correspondence import measure
from ccad.component_correspondence import component_update,predefined_masks
from ccad.source_state_projection import project_source_state,projection_diagnostics

BASE_METHODS=['aggregate_ols','dense_half','matched_atoms','aggregate_full','aggregate_raw']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_source_state_correspondence.py',['scripts/run_component_correspondence.py','src/ccad/component_correspondence.py','src/ccad/source_state_projection.py']);error=None
    manifest=json.loads((work.run/'manifest.json').read_text());manifest.update(milestone='admissible absolute source state',candidate_family_frozen=bool(work.cfg.get('freeze_json')),mean_constants_source_split='Uncentered absolute source codes; project before signed differences',statistics_unit='Authored lexical/cue pairs and source components, shared SAE seeds');write(work.run/'manifest.json',manifest)
    try:
        cfg=work.cfg
        if cfg.get('freeze_json'):
            frozen=json.loads(work.checked(ROOT/cfg['freeze_json']).read_text())
            for record in frozen['files']:
                path=work.checked(ROOT/record['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['path']
            work.checks['frozen_input_identity']=True
        work.load()
        if cfg.get('freeze_json'):
            assert hashlib.sha256(json.dumps([r['text'] for r in work.rows],ensure_ascii=False).encode()).hexdigest()==frozen['evaluation_texts_sha256']
            work.checks['frozen_new_texts']=True
        assets={}
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];b=np.load(work.checked(ROOT/cfg['evaluation_code_run']/f'seed{seed}_codes.npz'))
            assets[seed]=b['number_z'].astype(np.float64)
        geometry=[];diagnostics=[];masks_record=[];expected_rows=0
        for source_seed,target_seed in cfg['seed_pairs']:
            for factor,budget in cfg['source_budgets'].items():
                origin=ROOT/cfg['initial_map_run'];meta=json.loads(work.checked(origin/f'maps_s{source_seed}_t{target_seed}_{factor}.json').read_text());old=np.load(work.checked(origin/f'maps_s{source_seed}_t{target_seed}_{factor}.npz'))
                support=old['source_members'];decoder=old['source_decoder'];ze=assets[source_seed][:,support]
                assert np.all(ze>=0) and len(support)==budget
                predictions={};arrays=dict(source_members=support,source_decoder=decoder);kinds={};projection_names={}
                for name in BASE_METHODS:
                    members=old[name+'_members'];weights=old[name+'_weights'];kind=meta['input_kinds'][name]
                    v=(work.h[:,0] if kind=='raw' else assets[target_seed])[:,members]@weights
                    predictions[name]=v;arrays.update({name+'_members':members,name+'_weights':weights});kinds[name]=kind;projection_names[name]='identity'
                    for projection in cfg['state_projections']:
                        start=time.perf_counter();u,metric=project_source_state(v,decoder,projection)
                        row=dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,projection=projection,seconds=time.perf_counter()-start,**projection_diagnostics(v,u,ze,metric));diagnostics.append(row)
                        assert row['negative_output_fraction']==0 and row['minimum_projection_inequality_slack']>-1e-7
                        assert row['dual_violation_relative']<1e-8 and row['complementarity_relative']<1e-8
                        key=name+'_'+projection;predictions[key]=u;arrays.update({key+'_members':members,key+'_weights':weights});kinds[key]=kind;projection_names[key]=projection
                        arrays['metric_'+projection]=metric
                np.savez_compressed(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.npz',**arrays)
                write(work.run/f'maps_s{source_seed}_t{target_seed}_{factor}.json',dict(source_seed=source_seed,target_seed=target_seed,factor=factor,input_kinds=kinds,state_projections=projection_names,fit_rows=0,scope='Frozen signed source allocation followed by explicit absolute-state projection; not target-native membership'))
                np.savez_compressed(work.run/f'predictions_s{source_seed}_t{target_seed}_{factor}.npz',source=ze,**predictions)
                write(work.run/'projection_diagnostics.json',dict(rows=diagnostics))
                for mask_name,mask in predefined_masks(budget,source_seed*100+(factor=='time')):
                    masks_record.append(dict(source_seed=source_seed,factor=factor,name=mask_name,source_members=support.tolist(),scales=mask.tolist()))
                    for consumer in cfg['consumers']:
                        change=lambda z:z[work.donors[factor]]-z if consumer=='contrast' else -z
                        for dose in cfg.get('mask_doses',{}).get(mask_name,cfg['doses']):
                            teacher=dose*component_update(change(ze),decoder,mask)
                            reference=measure(work,'source',factor,mask_name,consumer,dose,teacher,None,source_seed,target_seed)
                            for name,pred in predictions.items():
                                vectors=dose*component_update(change(pred),decoder,mask)
                                measure(work,name,factor,mask_name,consumer,dose,vectors,reference,source_seed,target_seed)
                                for role in ['temporal','quoted']:
                                    ids=np.array([i for i in work.evaluation_ids if work.rows[i].get('cue_role','temporal')==role],dtype=int)
                                    if len(ids):geometry.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,mask=mask_name,consumer=consumer,dose=dose,role=role,method=name,n=len(ids),error_sse=float(np.sum((vectors[ids]-teacher[ids])**2)),teacher_energy=float(np.sum(teacher[ids]**2))))
                            expected_rows+=len(work.evaluation_ids)*(1+len(predictions))
                work.progress('STATE_FACTOR_COMPLETE',source_seed=source_seed,target_seed=target_seed,factor=factor,methods=len(predictions),fit_rows=0)
        write(work.run/'geometry.json',dict(rows=geometry));write(work.run/'source_masks.json',dict(rows=masks_record));write(work.run/'projection_diagnostics.json',dict(rows=diagnostics))
        work.checks['all_rows']=len(work.metrics)==expected_rows
        work.checks['unique_rows']=len(work.metrics)==len({(r['source_seed'],r['target_seed'],r['factor'],r['mask'],r['consumer'],r['dose'],r['method'],r['row_id']) for r in work.metrics})
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
