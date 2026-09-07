"""Five-seed source-fixed correspondences and unrefitted two-factor composition."""
import argparse,json,traceback
from pathlib import Path
from composition_runtime import CompositionRun, ROOT, write, np
from ccad.factor_correspondence import fit_composition_controls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_composition_correspondence.py',['src/ccad/factor_correspondence.py','src/ccad/nip_baselines.py','src/ccad/proposal.py']);error=None
    try:
        work.load();cfg=work.cfg;torch=work.torch
        fit=work.discovery&np.array([r['block'] in cfg['fit_blocks'] for r in work.rows]);cal=work.discovery&~fit
        from sparsify import SparseCoder
        assets={}
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];path=Path(spec['path']);work.checked(path/'sae.safetensors');work.checked(path/'cfg.json');sae=SparseCoder.load_from_disk(path,device='cpu').eval();decoder=sae.W_dec.detach().numpy().astype(np.float64);del sae
            codes=np.load(work.checked(ROOT/cfg['material_run']/f'seed{seed}_codes.npz'));coords=np.load(work.checked(ROOT/cfg['source_run']/f'seed{seed}_source_coordinates.npz'));factors={}
            for factor,slot in [('number',1),('time',0)]:
                z=codes[factor+'_z'].astype(np.float64);dz=z[work.donors[factor]]-z;budget=cfg['source_budgets'][factor]
                source={key:coords[f'{factor}_{budget}_{key}'] for key in ['support','basis','coefficients','coordinates']}
                factors[factor]=dict(dz=dz,source=source)
            assets[seed]=dict(decoder=decoder,**factors)
        work.progress('ASSETS_READY')
        diagnostics=[]; method_count=None
        for source_seed,s in assets.items():
            teachers={};teacher_delta=np.zeros_like(work.h,dtype=np.float64)
            for factor,slot in [('number',1),('time',0)]:
                q=s[factor]['source']['coordinates']@s[factor]['source']['basis'].T;teacher_delta[:,slot]=q
                delta=np.zeros_like(teacher_delta);delta[:,slot]=q;teachers[factor]=work.measure('source_teacher',factor,delta,source_seed=source_seed)
            teachers['joint']=work.measure('source_teacher','joint',teacher_delta,source_seed=source_seed)
            if cfg.get('simple_source_controls'):
                constant=np.zeros_like(teacher_delta);by_cue=np.zeros_like(teacher_delta);null_arrays={}
                for factor,slot in [('number',1),('time',0)]:
                    direction=np.array([1 if work.rows[i]['number' if factor=='number' else 'past'] else -1 for i in work.donors[factor]])
                    aligned=teacher_delta[:,slot]*direction[:,None];mean=aligned[work.discovery].mean(0)
                    constant[:,slot]=direction[:,None]*mean;null_arrays[factor+'_global']=mean
                    for cue_id in range(len(cfg['time_cues'])):
                        mask=np.array([r['cue_id']==cue_id for r in work.rows]);selected=mask&work.discovery
                        cue_mean=aligned[selected].mean(0) if selected.any() else mean
                        by_cue[mask,slot]=direction[mask,None]*cue_mean;null_arrays[factor+f'_cue{cue_id}']=cue_mean
                np.savez_compressed(work.run/f'simple_source{source_seed}.npz',**null_arrays)
                write(work.run/f'simple_source{source_seed}.json',dict(cue_pairs=cfg['time_cues'],rule='Mean source edit aligned to reciprocal donor direction; cue-specific mean when cue spelling was in discovery, otherwise global mean. Source-only, generator direction/cue metadata are explicit extra information.'))
                for name,all_delta in [('global_factor_mean',constant),('cue_factor_mean',by_cue)]:
                    for factor,slot in [('number',1),('time',0),('joint',None)]:
                        delta=all_delta.copy()
                        if slot is not None:delta[:,1-slot]=0
                        work.measure(name,factor,delta,teachers[factor],source_seed=source_seed)
            for target_seed,t in assets.items():
                if source_seed==target_seed:continue
                changes={}; map_arrays={}; pair_diag=[]
                for factor,slot in [('number',1),('time',0)]:
                    source=s[factor]['source'];xt=t[factor]['dz'];xs=s[factor]['dz'];xraw=(work.h[work.donors[factor]]-work.h)[:,slot]
                    levels=np.array([1 if work.rows[i]['number' if factor=='number' else 'past'] else -1 for i in work.donors[factor]])
                    out,diag=fit_composition_controls(xt,xraw,xs,source,fit,cal,work.discovery,levels,cfg['target_budgets'][factor],cfg['alphas'],1000*source_seed+10*target_seed+slot)
                    members=np.array(diag['fcc_group']['members']);native=xt[:,members]@t['decoder'][members];projected=native@source['basis']@source['basis'].T
                    full_native=xt@t['decoder'];basis=source['basis'];basis_q=source['coordinates']@basis.T
                    physical={name:pred@basis.T for name,(pred,w) in out.items()}
                    physical.update(same_members_native=native,same_members_projected=projected,direct_target_native=t[factor]['source']['coordinates']@t[factor]['source']['basis'].T,full_native=full_native)
                    for name in ['same_members_native','direct_target_native']:
                        q=physical[name];gain=float(np.sum(q[work.discovery]*basis_q[work.discovery])/max(np.sum(q[work.discovery]**2),1e-12))
                        physical[name+'_gain']=gain*q;diag[name+'_gain']=dict(gain=gain,gain_source='discovery least squares to frozen source hook delta')
                    # The other factor is deliberately applied at the wrong site;
                    # this is an explicit wrong-query control, not a role proof.
                    other='time' if factor=='number' else 'number';physical['wrong_factor']=t[other]['source']['coordinates']@t[other]['source']['basis'].T
                    for name,q in physical.items():
                        if name not in changes:changes[name]=np.zeros_like(work.h,dtype=np.float64)
                        changes[name][:,slot]=q
                        vector_rows=[]
                        for part,mask in [('discovery',work.discovery),('new_lex_known_cue',np.array([r['block'] not in cfg['discovery_blocks'] and r['cue_id'] in cfg['discovery_cues'] for r in work.rows])),('new_lex_new_cue',np.array([r['block'] not in cfg['discovery_blocks'] and r['cue_id'] not in cfg['discovery_cues'] for r in work.rows])),('known_lex_new_cue',np.array([r['block'] in cfg['discovery_blocks'] and r['cue_id'] not in cfg['discovery_cues'] for r in work.rows]))]:
                            vector_rows.append(dict(partition=part,n=int(mask.sum()),squared_error_sum=float(np.sum((q[mask]-basis_q[mask])**2)),teacher_energy_sum=float(np.sum(basis_q[mask]**2)),candidate_energy_sum=float(np.sum(q[mask]**2))))
                        pair_diag.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,vector=vector_rows,**diag.get(name,{})))
                    for name,(pred,w) in out.items():
                        if w is not None:map_arrays[f'{factor}_{name}_coefficients']=w.astype(np.float32)
                    map_arrays[factor+'_fcc_members']=members;map_arrays[factor+'_source_basis']=basis.astype(np.float32)
                np.savez_compressed(work.run/f'maps_s{source_seed}_t{target_seed}.npz',**map_arrays)
                for name,delta in changes.items():
                    for factor,slot in [('number',1),('time',0),('joint',None)]:
                        if factor!='joint' and name not in cfg['single_factor_methods']:continue
                        single=delta.copy()
                        if slot is not None:single[:,1-slot]=0
                        values=work.measure(name,factor,single,teachers[factor],source_seed=source_seed,target_seed=target_seed)
                        # Augment raw rows with the teacher/noop denominator using
                        # a separate compact per-pair endpoint file below.
                endpoint=[]
                for factor,teacher in teachers.items():
                    for offset,i in enumerate(work.evaluation_ids):
                        small=teacher[offset,work.labels];endpoint.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,row_id=int(i),teacher_label=int(np.argmax(small)),teacher_label_logprobs=small.tolist(),teacher_noop_kl=max(0.,float(np.sum(np.exp(teacher[offset])*(teacher[offset]-work.base[i]))))))
                write(work.run/f'pair_s{source_seed}_t{target_seed}.json',dict(diagnostics=pair_diag,endpoint=endpoint));diagnostics.extend(pair_diag);method_count=len(changes)
                work.progress('PAIR_COMPLETE',source_seed=source_seed,target_seed=target_seed,methods=method_count)
        work.checks['all_rows']=len(work.metrics)==len(work.evaluation_ids)*((9 if cfg.get('simple_source_controls') else 3)*len(assets)+len(assets)*(len(assets)-1)*(method_count+2*len(cfg['single_factor_methods'])))
        work.checks['unique']=len(work.metrics)==len({(r['source_seed'],r.get('target_seed'),r['factor'],r['method'],r['row_id']) for r in work.metrics})
        write(work.run/'fit_diagnostics.json',dict(rows=diagnostics,fit_row_ids=np.flatnonzero(fit).tolist(),calibration_row_ids=np.flatnonzero(cal).tolist(),discovery_row_ids=np.flatnonzero(work.discovery).tolist(),joint_refit=False))
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
