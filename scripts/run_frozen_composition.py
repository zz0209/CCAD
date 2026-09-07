"""Evaluate frozen R3 source members, maps and controls on untouched material.

This consumer imports no fitting or support-selection function. All numerical
maps, direct-target supports, scalar gains and DAS vectors predate this panel.
"""
import argparse,json,traceback
from pathlib import Path
from composition_runtime import CompositionRun,ROOT,write,np
from ccad.artifacts import sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    work=CompositionRun(args.config,'scripts/run_frozen_composition.py',[]);error=None
    try:
        cfg=work.cfg;freeze=json.loads(work.checked(ROOT/cfg['frozen_dir']/'FREEZE.json').read_text())
        work.checks['frozen_inputs_exact']=all(sha256(ROOT/r['path'])==r['sha256'] for r in freeze['files'])
        if not work.checks['frozen_inputs_exact']:raise ValueError('Frozen input changed')
        work.load();torch=work.torch;from_sae={}
        from sparsify import SparseCoder
        for spec in cfg['sae_checkpoints']:
            seed=spec['seed'];path=Path(spec['path']);work.checked(path/'sae.safetensors');work.checked(path/'cfg.json');sae=SparseCoder.load_from_disk(path,device='cuda').eval();decoder=sae.W_dec.detach().cpu().numpy().astype(np.float64)
            with torch.no_grad():
                acts,inds,_=sae.encode(torch.as_tensor(work.h[:,:2].reshape(-1,work.h.shape[-1]),device='cuda'));z=torch.zeros((len(acts),sae.num_latents),device='cuda').scatter_(1,inds,acts).cpu().numpy().reshape(work.n,2,-1)
            del sae;np.savez_compressed(work.run/f'seed{seed}_codes.npz',time_z=z[:,0],number_z=z[:,1]);coords=np.load(work.checked(ROOT/cfg['source_run']/f'seed{seed}_source_coordinates.npz'));q=np.zeros_like(work.h,dtype=np.float64);factors={}
            for factor,slot in [('number',1),('time',0)]:
                dz=z[work.donors[factor],slot].astype(np.float64)-z[:,slot];budget=cfg['source_budgets'][factor];support=coords[f'{factor}_{budget}_support'];basis=coords[f'{factor}_{budget}_basis'];coef=coords[f'{factor}_{budget}_coefficients'];q[:,slot]=dz[:,support]@coef@basis.T
                native=dz[:,support]@decoder[support];work.checks[f'source_reconstruction_{seed}_{factor}']=np.allclose(q[:,slot],native,rtol=1e-8,atol=1e-8);factors[factor]=dict(dz=dz,basis=basis,support=support)
            from_sae[seed]=dict(decoder=decoder,q=q,**factors);work.progress('ENCODED',seed=seed)
        for source_seed,s in from_sae.items():
            teachers={}
            for factor,slot in [('number',1),('time',0),('joint',None)]:
                q=s['q'].copy()
                if slot is not None:q[:,1-slot]=0
                teachers[factor]=work.measure('source_teacher',factor,q,source_seed=source_seed)
            endpoint=[]
            for factor,lp in teachers.items():
                for off,i in enumerate(work.evaluation_ids):
                    ref=lp[off];endpoint.append(dict(source_seed=source_seed,factor=factor,row_id=int(i),teacher_label=int(np.argmax(ref[work.labels])),teacher_label_logprobs=ref[work.labels].tolist(),teacher_noop_kl=max(0.,float(np.sum(np.exp(ref)*(ref-work.base[i]))))))
            write(work.run/f'source{source_seed}_endpoint.json',dict(rows=endpoint))
            if cfg.get('simple_source_controls'):
                null=np.load(work.checked(ROOT/cfg['correspondence_run']/f'simple_source{source_seed}.npz'))
                meta=json.loads(work.checked(ROOT/cfg['correspondence_run']/f'simple_source{source_seed}.json').read_text())
                old_cues={tuple(pair):i for i,pair in enumerate(meta['cue_pairs'])};new_cfg=json.loads(work.checked(ROOT/cfg['material_run']/'config.resolved.json').read_text())
                constant=np.zeros_like(work.h,dtype=np.float64);by_cue=np.zeros_like(constant)
                for factor,slot in [('number',1),('time',0)]:
                    direction=np.array([1 if work.rows[i]['number' if factor=='number' else 'past'] else -1 for i in work.donors[factor]])
                    constant[:,slot]=direction[:,None]*null[factor+'_global']
                    for cue_id,pair in enumerate(new_cfg['time_cues']):
                        mask=np.array([r['cue_id']==cue_id for r in work.rows]);old_id=old_cues.get(tuple(pair));mean=null[factor+'_global'] if old_id is None else null[factor+f'_cue{old_id}']
                        by_cue[mask,slot]=direction[mask,None]*mean
                for name,all_delta in [('global_factor_mean',constant),('cue_factor_mean',by_cue)]:
                    for factor,slot in [('number',1),('time',0),('joint',None)]:
                        delta=all_delta.copy()
                        if slot is not None:delta[:,1-slot]=0
                        work.measure(name,factor,delta,teachers[factor],source_seed=source_seed)
            das=np.zeros_like(work.h,dtype=np.float64);dasspec=json.loads(work.checked(ROOT/cfg['behavior_run']/f'das_source{source_seed}.json').read_text())
            for factor,slot in [('number',1),('time',0)]:
                a=np.array(next(r['direction'] for r in dasspec['factors'] if r['factor']==factor));raw=(work.h[work.donors[factor]]-work.h)[:,slot];das[:,slot]=(raw@a)[:,None]*a
            for factor,slot in [('number',1),('time',0),('joint',None)]:
                q=das.copy()
                if slot is not None:q[:,1-slot]=0
                work.measure('das_style_raw_rank1',factor,q,teachers[factor],source_seed=source_seed)
            rawmaps=np.load(work.checked(ROOT/cfg['frozen_dir']/f'raw_source{source_seed}.npz'))
            for target_seed,t in from_sae.items():
                if source_seed==target_seed:continue
                maps=np.load(work.checked(ROOT/cfg['correspondence_run']/f'maps_s{source_seed}_t{target_seed}.npz'));old=json.loads(work.checked(ROOT/cfg['correspondence_run']/f'pair_s{source_seed}_t{target_seed}.json').read_text());beh=json.loads(work.checked(ROOT/cfg['behavior_run']/f'pair_s{source_seed}_t{target_seed}.json').read_text());changes={};diagnostics=[]
                for factor,slot in [('number',1),('time',0)]:
                    dz=t[factor]['dz'];basis=s[factor]['basis'];raw=(work.h[work.donors[factor]]-work.h)[:,slot];physical={}
                    prefix=factor+'_';suffix='_coefficients'
                    for key in maps.files:
                        if key.startswith(prefix) and key.endswith(suffix):physical[key[len(prefix):-len(suffix)]]=dz@maps[key].astype(np.float64)@basis.T
                    for name in ['raw_native_units','raw_feature_rms']:physical[name]=raw@rawmaps[f'{factor}_{name}']@basis.T
                    members=maps[factor+'_fcc_members'];physical['same_members_native']=dz[:,members]@t['decoder'][members];physical['same_members_projected']=physical['same_members_native']@basis@basis.T;physical['direct_target_native']=t['q'][:,slot];physical['full_native']=dz@t['decoder']
                    for name in ['same_members_native','direct_target_native']:
                        gain=next(r['gain'] for r in old['diagnostics'] if r['factor']==factor and r['method']==name+'_gain');physical[name+'_gain']=physical[name]*gain
                    for name,base in [('same_members_behavior_gain','same_members_native'),('direct_target_behavior_gain','direct_target_native')]:
                        gain=next(r['gain'] for r in beh['gain_fit'] if r['factor']==factor and r['method']==name);physical[name]=physical[base]*gain
                    physical['wrong_factor']=t['q'][:,1-slot]
                    for name,q in physical.items():
                        changes.setdefault(name,np.zeros_like(work.h,dtype=np.float64))[:,slot]=q
                        for template in dict.fromkeys(r['template'] for r in work.rows):
                            ix=np.array([r['template']==template for r in work.rows]);teacher=s['q'][:,slot];diagnostics.append(dict(source_seed=source_seed,target_seed=target_seed,factor=factor,method=name,template=template,n=int(ix.sum()),squared_error_sum=float(np.sum((q[ix]-teacher[ix])**2)),teacher_energy_sum=float(np.sum(teacher[ix]**2)),candidate_energy_sum=float(np.sum(q[ix]**2))))
                for name,delta in changes.items():
                    for factor,slot in [('number',1),('time',0),('joint',None)]:
                        if slot is not None and name not in cfg['single_factor_methods']+['same_members_behavior_gain','direct_target_behavior_gain']:continue
                        q=delta.copy()
                        if slot is not None:q[:,1-slot]=0
                        work.measure(name,factor,q,teachers[factor],source_seed=source_seed,target_seed=target_seed)
                write(work.run/f'pair_s{source_seed}_t{target_seed}.json',dict(diagnostics=diagnostics,frozen_map_hash=sha256(ROOT/cfg['correspondence_run']/f'maps_s{source_seed}_t{target_seed}.npz'),fit_rows=0,joint_refit=False));work.progress('PAIR_COMPLETE',source_seed=source_seed,target_seed=target_seed,methods=len(changes))
        work.checks['unique']=len(work.metrics)==len({(r['source_seed'],r.get('target_seed'),r['factor'],r['method'],r['row_id']) for r in work.metrics})
        work.checks['all_rows']=len(work.metrics)==work.n*(5*(12 if cfg.get('simple_source_controls') else 6)+20*(len(changes)+2*(len(cfg['single_factor_methods'])+2)))
        work.checks['all_frozen_inputs_still_exact']=all(sha256(ROOT/r['path'])==r['sha256'] for r in freeze['files'])
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(work.run/'stderr.log').write_text(traceback.format_exc())
    return work.finish(error)


if __name__=='__main__':raise SystemExit(main())
