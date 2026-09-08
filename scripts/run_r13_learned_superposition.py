"""Train actual superposition and SAEs; compare operation and latent truth."""
from __future__ import annotations
import argparse,datetime as dt,json,os,platform,sys,time,traceback
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
import torch
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ccad.toy_superposition import sample,train,ridge_fit,threshold_state,fit_rectified_state,ToyModel,SAEs
from ccad.complete_support import orthogonal_least_squares_support
from ccad.artifacts import sha256,validate_run_directory
from run_r011s1_raw_hook_asset import write_json as write,aggregate,entry

def now():return dt.datetime.now(dt.timezone.utc).isoformat()

def numpy_codes(model,sae,x):
    with torch.no_grad():
        h=x@model.w;a,z=sae.encode(h)
    return {k:v.detach().cpu().numpy().astype(np.float64) for k,v in dict(x=x,h=h,a=a,z=z).items()}

def evaluate(cfg,run,regime,base,model,sae,emit):
    arrays={};data={};fits=[];quality=[];pending=[]
    def generate(split):
        n=cfg['samples'][split]
        g=torch.Generator(device=cfg['device']).manual_seed(cfg['data_seeds'][split])
        source_regime='independent' if split=='shift' else regime
        data[split]=numpy_codes(model,sae,sample(n,cfg['features'],cfg['probability'],source_regime,g,cfg['device']))
        for k,v in data[split].items():arrays[split+'_'+k]=v
    for split in ['mean','discovery','calibration']:generate(split)
    w=model.w.detach().cpu().numpy().astype(np.float64);bias=model.bias.detach().cpu().numpy().astype(np.float64)
    dec=sae.decoder.detach().cpu().numpy().astype(np.float64)
    arrays.update(toy_w=w,toy_bias=bias,decoder=dec,encoder=sae.encoder.detach().cpu().numpy(),encoder_bias=sae.encoder_bias.detach().cpu().numpy(),decoder_bias=sae.decoder_bias.detach().cpu().numpy())
    def output(h):return np.maximum(h@w.T+bias,0)
    for si,source in enumerate(cfg['sae_seeds']):
        # Source-only grouping: nearest positive cosine to a known toy direction.
        # Every source atom is assigned once; all factors including empty retained.
        sim=dec[si]@w.T/np.maximum(np.linalg.norm(w,axis=1),1e-12)[None,:]
        labels=sim.argmax(1)
        for factor in range(cfg['features']):
            sg=np.flatnonzero(labels==factor)
            empty=not len(sg)
            ds=dec[si,sg];yd=data['discovery']['z'][:,si][:,sg];ym=data['mean']['z'][:,si][:,sg].mean(0)
            ad=data['discovery']['a'][:,si][:,sg];am=data['mean']['a'][:,si][:,sg].mean(0)
            for ti,target in enumerate(cfg['sae_seeds']):
                if si==ti:continue
                xd=data['discovery']['z'][:,ti];xm=data['mean']['z'][:,ti].mean(0)
                models={};budget=cfg['target_budget']
                for space,y,mu in [('code',yd,ym),('preactivation',ad,am)]:
                    # Output-channel metric corresponds to source physical updates.
                    support,path=orthogonal_least_squares_support(xd-xm,(y-mu)@ds,budget)
                    full=np.arange(cfg['sae_width']);fullcoef=ridge_fit(xd,y,xm,mu,cfg['ridge_fraction'])
                    score=np.linalg.norm((fullcoef@ds),axis=1)*np.std(xd,axis=0)
                    dense=np.argsort(-score,kind='stable')[:budget]
                    rng=np.random.default_rng(37000+source*1000+target*50+factor)
                    random=rng.choice(cfg['sae_width'],budget,replace=False)
                    candidates={'ols':support,'dense':dense,'full':full,'random':random}
                    if space=='code':
                        # Strong single dynamic atom; same all-source outputs.
                        candidates['atom1']=orthogonal_least_squares_support(xd-xm,(y-mu)@ds,1)[0]
                    for selection,idx in candidates.items():
                        # Keep every factor and every method, even with a zero
                        # teacher operation: semantic truth still has an error.
                        if empty:idx=np.array([],dtype=np.int64)
                        coef=ridge_fit(xd[:,idx],y,xm[idx],mu,cfg['ridge_fraction'])
                        name=selection+'_'+space
                        models[name]=dict(indices=idx,coef=coef,mu=mu,space=space)
                        if selection=='ols' and space=='code':models['ols_code_clip']=dict(indices=idx,coef=coef,mu=mu,space='clip')
                if cfg.get('rectified_fit'):
                    for selection in ['ols','dense','full','atom1','random']:
                        original=models[selection+'_code'];idx=original['indices']
                        free_x=xd[:,idx].mean(0);free_y=yd.mean(0)
                        free_coef=ridge_fit(xd[:,idx],yd,free_x,free_y,cfg['ridge_fraction'])
                        free_intercept=free_y+(xm[idx]-free_x)@free_coef
                        models[selection+'_affine_clip']=dict(indices=idx,coef=free_coef,mu=free_intercept,space='clip')
                        if empty:
                            coef,intercept,info=original['coef'],ym,dict(converged=True,message='Empty source; exact zero operation',iterations=0,initial_loss=0,final_loss=0)
                        else:coef,intercept,info=fit_rectified_state(xd[:,idx],yd,ds,original['coef'],xm[idx],ym,cfg['rectified_maxiter'])
                        models[selection+'_rectified']=dict(indices=idx,coef=coef,mu=intercept,space='clip',optimization=info)
                hd=data['discovery']['h'];hm=data['mean']['h'].mean(0)
                rawcoef=ridge_fit(hd,yd,hm,ym,cfg['ridge_fraction'])
                identity=dict(regime=regime,toy_seed=base,source_seed=source,target_seed=target,factor=factor,source_group=sg.tolist())
                fits.append(dict(**identity,status='EMPTY_SOURCE_GROUP' if empty else 'FIT',source_cosines=sim[sg,factor].tolist(),target_mean=xm.tolist(),source_code_mean=ym.tolist(),
                                 source_preactivation_mean=am.tolist(),raw_coef=rawcoef.tolist(),raw_mean=hm.tolist(),
                                 models={k:{q:(v.tolist() if isinstance(v,np.ndarray) else v) for q,v in m.items()} for k,m in models.items()}))
                pending.append((si,ti,sg,ds,xm,ym,hm,rawcoef,models,identity))
    # All actual maps/source groups for this trained material are fixed before
    # evaluation or shift arrays are generated, encoded, or scored.
    frozen_path=run/f'{regime}_{base}_frozen_maps.json'
    write(frozen_path,dict(frozen_at_utc=now(),fits=fits,evaluation_arrays_generated=False,config_sha256=sha256(run/'config.resolved.json')))
    for split in ['evaluation','shift']:generate(split)
    np.savez_compressed(run/f'{regime}_{base}_data.npz',**arrays)
    for split in ['calibration','evaluation','shift']:
        d=data[split];out=output(d['h']);contrast=d['x'][:,0]-d['x'][:,1];den=float(np.sum(contrast**2))
        quality.append(dict(regime=regime,toy_seed=base,split=split,toy_mse=float(np.mean((out-d['x'])**2)),
                            pair_difference_numerator=float(np.sum((out[:,0]-out[:,1]-contrast)**2)),pair_difference_denominator=den,
                            pair_difference_normalized_error=float(np.sum((out[:,0]-out[:,1]-contrast)**2)/den) if den else None,
                            normalized_pair_separation=float(np.linalg.norm(w[0]-w[1])/max(np.linalg.norm(w[0]+w[1]),1e-12)),full_w_norms=np.linalg.norm(w,axis=1).tolist()))
    for si,ti,sg,ds,xm,ym,hm,rawcoef,models,identity in pending:
        factor=identity['factor'];source=identity['source_seed'];target=identity['target_seed'];per_pair={}
        keep=cfg.get('retain_all_predictions',False) or [source,target,factor] in cfg.get('prediction_examples',[[1,2,0],[1,2,2]])
        for split in ['calibration','evaluation','shift']:
            d=data[split];xt=d['z'][:,ti];truth=-d['x'][:,factor,None]*w[factor];actual=-d['z'][:,si][:,sg]@ds
            source_out=output(d['h']+actual);ideal_out=output(d['h']+truth);base_out=output(d['h'])
            predictions={}
            for name,m in models.items():
                state=(xt[:,m['indices']]-xm[m['indices']])@m['coef']+m['mu']
                if m['space'] in ['preactivation','clip']:state=np.maximum(state,0)
                predictions[name]=-state@ds
            predictions['raw_linear']=-((d['h']-hm)@rawcoef+ym)@ds
            predictions['source_encoder_oracle']=actual.copy()
            predictions['factor_truth_oracle']=truth.copy()
            reconstructed=xt@dec[ti]+arrays['decoder_bias'][ti]
            state=np.maximum((reconstructed-arrays['decoder_bias'][si])@arrays['encoder'][si]+arrays['encoder_bias'][si],0)
            predictions['target_full_reencode']=-state[:,sg]@ds
            for method,delta in predictions.items():
                predout=output(d['h']+delta)
                values=dict(operation_error=np.sum((delta-actual)**2,axis=1),operation_energy=np.sum(actual**2,axis=1),
                            truth_error=np.sum((delta-truth)**2,axis=1),truth_energy=np.sum(truth**2,axis=1),
                            function_error=np.sum((predout-source_out)**2,axis=1),function_energy=np.sum((source_out-base_out)**2,axis=1),
                            truth_function_error=np.sum((predout-ideal_out)**2,axis=1),truth_function_energy=np.sum((ideal_out-base_out)**2,axis=1))
                for block,start in enumerate(range(0,len(xt),cfg['block_size'])):
                    stop=min(start+cfg['block_size'],len(xt));sl=slice(start,stop)
                    emit(dict(**identity,split=split,method=method,block=block,samples=stop-start,
                              target_members=len(models[method]['indices']) if method in models else (cfg['hidden'] if method=='raw_linear' else cfg['sae_width'] if method=='target_full_reencode' else 0),
                              **{k:float(v[sl].sum()) for k,v in values.items()}))
                if keep and split!='calibration':per_pair[split+'_'+method+'_delta']=delta.astype(np.float32)
            if keep and split!='calibration':per_pair[split+'_source_delta']=actual.astype(np.float32);per_pair[split+'_truth_delta']=truth.astype(np.float32)
        if keep:np.savez_compressed(run/f'{regime}_{base}_s{source}_t{target}_f{factor}.npz',**per_pair)
    write(run/f'{regime}_{base}_fits.json',dict(fits=fits,quality=quality))
    return fits,quality

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2)
    write(run/'config.resolved.json',cfg);code=[]
    for rel in ['scripts/run_r13_learned_superposition.py','src/ccad/toy_superposition.py','src/ccad/complete_support.py','src/ccad/pair_complete_correspondence.py','src/ccad/artifacts.py','scripts/run_r011s1_raw_hook_asset.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    input_list=[entry(a.config,'Prospective learned toy configuration','config'),entry(ROOT/'scripts/licenses/anthropic_toy_models_MIT.txt','Anthropic official repository commit562710e','license')]
    if cfg.get('load_training_run'):
        for regime in cfg['regimes']:
            for base in cfg['toy_seeds']:input_list.append(entry(ROOT/'runs'/cfg['load_training_run']/f'{regime}_{base}_weights.pt','Retained project trained toy and SAE state','trained_model'))
    write(run/'inputs.json',dict(inputs=input_list))
    write(run/'manifest.json',dict(schema_version='fcc.learned.superposition.v1',run_id=run.name,run_parent='INSERTED_R13',purpose=cfg['purpose'],milestone='M4',evidence_level=f"trained_toy_and_{len(cfg['sae_seeds'])}_controlled_SAEs",started_utc=now(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=cfg['audit_opened'],candidate_family_frozen=cfg['frozen'],mean_constants_source_split='independent generated mean; finite-fit intercept is a fitted parameter, affine controls matched',threshold_source_split='source trained ReLU; no target evaluation labels',statistics_unit='generated blocks; shared source/target SAE and base toy model dependencies',device=cfg['device'],seeds=cfg['sae_seeds'],resource_lease=os.environ.get('CCAD_RESOURCE_LEASE_ID','resource-manager gpu-0 wrapper'),resource_lease_reason='Toy compressor and SAE training; bounded 5400s round budget'))
    write(run/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pytorch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),transformers='not_imported',sae='Controlled unit-decoder ReLU L1',platform=platform.platform()))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now()));error=None;count=0;traces=[];qualities=[];totals={}
    raw=(run/'metrics.raw.jsonl').open('w',encoding='utf8');stdout=(run/'stdout.log').open('w',encoding='utf8')
    def progress(row):
        traces.append(row);stdout.write(json.dumps(row)+'\n');stdout.flush()
        if row['stage']=='toy' or row.get('sae_seed')==cfg['sae_seeds'][0]:print(json.dumps(row),flush=True)
    def emit(row):
        nonlocal count
        raw.write(json.dumps(row,separators=(',',':'))+'\n');count+=1
        key=tuple(row[k] for k in ['regime','toy_seed','source_seed','target_seed','split','method'])
        item=totals.setdefault(key,{k:row[k] for k in ['regime','toy_seed','source_seed','target_seed','split','method']})
        for k in ['samples','operation_error','operation_energy','truth_error','truth_energy','function_error','function_energy','truth_function_error','truth_function_energy']:item[k]=item.get(k,0)+row[k]
    try:
        for regime in cfg['regimes']:
            for base in cfg['toy_seeds']:
                if cfg.get('load_training_run'):
                    saved=torch.load(ROOT/'runs'/cfg['load_training_run']/f'{regime}_{base}_weights.pt',map_location=cfg['device'],weights_only=True)
                    if cfg.get('retrain_sae'):model,sae,_=train(cfg,regime,base,progress,toy_state=saved['toy'])
                    else:
                        model=ToyModel(cfg['features'],cfg['hidden'],base,cfg['device']);model.load_state_dict(saved['toy']);model.requires_grad_(False)
                        sae=SAEs(cfg['hidden'],cfg['sae_width'],cfg['sae_seeds'],cfg['device']);sae.load_state_dict(saved['saes']);sae.requires_grad_(False)
                else:model,sae,_=train(cfg,regime,base,progress)
                torch.save(dict(toy=model.state_dict(),saes=sae.state_dict(),config=cfg,regime=regime,toy_seed=base),run/f'{regime}_{base}_weights.pt')
                _,q=evaluate(cfg,run,regime,base,model,sae,emit);qualities+=q;raw.flush()
                print(json.dumps(dict(stage='regime_complete',regime=regime,toy_seed=base,rows=count,wall_seconds=time.perf_counter()-start)),flush=True)
    except Exception:error=traceback.format_exc()
    raw.close();stdout.close();write(run/'training_traces.json',dict(traces=traces));write(run/'quality.json',dict(quality=qualities))
    gen='scripts/run_r13_learned_superposition.py'
    summary=dict(rows=count,wall_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path=gen,generator_script_sha256=next(c['sha256'] for c in code if c['path']==gen),aggregates=list(totals.values()))
    write(run/'metrics.summary.json',summary);(run/'stderr.log').write_text(error or '',encoding='utf8');write(run/'status.json',dict(status='FAIL' if error else 'PASS',error=error,updated_utc=now()))
    valid=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=valid.ok,errors=list(valid.errors)))
    print(json.dumps(dict(run=run.name,error=error,contract_ok=valid.ok,rows=count,wall_seconds=summary['wall_seconds'])),flush=True)
    return int(bool(error) or not valid.ok)
if __name__=='__main__':raise SystemExit(main())
