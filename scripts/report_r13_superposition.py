"""Build source-backed scientific readout and figure inputs for learned toys."""
import datetime as dt,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/inserted_superposition_20260907'

def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    summaries={label:read(BASE/f'CONFIRM_{label}_SUMMARY.json') for label in ['LOW','HIGH']}
    cfg=summaries['LOW']['config'];material=[];quality=[];geometry=[];atomic=[];identity_checks=[];examples=[]
    runpaths={label:ROOT/'runs'/summaries[label]['run_id'] for label in summaries}
    traces={label:read(p/'training_traces.json')['traces'] for label,p in runpaths.items()}
    for regime in cfg['regimes']:
        for split in ['evaluation','shift']:
            vals={label:next(r for r in s['pooled'] if (r['regime'],r['split'],r['family'],r['method'])==(regime,split,'all','source_encoder_oracle')) for label,s in summaries.items()}
            per_seed=[]
            for base in cfg['toy_seeds']:
                for seed in cfg['sae_seeds']:
                    q={label:next(r for r in s['source_quality'] if (r['regime'],r['split'],r['toy_seed'],r['sae_seed'])==(regime,split,base,seed)) for label,s in summaries.items()}
                    per_seed.append(dict(toy_seed=base,sae_seed=seed,low_error=q['LOW']['truth_nmse'],high_error=q['HIGH']['truth_nmse'],relative_reduction=1-q['HIGH']['truth_nmse']/q['LOW']['truth_nmse']))
            material.append(dict(regime=regime,split=split,low_truth_nmse=vals['LOW']['truth_nmse'],high_truth_nmse=vals['HIGH']['truth_nmse'],relative_reduction=1-vals['HIGH']['truth_nmse']/vals['LOW']['truth_nmse'],per_seed=per_seed,seeds_better=sum(r['relative_reduction']>0 for r in per_seed)))
        for label,s in summaries.items():
            final=[r for r in traces[label] if r['stage']=='sae' and r['step']==cfg['sae_steps'] and r['regime']==regime]
            quality.append(dict(material=label,regime=regime,saes=len(final),**{key:dict(mean=float(np.mean([r[key] for r in final])),minimum=min(r[key] for r in final),maximum=max(r[key] for r in final)) for key in ['fve','l0','alive','source_factor_error']},points=final))
        for base in cfg['toy_seeds']:
            arrays={}
            for label,path in runpaths.items():
                with np.load(path/f'{regime}_{base}_data.npz') as packed:arrays[label]={k:packed[k] for k in packed.files}
            low=arrays['LOW'];high=arrays['HIGH'];checks={k:bool(np.array_equal(low[k],high[k])) for k in ['toy_w','toy_bias']+[split+'_'+key for split in cfg['samples'] for key in ['x','h']]}
            identity_checks.append(dict(regime=regime,toy_seed=base,checks=checks,all_identical=all(checks.values())))
            w=low['toy_w'];norm=np.linalg.norm(w,axis=1);gram=w@w.T/norm[:,None]/norm[None,:]
            q=read(runpaths['LOW']/f'{regime}_{base}_fits.json')['quality'];shift=next(r for r in q if r['split']=='shift')
            geometry.append(dict(regime=regime,toy_seed=base,gram=gram.tolist(),w=w.tolist(),w_norm=norm.tolist(),pair_cosine=float(gram[0,1]),pair_separation=shift['normalized_pair_separation'],shift_pair_difference_nmse=shift['pair_difference_normalized_error'],quality=q))
            for label,arr in arrays.items():
                dec=arr['decoder'];dec=dec/np.linalg.norm(dec,axis=2,keepdims=True)
                for si,source in enumerate(cfg['sae_seeds']):
                    for ti,target in enumerate(cfg['sae_seeds']):
                        if si>=ti:continue
                        cosine=dec[si]@dec[ti].T;i,j=linear_sum_assignment(-cosine);ia,ja=linear_sum_assignment(-np.abs(cosine))
                        atomic.append(dict(material=label,regime=regime,toy_seed=base,source_seed=source,target_seed=target,signed_mcc=float(cosine[i,j].mean()),absolute_mcc=float(np.abs(cosine[ia,ja]).mean()),all_24_columns=True))
            if regime=='independent' and base==202:
                arr=high;collection=read(BASE/'operations/l1_015/independent_202_maps.json');op=next(r for r in collection['operations'] if (r['source_seed'],r['target_seed'],r['factor'])==(1,2,2))
                fits=read(runpaths['HIGH']/'independent_202_frozen_maps.json')['fits'];fit=next(r for r in fits if (r['source_seed'],r['target_seed'],r['factor'])==(1,2,2))
                replay=read(BASE/'operations/l1_015/REPLAY.json')['examples'][0];ids=replay['ids'];labels=replay['labels'];sg=fit['source_group'];ds=arr['decoder'][0,sg]
                for index,label in zip(ids,labels):
                    h=arr['evaluation_h'][index];x=arr['evaluation_x'][index];zt=arr['evaluation_z'][index,1];source_state=arr['evaluation_z'][index,0,sg]
                    pred=dict(source=-source_state@ds,factor_truth=-x[2]*w[2]);states={};preacts={}
                    for method in ['ols_affine_clip','ols_rectified','full_rectified']:
                        m=fit['models'][method];idx=m['indices'];pre=(zt[idx]-np.array(fit['target_mean'])[idx])@np.array(m['coef'])+m['mu'];state=np.maximum(pre,0)
                        preacts[method]=pre.tolist();states[method]=state.tolist();pred[method]=-state@ds
                    before=np.maximum(h@w.T+arr['toy_bias'],0)
                    examples.append(dict(label=label,evaluation_row=index,latent=x.tolist(),active_factors=np.flatnonzero(x).tolist(),source_members=sg,source_state=source_state.tolist(),target_members=op['target_members'],target_codes=zt[op['target_members']].tolist(),source_decoder=ds.tolist(),coefficient=op['coefficient'],target_mean=op['target_mean'],intercept=op['intercept'],states=states,preactivations=preacts,delta={k:v.tolist() for k,v in pred.items()},output_changes={k:(np.maximum((h+v)@w.T+arr['toy_bias'],0)-before).tolist() for k,v in pred.items()},original_output=before.tolist()))
    assert all(r['all_identical'] for r in identity_checks)
    freeze=read(BASE/'FREEZE.json');frozen_checks=[]
    for label,path in runpaths.items():
        hashes=read(path/'code_hashes.json')['files']
        for file in freeze['files']:
            observed=next((r for r in hashes if r['path']==file['path']),None)
            if observed:frozen_checks.append(dict(material=label,path=file['path'],matched=observed['sha256']==file['sha256']))
    assert all(r['matched'] for r in frozen_checks)
    result=dict(written_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),material_comparison=material,quality=quality,geometry=geometry,atomic_matching=atomic,examples=examples,material_identity_checks=identity_checks,code_freeze_checks=frozen_checks,
                primary_comparisons=[dict(material=label,**r) for label,s in summaries.items() for r in s['comparisons'] if r['family']=='all' and r['split']!='calibration'],
                inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in [BASE/'CONFIRM_LOW_SUMMARY.json',BASE/'CONFIRM_HIGH_SUMMARY.json',BASE/'FREEZE.json',BASE/'SOURCE_REVIEW.json']],
                claims_boundary='The fixed nearest-direction source grouping is evaluated, not the optimal semantic content of an SAE. Both source and target change in the L1 comparison, so only true-factor errors use a common external teacher; fidelity gains compare methods within one material.')
    (BASE/'KEY_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print('MATERIAL',[(r['regime'],r['split'],round(r['low_truth_nmse'],4),round(r['high_truth_nmse'],4),round(100*r['relative_reduction'],2),r['seeds_better']) for r in material])
    print('GEOMETRY',[(r['regime'],r['toy_seed'],round(r['pair_separation'],5),round(r['shift_pair_difference_nmse'],5)) for r in geometry])
    print('PRIMARY',[(r['regime'],r['split'],round(100*r['function_relative_reduction'],2),sum(x['function_relative_reduction']>0 for x in r['per_direction']),len(r['per_direction']),sum(x['function_relative_reduction']>0 for x in r['incident_seed_deletions'])) for r in summaries['HIGH']['comparisons'] if r['family']=='all' and r['split']!='calibration' and r['comparator']=='ols_affine_clip'])
    print('EXAMPLES',[(e['label'],e['evaluation_row'],e['source_members'],e['source_state'],e['latent'][2]) for e in examples])
if __name__=='__main__':main()
