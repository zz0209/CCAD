"""Export and replay rectified maps against the run's stored direct arrays."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ccad.rectified_operation import RectifiedOperation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);cfg=json.loads((a.run/'config.resolved.json').read_text());checks=[];examples=[]
    for regime in cfg['regimes']:
        for base in cfg['toy_seeds']:
            with np.load(a.run/f'{regime}_{base}_data.npz') as packed:arrays={k:packed[k] for k in packed.files}
            fitpath=a.run/f'{regime}_{base}_frozen_maps.json';fitdata=json.loads(fitpath.read_text());operations=[]
            for fit in fitdata['fits']:
                model=fit['models']['ols_rectified'];si=cfg['sae_seeds'].index(fit['source_seed']);ti=cfg['sae_seeds'].index(fit['target_seed']);sg=fit['source_group'];idx=model['indices'];coef=np.asarray(model['coef']).reshape(len(idx),len(sg))
                rec={**{k:fit[k] for k in ['regime','toy_seed','source_seed','target_seed','factor']},'target_members':idx,'source_members':sg,'target_mean':np.array(fit['target_mean'])[idx].tolist(),'coefficient':coef.tolist(),'intercept':model['mu'],'source_decoder':arrays['decoder'][si,sg].tolist(),'hidden_dimension':cfg['hidden'],'method':'ols_rectified'}
                operations.append(rec);op=RectifiedOperation.from_record(rec)
                for split in ['evaluation','shift']:
                    x=arrays[split+'_z'][:,ti];direct=-np.maximum((x[:,idx]-np.array(fit['target_mean'])[idx])@coef+model['mu'],0)@arrays['decoder'][si,sg]
                    result=op.apply(x,components=True)
                    replay=float(np.max(np.abs(result['delta']-direct)));parts=float(np.max(np.abs(result['component_deltas'].sum(1)+result['baseline_delta']-direct)))
                    checks.append(dict(regime=regime,toy_seed=base,source_seed=fit['source_seed'],target_seed=fit['target_seed'],factor=fit['factor'],split=split,rows=len(x),max_prediction_error=replay,max_component_sum_error=parts))
                if base==202 and regime=='independent' and fit['source_seed']==1 and fit['target_seed']==2 and fit['factor']==2:
                    xtrue=arrays['evaluation_x'];state=arrays['evaluation_z'][:,si][:,sg];active=(xtrue>0).sum(1)
                    conditions=[('factor_alone',(xtrue[:,2]>0)&(active==1)),('factor_with_others',(xtrue[:,2]>0)&(active>=3)),('factor_absent_source_active',(xtrue[:,2]==0)&(state.sum(1)>1e-6))]
                    ids=[];labels=[]
                    for label,condition in conditions:
                        eligible=np.flatnonzero(condition)
                        if len(eligible):ids.append(int(eligible[0]));labels.append(label)
                    target=arrays['evaluation_z'][ids,ti];donor=target[::-1].copy();res=op.apply(target,components=True)
                    np.savez_compressed(a.out/'example_inputs.npz',recipient_codes=target,donor_codes=donor,example_ids=ids,target_members=op.target_members,expected_delta=res['delta'],expected_contrast=op.apply(target,donor)['delta'])
                    examples.append(dict(regime=regime,toy_seed=base,source_seed=1,target_seed=2,factor=2,ids=ids,labels=labels,rule='First evaluation row meeting each predeclared source/truth criterion, never selected by relative method error.'))
            collection=dict(schema='ccad.rectified.operations.v1',run_id=a.run.name,regime=regime,toy_seed=base,l1=cfg['l1'],frozen_maps_sha256=hashlib.sha256(fitpath.read_bytes()).hexdigest(),operations=operations)
            (a.out/f'{regime}_{base}_maps.json').write_text(json.dumps(collection,indent=2)+'\n')
    result=dict(run=a.run.name,maps=sum(1 for r in checks if r['split']=='evaluation'),rows=sum(r['rows'] for r in checks),max_prediction_error=max(r['max_prediction_error'] for r in checks),max_component_sum_error=max(r['max_component_sum_error'] for r in checks),checks=checks,examples=examples,scope='NumPy consumer versus saved fitted arrays on every audit/shift row; source fidelity and semantic truth are distinct scientific checks.')
    (a.out/'REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','examples']}))
if __name__=='__main__':main()
