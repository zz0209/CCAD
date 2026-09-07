"""Export shared complete-support operations and replay all retained inputs."""
import argparse, hashlib, json, sys
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ccad.predictive_operation import PredictiveOperation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    run=args.run;cfg=json.loads((run/'config.resolved.json').read_text());assert json.loads((run/'metrics.summary.json').read_text())['status']=='PASS'
    assert json.loads((run/'contract_validation.json').read_text())['ok']
    args.output.mkdir(exist_ok=False,parents=True)
    evaluation=ROOT/cfg['evaluation_code_run'] if not cfg.get('target_code_run') else ROOT/cfg['target_code_run']/'evaluation'
    specs=cfg.get('target_sae_checkpoints',cfg['sae_checkpoints']);panel=json.loads((ROOT/cfg['material_run']/'panel.json').read_text());entries=[];maxerr=0.;replayed=0
    for source,target in cfg['seed_pairs']:
        z=np.load(evaluation/f'seed{target}_codes.npz')['number_z'].astype(np.float64)
        for factor in ['number','time']:
            path=run/f'maps_s{source}_t{target}_{factor}.npz';a=np.load(path);members=a['ols_complete_members'];w=a['ols_complete_weights'];basis=a['basis']
            meta=dict(input_kind='absolute_target_code',factor=factor,source_seed=source,target_seed=target,source_training_tokens=4194304,
                target_checkpoint=next(x['path'] for x in specs if x['seed']==target),model_local_identity=cfg['model_local_dir'],hook='gpt_neox.layers.15 resid_post at final token before finalLayerNorm',
                source_group=a['source_support'].tolist(),input_shape='rows by target SAE codes in original native code units',method='shared OLS complete-support with source-balanced pair ridge',
                scope='Retrospective development; complete source-group prediction, not native target or semantic deletion',map_run=run.as_posix(),map_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            op=PredictiveOperation(members,w@basis.T,z.shape[1],meta);op_path=args.output/f's{source}_t{target}_{factor}.npz';op.save(op_path);loaded=PredictiveOperation.load(op_path)
            donor=np.array([x[factor] for x in panel['pairs']]);truth=z[:,members]@w@basis.T
            for consumer,data,expected in [('contribution',z[:,members],truth),('removal',-z[:,members],-truth),('contrast',z[donor][:,members]-z[:,members],truth[donor]-truth)]:
                err=float(np.max(np.abs(loaded.predict_selected(data)-expected)));assert err<1e-9;maxerr=max(maxerr,err);replayed+=len(z)
            ids=np.array([0,256]);example=args.output/f'example_s{source}_t{target}_{factor}.npz'
            np.savez_compressed(example,selected_codes=z[ids][:,members],selected_donor_codes=z[donor[ids]][:,members],target_members=members,row_ids=ids,expected_contribution=truth[ids],expected_removal=-truth[ids],expected_contrast=truth[donor[ids]]-truth[ids])
            entries.append(dict(operation=op_path.name,sha256=hashlib.sha256(op_path.read_bytes()).hexdigest(),example=example.name,source_seed=source,target_seed=target,factor=factor,members=len(members)))
    result=dict(written_utc=datetime.now(timezone.utc).isoformat(),run=run.as_posix(),entries=entries,replayed_vectors=replayed,maximum_absolute_error=maxerr,scope='Numerical replay of saved maps on all512rows for three linear consumers. Not an independent scientific evaluation or model inference.')
    (args.output/'INDEX.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='entries'}))


if __name__=='__main__':main()
