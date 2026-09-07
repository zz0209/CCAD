"""Export the frozen primary source allocations and replay every tested mask."""
import argparse,json,sys,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ccad.component_operation import ComponentOperation
from ccad.component_correspondence import predefined_masks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    run=args.run;cfg=json.loads((run/'config.resolved.json').read_text());assert json.loads((run/'metrics.summary.json').read_text())['status']=='PASS'
    assert json.loads((run/'contract_validation.json').read_text())['ok'];args.output.mkdir(exist_ok=False,parents=True)
    panel=json.loads((ROOT/cfg['material_run']/'panel.json').read_text());entries=[];maximum=0.;count=0
    for source,target in cfg['seed_pairs']:
        z=np.load(ROOT/cfg['evaluation_code_run']/f'seed{target}_codes.npz')['number_z'].astype(np.float64)
        for factor in ['number','time']:
            p=run/f'maps_s{source}_t{target}_{factor}.npz';a=np.load(p);members=a['aggregate_ols_members'];d=a['source_decoder'];h=a['aggregate_ols_weights'];sm=a['source_members']
            meta=dict(factor=factor,source_seed=source,target_seed=target,hook='gpt_neox.layers.15 resid_post at final token',source_checkpoint=next(x['path'] for x in cfg['sae_checkpoints'] if x['seed']==source),target_checkpoint=next(x['path'] for x in cfg['sae_checkpoints'] if x['seed']==target),model=cfg['model_local_dir'],fit_run=cfg['frozen_fit_run'],confirmation_run=run.as_posix(),map_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope=cfg['scope'])
            op=ComponentOperation(members,sm,h,d,meta);op_path=args.output/f's{source}_t{target}_{factor}.npz';op.save(op_path);op=ComponentOperation.load(op_path)
            donor=np.array([x[factor] for x in panel['pairs']]);pred=np.load(run/f'predictions_s{source}_t{target}_{factor}.npz')['aggregate_ols']
            masks=predefined_masks(len(sm),source*100+(factor=='time'))
            for name,mask in masks:
                for consumer in cfg['consumers']:
                    for dose in cfg.get('mask_doses',{}).get(name,cfg['doses']):
                        expected=dose*((pred[donor]-pred if consumer=='contrast' else -pred)*mask)@d
                        delta=op.apply(z[:,members],mask,'contrast' if consumer=='contrast' else 'removal',z[donor][:,members] if consumer=='contrast' else None,dose)
                        err=float(np.max(np.abs(delta-expected)));assert err<1e-9;maximum=max(maximum,err);count+=len(z)
            ids=np.array([0,256]);mask=masks[1][1];example=args.output/f'example_s{source}_t{target}_{factor}.npz'
            np.savez_compressed(example,selected_codes=z[ids][:,members],selected_donor_codes=z[donor[ids]][:,members],target_members=members,source_members=sm,source_scales=mask,row_ids=ids,expected_removal=(-pred[ids]*mask)@d,expected_contrast=((pred[donor[ids]]-pred[ids])*mask)@d)
            entries.append(dict(operation=op_path.name,sha256=hashlib.sha256(op_path.read_bytes()).hexdigest(),example=example.name,source_seed=source,target_seed=target,factor=factor,target_members=len(members),source_members=len(sm)))
    (args.output/'EXAMPLE_CONTEXTS.json').write_text(json.dumps(dict(rows=[r for r in panel['rows'] if r['id'] in [0,256]],source_mask='First half in the recorded original source rank; member IDs and mask array in each example NPZ. Not a semantic category.',consumer='Recipient-only removal by default; contrast additionally uses the paired donor codes.'),indent=2)+'\n')
    out=dict(written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run.as_posix(),entries=entries,replayed_vectors=count,maximum_absolute_error=maximum,scope='Numerical identity check for primary saved map over every frozen operation and row; no new model evaluation or scientific independence.')
    (args.output/'INDEX.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='entries'}))


if __name__=='__main__':main()
