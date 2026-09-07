"""Export NumPy-only paired maps and replay them on the retained input panel."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ccad.predictive_operation import PredictiveOperation


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run=args.run;out=args.out;out.mkdir(exist_ok=False,parents=True)
    cfg=json.loads((run/'config.resolved.json').read_text())
    if not cfg.get('pair_anchor_followup'):raise ValueError('Requires a paired-completion run')
    panel=json.loads((ROOT/cfg['material_run']/'panel.json').read_text());entries=[];maxnorm=0.;maxcontrast=0.;lpmax=0.;klmax=0.;replayed=0
    observed={};original={}
    for line in (run/'metrics.raw.jsonl').read_text().splitlines():
        row=json.loads(line);key=(row['source_seed'],row['factor'],row['consumer'],row['row_id'])
        if row['method']=='anchor_compact':observed[key]=row
        if row['method']=='legacy_contrast':original[key]=row
    for source,target in cfg['seed_pairs']:
        codepath=ROOT/cfg['evaluation_code_run']/f'seed{target}_codes.npz'
        with np.load(codepath,allow_pickle=False) as codes:z=codes['number_z'].astype(np.float64)
        for factor in ['number','time']:
            mappath=run/f'maps_s{source}_t{target}_{factor}.npz'
            with np.load(mappath,allow_pickle=False) as bundle:
                members=bundle['target_members'];basis=bundle['basis'];w=bundle['legacy_weights'];v=bundle['anchor_compact_common_weights']
            ident=f's{source}_t{target}_{factor}';donors=np.array([p[factor] for p in panel['pairs']]);x=z[:,members];d=x[donors]
            metadata=dict(paired_completion_id=ident+'_'+sha(mappath)[:12],source_seed=source,target_seed=target,factor=factor,
                model_local_dir=cfg['model_local_dir'],hook_layer=cfg['sae_layer'],position='final token after the last transformer block, before final LayerNorm',
                sae_checkpoint=next(s['path'] for s in cfg['sae_checkpoints'] if s['seed']==target),
                original_run=cfg['run_id'],map_sha256=sha(mappath),code_input_sha256=sha(codepath),
                scope='Retrospective donor-conditioned source-aligned completion, not native or semantic deletion. No fit occurs in this export.')
            odd=PredictiveOperation(members,w@basis.T,z.shape[1],dict(metadata,paired_part='odd'))
            common=PredictiveOperation(members,v@basis.T,z.shape[1],dict(metadata,paired_part='common'))
            oddpath=out/(ident+'_odd.npz');commonpath=out/(ident+'_common.npz');odd.save(oddpath);common.save(commonpath)
            # Reassociation is an independent consumer replay of saved parameters,
            # not refitting or independent scientific confirmation.
            removal=-(odd.predict_selected((x-d)/2)+common.predict_selected((x+d)/2))
            contrast=odd.predict_selected(d-x)
            old_absolute=(x@w)@basis.T;old_delta=old_absolute[donors]-old_absolute
            maxcontrast=max(maxcontrast,float(np.max(np.abs(contrast.astype(np.float32)-old_delta.astype(np.float32)))))
            for i in range(len(x)):
                row=observed[source,factor,'complete_removal',i]
                maxnorm=max(maxnorm,abs(float(np.linalg.norm(removal[i]))-row['delta_norm']))
                new=observed[source,factor,'contrast',i];old=original[source,factor,'contrast',i]
                lpmax=max(lpmax,max(abs(a-b) for a,b in zip(new['label_logprobs'],old['label_logprobs'])))
                klmax=max(klmax,abs(new['kl_reference']-old['kl_reference']))
            replayed+=len(x);ids=np.array([0,256]);inputpath=out/(ident+'_example.npz')
            np.savez_compressed(inputpath,selected_recipient=x[ids],selected_donor=d[ids],target_members=members,row_ids=ids)
            outputpath=out/(ident+'_applied.npz')
            command=[sys.executable,str(ROOT/'scripts/apply_paired_completion.py'),'--odd-operation',str(oddpath),'--common-operation',str(commonpath),'--input',str(inputpath),'--output',str(outputpath)]
            process=subprocess.run(command,capture_output=True,text=True)
            if process.returncode:raise RuntimeError(process.stderr)
            with np.load(outputpath,allow_pickle=False) as applied:
                np.testing.assert_allclose(applied['removal_delta'],removal[ids],atol=1e-10,rtol=1e-10)
            entries.append(dict(id=ident,source_seed=source,target_seed=target,factor=factor,members=members.tolist(),
                command=command,stdout=process.stdout,files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in [oddpath,commonpath,inputpath,outputpath]]))
    assert replayed==5120 and maxnorm<1e-9 and lpmax==0 and klmax==0
    result=dict(written_utc=datetime.now(timezone.utc).isoformat(),entries=entries,replayed_rows=replayed,
        maximum_removal_norm_error=maxnorm,maximum_float32_contrast_difference=maxcontrast,
        maximum_recorded_label_logprob_difference=lpmax,maximum_full_vocab_kl_difference=klmax,
        scope='Numerical consumer replay of existing coefficients and all exposed rows; no new model inference, fit or independent replication.')
    (out/'INDEX.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='entries'}))


if __name__=='__main__':main()
