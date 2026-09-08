"""Raw and complete-SAE coverage for independently named RAVEL attributes."""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write

def semantic_measure(w,ids,delta,method,mode,control,seed=0,target_seed=None,reference=None,**extra):
    outputs=np.zeros((len(ids),w.baseline.shape[-1]),np.float64)
    for jj in w.batches(np.arange(len(ids))):outputs[jj],_=w.forward(ids[jj],delta[jj])
    attrs=w.cfg['tasks']
    for j,i in enumerate(ids):
        row=w.panel[i];a=attrs.index(row['task']);cause=bool(control[a]>.5)
        accepted=row['donor_expected_ids'] if cause else row['expected_ids']
        top=int(outputs[j].argmax());expected_lp=float(np.logaddexp.reduce(outputs[j,accepted]))
        before=float(np.logaddexp.reduce(w.baseline[i,accepted]))
        ref=w.baseline[w.donors[i]] if reference is None else reference[j]
        w.record(kind='semantic',task=row['task'],row_id=int(i),component=row['component'],pair_key=row['pair_key'],
            split=row['split'],mode=mode,method=method,seed=seed,target_seed=target_seed,
            operation=''.join(str(int(x)) if x in [0,1] else str(x) for x in control),control=list(map(float,control)),
            endpoint='Cause' if cause else 'Iso',first_token_correct=top in accepted,expected_probability=float(np.exp(expected_lp)),
            expected_logprob=expected_lp,expected_logprob_change=expected_lp-before,
            predicted_token_id=top,predicted_token_text=w.tokenizer.decode([top]),entity=row['entity'],donor_entity=row['donor_entity'],
            base_label_correct=int(w.baseline[i].argmax()) in row['expected_ids'],
            kl_to_reference=max(0.,float(np.sum(np.exp(ref)*(ref-outputs[j])))),
            reference_identity='natural_donor' if reference is None else 'specified_source_control',
            edit_norm=float(np.linalg.norm(delta[j])),**extra)
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    sources=['scripts/run_ravel_source_coverage.py','scripts/run_causalgym_multisite.py','scripts/prepare_ravel_semantic_panel.py',
        'scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,sources);error=None
    try:
        w.setup();ids=np.arange(len(w.panel))
        zero=np.zeros_like(w.hidden)
        semantic_measure(w,ids,zero,'unedited','none',[0,0,0])
        for mode in cfg['modes']:
            delta,valid,_=w.delta(ids,mode,w.hidden)
            if not valid.all():raise ValueError('Incomplete raw coverage')
            semantic_measure(w,ids,delta,'raw_donor',mode,[1,1,1]);w.progress('RAW_COVERAGE',mode=mode)
        material=[]
        compact=cfg.get('compact_single_site',False)
        if compact:
            if len(cfg['modes'])!=1:raise ValueError('Compact coverage needs one common site definition')
            valid,aa=w.alignment(ids,cfg['modes'][0])
            if not valid.all() or any(len(a)!=1 for a in aa):raise ValueError('Compact coverage needs one aligned token')
            positions=np.asarray([a[0][0] for a in aa]);dpositions=np.asarray([a[0][1] for a in aa])
            if not np.array_equal(positions[w.donors],dpositions):raise ValueError('Donor compact positions differ')
        for seed in cfg['seeds']:
            if seed==0:continue
            sae=w.load_sae(seed,positions if compact else None)
            for mode in cfg['modes']:
                if compact:
                    with w.torch.no_grad():
                        decoded=(w.torch.as_tensor(sae['codes'][w.donors]-sae['codes'],device=w.device)@
                                 w.torch.as_tensor(sae['decoder'],device=w.device)).cpu().numpy()
                    delta=np.zeros_like(w.hidden);delta[ids,positions]=decoded
                    raw_change=w.hidden[w.donors,dpositions]-w.hidden[ids,positions]
                    for j,i in enumerate(ids):
                        material.append(dict(seed=seed,row_id=int(i),component=w.panel[i]['component'],task=w.panel[i]['task'],
                            split=w.panel[i]['split'],raw_energy=float(np.sum(raw_change[j].astype(float)**2)),
                            decoded_energy=float(np.sum(decoded[j].astype(float)**2)),
                            squared_difference_error=float(np.sum((decoded[j].astype(float)-raw_change[j])**2))))
                else:delta,valid,_=w.delta(ids,mode,sae['codes'],sae['decoder'])
                semantic_measure(w,ids,delta,'full_sae',mode,[1,1,1],seed=seed)
                w.progress('SAE_COVERAGE',mode=mode,seed=seed)
            del sae
        if material:write(w.run/'difference_reconstruction.json',dict(rows=material,
            definition='Sum squared SAE donor-difference error divided by sum squared raw donor differences within a stated panel. No per-row ratio averaging or nonnegative clipping; distinct from ordinary natural-state FVE.'))
        groups={}
        for r in w.metrics:groups.setdefault((r['task'],r['mode'],r['method'],r['seed'],r['split']),[]).append(r)
        write(w.run/'semantic_summary.json',dict(cells=[dict(task=k[0],mode=k[1],method=k[2],seed=k[3],split=k[4],n=len(rr),
            first_token_correct=float(np.mean([r['first_token_correct'] for r in rr])),
            expected_probability=float(np.mean([r['expected_probability'] for r in rr]))) for k,rr in groups.items()]))
    except Exception:error=traceback.format_exc()
    return w.finish(error)

if __name__=='__main__':raise SystemExit(main())
