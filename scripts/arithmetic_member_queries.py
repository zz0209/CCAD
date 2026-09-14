"""Materialize unfitted complementary source-member queries for generation."""
import hashlib,json
from datetime import datetime,timezone


def prepare_queries(w,cfg,saes):
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    spec=cfg['member_queries'];parent=w.run.parent.parent/spec['relation_run']
    assert json.loads(w.checked(parent/'status.json').read_text())['status']=='PASS'
    prior=json.loads(w.checked(parent/'config.resolved.json').read_text())
    assert all(prior[k]==cfg[k] for k in ['training_run','checkpoint_step','model_revision','source_cache_run'])
    cap=cfg['members'][0];gates={};records=[]
    for s,t in prior['relation_transfer']['seed_pairs']:
        path=w.checked(parent/f'relation_s{s}_t{t}.npz')
        with np.load(path) as z:
            source=torch.tensor(z['source_gate'],device=w.device)
            masks={f'{name}_part{part}':torch.zeros_like(source) for name in
                   ['source','member','assignment','two_assignment','wrong'] for part in [0,1]}
            query_records=[]
            ds,dt=saes[s].decoder.weight.T,saes[t].decoder.weight.T
            for c,op in enumerate(['unit','tens']):
                si=z[f'{op}_source_indices'];ti=z[f'{op}_target_indices']
                relation=torch.tensor(z[f'{op}_weights'],device=w.device)
                order=np.random.default_rng(spec['partition_seed']+s*10+c).permutation(len(si))
                q=np.zeros(len(si),dtype=np.float32);q[order[:len(si)//2]]=1
                qq=torch.tensor(q,device=w.device)
                ss=torch.tensor(si,device=w.device);tt=torch.tensor(ti,device=w.device)
                sd=ds[ss]
                cosine=((sd/sd.norm(dim=1,keepdim=True).clamp_min(1e-12))@
                        (dt/dt.norm(dim=1,keepdim=True).clamp_min(1e-12)).T).cpu().numpy()
                ar,ac=linear_sum_assignment(-cosine)
                for part,partq in enumerate([qq,1-qq]):
                    active=torch.where(partq>0)[0]
                    masks[f'source_part{part}'][:,ss,c]=source[:,ss,c]*partq
                    masks[f'member_part{part}'][:,tt,c]=relation@partq
                    masks[f'wrong_part{part}'][:,tt,c]=relation@(1-partq)
                    masks[f'assignment_part{part}'][:,torch.tensor(ac,device=w.device),c]=source[:,torch.tensor(si[ar],device=w.device),c]*partq[torch.tensor(ar,device=w.device)]
                    duplicates=np.repeat(active.cpu().numpy(),2)
                    assert len(duplicates)<=cap
                    rr,cc=linear_sum_assignment(-cosine[duplicates])
                    masks[f'two_assignment_part{part}'][:,torch.tensor(cc,device=w.device),c]=source[:,torch.tensor(si[duplicates[rr]],device=w.device),c]
                query_records.append(dict(operation=op,source_indices=si.tolist(),part0_indices=si[q>0].tolist(),
                                          target_indices=ti.tolist(),source_members=len(si)))
            for name,g in masks.items():
                assert bool(((g>=0)&(g<=1+1e-6)).all())
                assert bool(((g>0).any(0).sum(0)<=cap).all())
                seed=s if name.startswith('source_') else t
                assert (seed,name,cap) not in gates
                gates[seed,name,cap]=g.clamp(0,1)
            np.savez_compressed(w.run/f'member_queries_s{s}_t{t}.npz',
                                **{name:g.cpu().numpy() for name,g in masks.items()})
            records.append(dict(source_seed=s,target_seed=t,parent=path.as_posix(),
                                parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),queries=query_records))
    meta=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),records=records,
              scope='Exposed development. Two source-only complementary partitions, no query-output fitting. Source rows indexed by sourceSAE, translated rows by targetSAE; analysis aligns the cycle.',
              target_output_gradients=0,target_output_labels=0,partition_seed=spec['partition_seed'],
              assignment_budget='One-to-one uses32; duplicated assignment uses64 distinct target members per32-member sourcepart. All methods allowed64.')
    (w.run/'MEMBER_QUERY_FREEZE.json').write_text(json.dumps(meta,indent=2)+'\n')
    return gates,meta
