"""Fixed discovery groups; functional interventions, not semantic atom labels."""
import hashlib
import json
from pathlib import Path
import numpy as np


def choose_groups(order, energies, *, budget, pool_size, seed):
    order=np.asarray(order,dtype=int);energies=np.asarray(energies,dtype=float)
    if not 0<budget<=pool_size<=len(order) or len(set(order.tolist()))!=len(order):
        raise ValueError('Invalid component group budget/order')
    pool=order[:pool_size]
    random=np.sort(np.random.default_rng(seed).choice(pool,size=budget,replace=False))
    top=order[:budget]
    return dict(top_atoms=top.tolist(),random_atoms=random.tolist(),random_pool=pool.tolist(),
        overlap=int(len(set(top)&set(random))),top_discovery_energy=float(energies[top].sum()),
        random_discovery_energy=float(energies[random].sum()),pool_discovery_energy=float(energies[pool].sum()))


def group_coordinates(z,beta,top,random):
    z=np.asarray(z);beta=np.asarray(beta)
    full=z@beta;head=z[:,top]@beta[top];tail=full-head;other=z[:,random]@beta[random]
    return dict(full=full,top=head,tail=tail,random=other)


def prepare_groups(cfg,families,run,paths,sha256,rank_order):
    spec=cfg['component_interventions'];saved=json.loads(paths['saved_readout'].read_text())
    manifest_path=Path(saved['discovery_manifest_path'])
    if sha256(manifest_path)!=saved['discovery_manifest_sha256']:
        raise ValueError('Component discovery manifest changed')
    asset=manifest_path.parent;manifest=json.loads(manifest_path.read_text())
    n=next(r['tokens'] for r in manifest['splits'] if r['split']=='discovery')
    shape=(n,cfg['k']);groups={};records=[]
    for r in saved['families']:
        s,a,t=(r[k] for k in ('source_seed','source_atom','target_seed'));family=families[s,a,t]
        rows=np.asarray(r['discovery_rows'],dtype=int)
        ind=np.memmap(asset/'discovery'/f'seed_{t}'/'top_indices.uint16.bin',dtype='<u2',mode='r',shape=shape)
        val=np.memmap(asset/'discovery'/f'seed_{t}'/'top_acts.float32.bin',dtype='<f4',mode='r',shape=shape)
        x=np.zeros((len(rows),cfg['num_latents']))
        np.add.at(x,(np.arange(len(rows))[:,None],ind[rows]),val[rows])
        order,energy=rank_order(x,family['beta'],r['discovery_weights'])
        if order[:spec['budget']].tolist()!=r['top_atoms'][:spec['budget']]:
            raise ValueError('Replayed discovery top group changed')
        seed=int.from_bytes(hashlib.sha256(f"{spec['random_seed']}:{s}:{a}:{t}".encode()).digest()[:8],'little')
        g=choose_groups(order,energy,budget=spec['budget'],pool_size=spec['random_pool_size'],seed=seed)
        groups[s,a,t]=g
        records.append(dict(source_seed=s,source_atom=a,target_seed=t,seed=seed,**g))
    payload=dict(spec=spec,fit_split='discovery',calibration_used_for_group_selection=False,refitted=False,
        discovery_manifest_path=str(manifest_path),discovery_manifest_sha256=sha256(manifest_path),groups=records,
        scope='Random16 sampled once from discovery-ranked top64; overlap/energies retained. No test-energy matching or semantic labels. Complement is all remaining atoms, not equal-size to top16.')
    (run/'component_group_selection.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return groups,dict(path=str(manifest_path),sha256=sha256(manifest_path),bytes=manifest_path.stat().st_size,
        source='Existing discovery assets',license_or_access_boundary='internal',role='component_discovery_manifest')
