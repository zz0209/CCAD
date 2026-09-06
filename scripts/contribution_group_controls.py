"""Source-only nuisance-matched random groups for contribution diagnostics."""
import hashlib
import numpy as np


def matched_group(cov, decoder, frequency, components, eligible, coherent, seed, query_id, cfg, outdir):
    import torch
    device = cov.device
    size = len(coherent)
    dg = decoder@decoder.T
    energy = torch.diag(cov)*torch.diag(dg)
    ef = torch.stack([torch.log(energy.clamp(min=1e-20)), torch.log(frequency.clamp(min=1e-6))],dim=1)
    candidate_atoms = sorted(set(eligible)-set(coherent))
    pools = []
    for atom in coherent[1:]:
        distance = torch.sum((ef[candidate_atoms]-ef[atom])**2,dim=1)
        pools.append(np.array(candidate_atoms)[torch.argsort(distance,stable=True)[:cfg['member_pool']].cpu().numpy()])
    salt = hashlib.sha256(f"{cfg['salt']}:{seed}:{query_id}".encode()).hexdigest()
    rng = np.random.default_rng(int(salt[:16],16))
    ids = np.empty((cfg['candidates'],size),dtype=np.int64)
    ids[:,0] = coherent[0]
    for j,pool in enumerate(pools,1):
        ids[:,j] = rng.choice(pool,len(ids))
        duplicates = np.any(ids[:,j,None]==ids[:,:j],axis=1)
        while np.any(duplicates):
            ids[duplicates,j] = rng.choice(pool,int(duplicates.sum()))
            duplicates = np.any(ids[:,j,None]==ids[:,:j],axis=1)

    def describe(ix):
        c = cov[ix[:,:,None],ix[:,None,:]]
        d = dg[ix[:,:,None],ix[:,None,:]]
        a = c@d
        en = torch.diagonal(a,dim1=1,dim2=2).sum(1)
        rank = en**2/torch.sum(a*a.transpose(1,2),dim=(1,2))
        dp = decoder[ix]@components
        pc = (dp*(c@dp)).sum(1).cumsum(1)/en[:,None]
        # Source-output spectrum, independent of vector orientation.
        ce,cu = torch.linalg.eigh(c)
        cs = (cu*torch.sqrt(ce.clamp(min=0))[:,None,:])@cu.transpose(1,2)
        spectrum = torch.linalg.eigvalsh(cs@d@cs).clamp(min=0).flip(1)
        cumulative = spectrum.cumsum(1)/en[:,None]
        desc = torch.stack([en,rank,cumulative[:,0],cumulative[:,3],pc[:,0],pc[:,15],pc[:,63],frequency[ix].mean(1)],dim=1)
        return desc,spectrum

    with torch.no_grad():
        ix = torch.tensor(ids,device=device)
        des,spectrum = describe(ix)
        goal,goal_spectrum = describe(torch.tensor([coherent],device=device))
        # Relative balance of energy/rank/frequency; bounded shares use fixed 0.10 scale.
        dist = torch.log(des[:,[0,1,7]]/goal[:,[0,1,7]]).square().sum(1)
        dist += ((des[:,[2,3,4,5,6]]-goal[:,[2,3,4,5,6]])/.10).square().sum(1)
        chosen = int(torch.argmin(dist))
        multiplier = float(torch.sqrt(goal[0,0]/des[chosen,0]))
    np.savez_compressed(outdir/f'control_search_s{seed}_q{query_id}.npz',candidate_atoms=ids,
        descriptors=des.cpu().numpy(),source_coherent_descriptor=goal.cpu().numpy(),spectra=spectrum.cpu().numpy(),
        source_coherent_spectrum=goal_spectrum.cpu().numpy(),objective=dist.cpu().numpy(),selected_index=chosen)
    return dict(source_atoms=ids[chosen].tolist(),source_weights=[multiplier]*size,
        control_balance=dict(columns=['energy','effective_rank','top1_share','top4_share','raw_pc1_share','raw_pc16_share','raw_pc64_share','mean_frequency'],
            coherent=goal[0].cpu().numpy().tolist(),selected_before_rescaling=des[chosen].cpu().numpy().tolist(),
            energy_multiplier=multiplier,source_only_objective=float(dist[chosen]),candidate_count=len(ids),
            source_energy_equal_on_discovery=True,rank_and_shares_matching_attempted=True))
