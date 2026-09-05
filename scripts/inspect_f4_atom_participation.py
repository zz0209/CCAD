"""Descriptive algebraic participation at the saved intervention positions.

No fitting or new forward passes. Energy participation is not atom necessity,
native deletion equivalence, or evidence of human-readable concepts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ.update(OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def participation(z, mean, weights):
    terms=(z-mean)[...,None]*weights[None,:,:]
    energy=np.sum(terms**2,axis=(0,2)); total=float(energy.sum())
    aggregate=np.sum(terms,axis=1); aggregate_energy=float(np.sum(aggregate**2))
    dynamic=z@weights; constant=mean@weights
    return {'effective_energy_atoms':float(total**2/np.sum(energy**2)) if total else None,
            'largest_atom_energy_share':float(energy.max()/total) if total else None,
            'sum_atom_energy_over_aggregate':total/aggregate_energy if aggregate_energy else None,
            'mean_energy_over_centered_aggregate':float(len(z)*np.sum(constant**2)/aggregate_energy) if aggregate_energy else None,
            'dynamic_energy_over_centered_aggregate':float(np.sum(dynamic**2)/aggregate_energy) if aggregate_energy else None,
            'aggregate_energy':aggregate_energy}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source-only',action='store_true');args=p.parse_args()
    run=args.run_dir;cfg=json.loads((run/'config.resolved.json').read_text())
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    raw=run/'metrics.raw.jsonl'
    selection=json.loads((run/'selection.json').read_text())['queries']
    surface={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in (json.loads(s) for s in (ROOT/cfg['surface_path']).read_text().splitlines()) if r['rank']==1 and r['query_role']=='anchor'}
    f=np.load(ROOT/cfg['factors_path'],allow_pickle=False)
    lookup={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(f['source_seed'],f['source_atom'],f['target_seed']))}
    means={s:np.zeros(cfg['num_latents']) for s in cfg['source_seeds']}
    for line in (ROOT/cfg['source_census_path']).read_text().splitlines():
        r=json.loads(line);means[r['seed']][r['atom']]=r['mean_code']
    asset=Path(cfg['bulk_asset_dir']);h=cfg['hook_hidden_size']
    asset_manifest=json.loads((asset/'asset_manifest.json').read_text())
    tokens=next(r['tokens'] for r in asset_manifest['splits'] if r['split']=='calibration')
    shape=(tokens,cfg['k'])
    indices={s:np.memmap(asset/'calibration'/f'seed_{s}'/'top_indices.uint16.bin',dtype='<u2',mode='r',shape=shape) for s in means}
    acts={s:np.memmap(asset/'calibration'/f'seed_{s}'/'top_acts.float32.bin',dtype='<f4',mode='r',shape=shape) for s in means}
    dec={s:np.memmap(asset/'decoders'/f'seed_{s}.float32.bin',dtype='<f4',mode='r',shape=(cfg['num_latents'],h)) for s in means}
    if args.source_only:
        raw_manifest=json.loads((Path(cfg['raw_hook_asset_dir'])/'raw_hook_manifest.json').read_text())
        entry=next(r for r in raw_manifest['splits'] if r['split']=='calibration')
        shared_hook=np.memmap(entry['path'],dtype='<f4',mode='r',shape=tuple(entry['shape']))
    out=[]
    for q in selection:
        s,a=q['source_seed'],q['source_atom'];t0=q['targets'][0];ids=surface[s,a,t0]['source_candidate_ids']
        for e in q['sequences']:
            pos=np.asarray(e['intervention_positions'],dtype=int)+e['sequence']*cfg['context_length']
            donor_pos=np.asarray(e.get('donor_positions',[]),dtype=int)+e.get('donor_sequence',e['sequence'])*cfg['context_length']
            missing_donor=bool(cfg.get('donor_difference') and len(pos)!=len(donor_pos))
            if missing_donor and not args.source_only:
                raise ValueError('Donor positions missing; use source-only mode to retain unsupported rows explicitly')
            z={}
            for seed in ([s] if args.source_only else [s]+q['targets']):
                z[seed]=np.zeros((len(pos),cfg['num_latents']))
                np.add.at(z[seed],(np.arange(len(pos))[:,None],indices[seed][pos]),acts[seed][pos])
                if cfg.get('donor_difference'):
                    if missing_donor:
                        z[seed][:]=0
                    else:
                        np.add.at(z[seed],(np.arange(len(pos))[:,None],indices[seed][donor_pos]),-acts[seed][donor_pos])
            active_means={seed:np.zeros_like(means[seed]) if cfg.get('donor_difference') else means[seed] for seed in z}
            for rank in cfg['ranks']:
                b=f['source_basis'][lookup[s,a,t0],:,:rank].astype(np.float64)
                common={'source_seed':s,'source_atom':a,'rank':rank,'condition':e['condition'],'sequence':e['sequence']}
                source=participation(z[s][:,ids],active_means[s][ids],dec[s][ids].astype(np.float64)@b)
                record=dict(common,side='source',target_seed=None,**source)
                if args.source_only:
                    hook_norm=float(np.linalg.norm(shared_hook[pos].astype(np.float64)))
                    fraction=np.sqrt(source['aggregate_energy'])/hook_norm if hook_norm else None
                    cap=cfg.get('maximum_source_hook_fraction')
                    scale=min(1,cap/fraction) if cap and fraction else 1.0
                    record.update(natural_source_hook_fraction=fraction,source_dose_scale=scale,
                                  dosed_source_hook_fraction=fraction*scale if fraction is not None else None,
                                  recipient_hook_norm=hook_norm,missing_donor=missing_donor,
                                  supported=bool(not missing_donor and source['aggregate_energy']>0),
                                  document_ids=e['document_ids'],donor_document_ids=e.get('donor_document_ids',[]))
                out.append(record)
                for t in ([] if args.source_only else q['targets']):
                    w=f['query_target'][lookup[s,a,t],:,:rank].astype(np.float64)
                    out.append(dict(common,side='target',target_seed=t,**participation(z[t],active_means[t],dec[t].astype(np.float64)@w)))
    groups={}
    for r in out:groups.setdefault((r['condition'],r['side'],r['rank'],r['source_seed'],r['source_atom']),[]).append(r)
    summaries={};valid_query_counts={}
    fields=list(source)
    for condition,side,rank,_,_ in groups:
        label=f'{condition}/{side}/r{rank}'
        if label in summaries:continue
        selected=[g for key,g in groups.items() if key[:3]==(condition,side,rank)]
        summaries[label]={};valid_query_counts[label]={}
        for field in fields:
            values=[statistics.median(r[field] for r in g if r[field] is not None) for g in selected if any(r[field] is not None for r in g)]
            summaries[label][field]=statistics.median(values) if values else None
            valid_query_counts[label][field]=len(values)
    report={'metrics_raw_sha256':None if args.source_only else hashlib.sha256(raw.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'definition':'e_j=sum_position,rank(((z_j-mean_j)*decoder_j@factor)^2); effective=(sum e)^2/sum(e^2). Means from independent mean split.','aggregation':'Median within query/condition/side/rank then across queries. Source counted once per sequence, not per target.','scope':'Descriptive coordinate-energy participation and mean/dynamic decomposition, not necessity or semantic evidence.','summary':summaries}
    if cfg.get('donor_difference'):
        report['definition']='e_j=sum_position,rank(((z_recipient,j-z_donor,j)*decoder_j@factor)^2); effective=(sum e)^2/sum(e^2). Mean cancels exactly. Participation and cancellation ratios are invariant to the shared nonzero dose; aggregate_energy is natural unscaled energy.'
    report['valid_query_counts_by_field']=valid_query_counts
    if args.source_only:
        report['access_boundary']='No endpoint metric file read; only each query source codes/decoder rows and source_basis materialized. Surface/selection identity metadata parsed; no query_target factor or target-code rows for that query materialized. Shared recipient hook gives amplitude denominator.'
        report['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [run/'config.resolved.json',run/'selection.json',ROOT/cfg['surface_path'],ROOT/cfg['factors_path'],ROOT/cfg['source_census_path'],asset/'asset_manifest.json',Path(cfg['raw_hook_asset_dir'])/'raw_hook_manifest.json']}
    prefix='source_participation' if args.source_only else 'atom_participation'
    (run/f'{prefix}.raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out))
    (run/f'{prefix}.summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
