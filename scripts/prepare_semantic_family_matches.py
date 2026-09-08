"""SemanticOT for frozen semantic source gates on the exact current SAE group.

Reuses the verified natural hook-state reference, never another checkpoint's
codes or task-selected target identities. The retrieval kernel is the existing
Algorithm1 adaptation, with its full shortlist/solver evidence retained.
"""
from __future__ import annotations
import argparse
import json
import platform
import runpy
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write
from ccad.semantic_context_matching import top_distributions,retrieve,euclidean_cost


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    files=['scripts/prepare_semantic_family_matches.py','src/ccad/semantic_context_matching.py','src/ccad/nip_baselines.py',
           'scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_transfer.py',
           'scripts/run_r011s1_raw_hook_asset.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,files);error=None
    try:
        import torch,scipy,psutil
        from safetensors import safe_open
        w.torch=torch;w.device=torch.device('cpu');torch.set_num_threads(cfg['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
            numpy=np.__version__,scipy=scipy.__version__,device='cpu',threads=torch.get_num_threads(),
            initial_available_ram=psutil.virtual_memory().available)
        def progress(stage,**values):
            if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Declared natural retrieval budget exhausted')
            w.progress(stage,rss=psutil.Process().memory_info().rss,**values)
        ref=ROOT/cfg['reference_run']
        if json.loads(w.checked(ref/'status.json').read_text())['status']!='PASS':raise ValueError('Natural reference run must pass')
        identity=json.loads(w.checked(ref/'reference_identity.json').read_text())
        if any(identity[k]!=cfg[k] for k in ['model_revision','layer']):raise ValueError('Natural reference model/hook differs')
        if identity['token_sha256']!=cfg['reference_tokens_sha256']:raise ValueError('Natural reference token identity differs')
        refsummary=json.loads(w.checked(ref/'metrics.summary.json').read_text())
        if not refsummary['checks'].get('early_stop_matches_full_model'):raise ValueError('Natural prefix/full witness unavailable')
        reference=np.load(w.checked(ref/'reference_states.npz'))['hidden']
        if len(reference)!=identity['eligible_tokens']:raise ValueError('Natural state count differs')
        kernel=runpy.run_path(str(w.checked(Path(cfg['sparsify_source'])/'sparsify/fused_encoder.py')))['fused_encoder']
        distributions={};centroids={};counts={};widths={};source_ids={}
        for source in sorted({p[0] for p in cfg['seed_pairs']}):
            parent=ROOT/cfg['source_gate_runs'][str(source)]
            parent_cfg=json.loads(w.checked(parent/'config.resolved.json').read_text())
            if json.loads(w.checked(parent/'status.json').read_text())['status']!='PASS':raise ValueError('Source gate producer must pass')
            if parent_cfg['source_seed']!=source or any(parent_cfg[k]!=cfg[k] for k in ['sae_root','model_revision','layer']):
                raise ValueError('Source gate checkpoint or model identity differs')
            gates=np.load(w.checked(parent/cfg['source_gate_filename']))['gates']
            source_ids[source]=np.flatnonzero(np.any(gates>0,axis=1)).tolist()
            if not source_ids[source]:raise ValueError('Source gate family is empty')
        for seed in cfg['seeds']:
            folder=Path(cfg['sae_root'])/f'seed_{seed}'
            sae_cfg=json.loads(w.checked(folder/'cfg.json').read_text())
            if sae_cfg['transcode'] or sae_cfg['skip_connection']:raise ValueError('Requires ordinary residual SAE')
            with safe_open(w.checked(folder/'sae.safetensors'),framework='pt',device='cpu') as f:
                ew,eb,db=[f.get_tensor(k) for k in ['encoder.weight','encoder.bias','b_dec']]
            width=ew.shape[0];widths[seed]=width
            if width>65536:raise ValueError('uint16 sparse reference index range exceeded')
            indices=np.empty((len(reference),sae_cfg['k']),np.uint16);activations=np.empty(indices.shape,np.float32)
            with torch.no_grad():
                for offset in range(0,len(reference),cfg['encode_batch_tokens']):
                    x=torch.as_tensor(reference[offset:offset+cfg['encode_batch_tokens']])-db
                    act,ix,_=kernel(x,ew,eb,sae_cfg['k'],sae_cfg['activation'])
                    indices[offset:offset+len(x)]=ix.numpy();activations[offset:offset+len(x)]=act.numpy()
            del ew,eb,db
            np.savez_compressed(w.run/f'seed{seed}_reference_codes.npz',indices=indices,activations=activations)
            distributions[seed],centroids[seed],counts[seed]=top_distributions(indices,activations,width,cfg['top_k'],reference)
            np.savez_compressed(w.run/f'seed{seed}_context_distributions.npz',centroids=centroids[seed],positive_counts=counts[seed])
            del indices,activations
            progress('REFERENCE_CODES_COMPLETE',seed=seed,active=int((counts[seed]>0).sum()),eligible=int((counts[seed]>=cfg['minimum_activations']).sum()))
        sample=np.random.default_rng(cfg['distance_scale_seed']).choice(len(reference),size=min(256,len(reference)),replace=False)
        distances=euclidean_cost(reference[sample].astype(float),reference[sample].astype(float))
        scale=float(np.median(distances[np.triu_indices(len(sample),1)]));epsilon=scale*cfg['sinkhorn_scale_fraction']
        for source,target in cfg['seed_pairs']:
            def record(row):
                w.record(kind='natural_matching',task='semantic_family',row_id=row['source_member'],component=f"feature_{source}_{row['source_member']}",
                    mode='same_token_reference',method='semanticot',seed=source,target_seed=target,operation='match',split='natural_reference',**row)
                progress('FEATURE_MATCH_COMPLETE',source_seed=source,target_seed=target,source_member=row['source_member'],status=row['status'])
            matched=retrieve(source_ids[source],distributions[source],distributions[target],centroids[source],centroids[target],
                counts[source],counts[target],reference,candidate_count=cfg['centroid_candidates'],minimum_count=cfg['minimum_activations'],
                regularization=epsilon,tolerance=cfg['sinkhorn_tolerance'],max_iterations=cfg['sinkhorn_max_iterations'],progress=record,dual_fallback=cfg['dual_fallback'])
            write(w.run/f's{source}_t{target}_matches.json',dict(source_seed=source,target_seed=target,source_ids=source_ids[source],rows=matched,
                reference_tokens=len(reference),reference_run=cfg['reference_run'],top_k=cfg['top_k'],regularization=epsilon,
                reference_distance_scale=scale,ground_cost=f"Euclidean in common same-token layer{cfg['layer']} residual space",
                unmatched_policy='Retain every source member; unmatched target contribution is zero and coverage is reported',scope=cfg['scope']))
            w.checks[f'all_members_s{source}_t{target}']=len(matched)==len(source_ids[source])
            w.checks[f'support_s{source}_t{target}']=all(r['reference_activations']>=cfg['minimum_activations'] for r in matched if r['status']=='MATCHED')
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
