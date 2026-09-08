"""Encode five controlled SAEs on a document-split natural correspondence set.

The existing quality-reference raw states are reused. Mixed-document chunks
are excluded and the nominal audit partition is neither encoded nor scored.
This is fitting material, not an independent quality or semantic result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import runpy
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork, ROOT, write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    files=['scripts/prepare_projected_family_reference.py','scripts/run_causalgym_multisite.py',
           'scripts/run_causalgym_native_transfer.py','scripts/run_r011s1_raw_hook_asset.py',
           'src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,files);error=None
    try:
        import torch,psutil
        from safetensors import safe_open
        w.torch=torch;w.device=torch.device('cpu');torch.set_num_threads(cfg['cpu_threads'])
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,
            numpy=np.__version__,device='cpu',cpu_threads=torch.get_num_threads(),available_ram=psutil.virtual_memory().available)
        ref=ROOT/cfg['reference_run']
        if json.loads(w.checked(ref/'status.json').read_text())['status']!='PASS':raise ValueError('Raw reference must pass')
        identity=json.loads(w.checked(ref/'reference_identity.json').read_text())
        if any(identity[k]!=cfg[k] for k in ['model_revision','layer']):raise ValueError('Reference model/hook differs')
        if identity['token_sha256']!=cfg['reference_tokens_sha256']:raise ValueError('Reference tokens differ')
        docs=json.loads(w.checked(ROOT/cfg['reference_document_records']).read_text())['documents']
        docs={r['document_id']:r for r in docs if r['split']=='validation'}
        seqs=json.loads(w.checked(ROOT/cfg['reference_sequence_records']).read_text())['sequences']
        seqs={r['sequence_index']:r for r in seqs if r['split']=='validation'}
        cache=np.load(w.checked(ref/'reference_states.npz'))
        # File decompression includes the existing full raw reference. Only the
        # selected mean/discovery/calibration rows enter encoding or fitting.
        packed=cache['packed_positions'];raw=cache['hidden'];tokens=cache['tokens']
        groups={};doc_splits={};mixed=0
        for i,pos in enumerate(packed):
            members=seqs[int(pos)//cfg['context_length']]['document_ids']
            if len(members)!=1:mixed+=1;continue
            doc=members[0];r=docs[doc]
            value=int(hashlib.sha256((cfg['reference_split_salt']+'|'+r['text_sha256']).encode()).hexdigest(),16)/(1<<256)
            split='mean' if value<.1 else 'discovery' if value<.5 else 'calibration' if value<.7 else 'audit'
            doc_splits[doc]=split;groups.setdefault(doc,[]).append(i)
        chosen=[];split_names=[];document_ids=[];inventory=[]
        for doc in sorted(groups):
            split=doc_splits[doc];all_rows=groups[doc]
            count=min(len(all_rows),cfg['maximum_tokens_per_document'])
            selected=sorted(all_rows,key=lambda i:hashlib.sha256((cfg['reference_split_salt']+'|token|'+str(int(packed[i]))).encode()).hexdigest())[:count]
            inventory.append(dict(document_id=doc,text_sha256=docs[doc]['text_sha256'],split=split,
                eligible_tokens=len(all_rows),selected_tokens=count if split!='audit' else 0))
            if split=='audit':continue
            for i in sorted(selected):chosen.append(i);split_names.append(split);document_ids.append(doc)
        selected=np.asarray(chosen,int);names=np.asarray(split_names)
        if not all(np.any(names==s) for s in ['mean','discovery','calibration']):raise ValueError('Empty natural fitting partition')
        raw_selected=raw[selected].copy();del raw,cache
        np.savez_compressed(w.run/'natural_rows.npz',hidden=raw_selected,packed_positions=packed[selected],
            tokens=tokens[selected],source_cache_rows=selected,split=names,document_ids=np.asarray(document_ids))
        write(w.run/'natural_partition.json',dict(documents=inventory,rows=len(selected),excluded_mixed_document_tokens=mixed,
            split_counts={s:int(np.sum(names==s)) for s in ['mean','discovery','calibration','audit']},
            split_rule='Text-document SHA256 with fixed salt;10/40/20/30 percent hash intervals. At most fixed tokens per document, selected by hash. Entire mixed-document128token contexts excluded.',
            audit_access='Full existing raw reference file decompressed; audit rows not encoded, fitted, scored or used for choice. Raw reference had already served natural quality validation; no new independent quality/audit claim.',
            purpose='Source-query-independent natural paired material, separate from SAE training and RAVEL task examples.'))
        kernel=runpy.run_path(str(w.checked(Path(cfg['sparsify_source'])/'sparsify/fused_encoder.py')))['fused_encoder']
        w.checked(ROOT/'.aris/compute/local-r006b1-env-spec.json')
        for seed in cfg['seeds']:
            folder=Path(cfg['sae_root'])/f'seed_{seed}'
            sc=json.loads(w.checked(folder/'cfg.json').read_text())
            if sc['transcode'] or sc['skip_connection']:raise ValueError('Requires ordinary SAE')
            with safe_open(w.checked(folder/'sae.safetensors'),framework='pt',device='cpu') as f:
                ew,eb,db=[f.get_tensor(k) for k in ['encoder.weight','encoder.bias','b_dec']]
            indices=np.empty((len(selected),sc['k']),np.uint16);values=np.empty(indices.shape,np.float32)
            for begin in range(0,len(selected),cfg['encode_batch_tokens']):
                if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Reference encoding budget exhausted')
                x=torch.as_tensor(raw_selected[begin:begin+cfg['encode_batch_tokens']])
                with torch.no_grad():act,ix,_=kernel(x-db,ew,eb,sc['k'],sc['activation'])
                indices[begin:begin+len(x)]=ix.numpy();values[begin:begin+len(x)]=act.numpy()
            np.savez_compressed(w.run/f'seed{seed}_natural_codes.npz',indices=indices,activations=values,width=ew.shape[0])
            w.checks[f'seed{seed}_finite_nonnegative']=bool(np.isfinite(values).all() and np.all(values>=0))
            w.record(kind='natural_encoding',task='natural',row_id=seed,component='five_seed_reference',method='topk_encoder',
                seed=seed,operation='encode',split='mean_discovery_calibration',rows=len(selected),width=ew.shape[0],k=sc['k'])
            w.progress('NATURAL_PAIRED_CODES_ENCODED',seed=seed,rows=len(selected),rss_bytes=psutil.Process().memory_info().rss)
            del ew,eb,db,indices,values
        w.checks['audit_codes_absent']=not np.any(names=='audit')
        w.checks['single_document_contexts_only']=all(len(seqs[int(packed[i])//cfg['context_length']]['document_ids'])==1 for i in selected)
        w.checks['five_controlled_seeds']=cfg['seeds']==[1,2,3,4,5]
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
