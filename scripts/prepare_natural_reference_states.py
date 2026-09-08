"""Capture an independently specified natural-context correspondence reference.

Prefix execution is checked against a full-model pass. No SAE or task outcome
selects rows. This cache can be built while independently controlled SAEs train.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args();cfg=json.loads(args.config.read_text())
    files=['scripts/prepare_natural_reference_states.py','scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=MultisiteWork(cfg,args.config,files);error=None
    try:
        import torch,transformers,psutil
        w.torch=torch;w.device=torch.device(cfg['device']);torch.set_num_threads(cfg['cpu_threads'])
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        w.environment=dict(python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
            numpy=np.__version__,device=str(w.device),threads=torch.get_num_threads(),initial_available_ram=psutil.virtual_memory().available)
        if psutil.virtual_memory().available<cfg.get('minimum_available_ram_bytes',0):raise MemoryError('Insufficient current free RAM for declared reference capture')
        path=w.checked(ROOT/cfg['reference_tokens'],'Natural validation reference, separate from SAE training','ODC-By-1.0')
        from ccad.artifacts import sha256
        if sha256(path)!=cfg['reference_tokens_sha256']:raise ValueError('Reference token identity changed')
        w.checked(ROOT/cfg['reference_document_records'],'Natural document/hash split provenance','ODC-By-1.0')
        w.checked(ROOT/'.aris/compute/local-r006b1-env-spec.json')
        tokens=np.memmap(path,dtype='<u2',mode='r').reshape(-1,cfg['context_length'])[:cfg['reference_sequences']].astype(np.int64)
        modeldir=Path(cfg['model_local_dir'])
        for name in ['config.json','model.safetensors']:w.checked(modeldir/name,'Pinned Pythia1B','Apache-2.0')
        model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').eval().to(w.device)
        model.config.use_cache=False
        for p in model.parameters():p.requires_grad_(False)
        contract=HookPointContract(f"gpt_neox.layers.{cfg['layer']}",cfg['layer'],'resid_post',model.config.hidden_size)
        capture=[];stop=True
        class PrefixCaptured(Exception):pass
        def hook(module,args,output):
            capture.append(extract_primary_hook_tensor(output,contract).detach().cpu().numpy().copy())
            if stop:raise PrefixCaptured()
        handle=model.get_submodule(contract.module_path).register_forward_hook(hook)
        chunks=[]
        try:
            for ids in w.batches(np.arange(len(tokens))):
                if time.perf_counter()-w.wall_start>cfg['budget_seconds']:raise TimeoutError('Reference capture wall budget exhausted')
                x=torch.as_tensor(tokens[ids],device=w.device);capture.clear()
                with torch.no_grad():
                    try:model.gpt_neox(input_ids=x,use_cache=False)
                    except PrefixCaptured:pass
                if len(capture)!=1:raise ValueError('Hook not captured exactly once')
                current=capture[0]
                if ids[0]==0:
                    stop=False;capture.clear()
                    with torch.no_grad():model.gpt_neox(input_ids=x,use_cache=False)
                    w.checks['early_stop_matches_full_model']=bool(np.array_equal(current,capture[0]));stop=True
                    if not w.checks['early_stop_matches_full_model']:raise ValueError('Prefix capture differs from full model')
                chunks.append(current)
                w.sequence_forwards+=len(ids);w.token_forwards+=int(x.numel())
                w.record(kind='reference_capture',task='natural',row_id=int(ids[0]),component='fixed_natural_reference',
                    mode='prefix_layer_'+str(cfg['layer']),method='raw_model',seed=0,target_seed=None,operation='capture',split='reference_fit',
                    sequences=len(ids),tokens=int(x.numel()))
                w.progress('NATURAL_REFERENCE_CAPTURE',completed=int(ids[-1]+1),total=len(tokens),rss_bytes=psutil.Process().memory_info().rss)
        finally:handle.remove()
        hidden=np.concatenate(chunks).reshape(-1,model.config.hidden_size);flat=tokens.reshape(-1)
        keep=~np.isin(flat,cfg['exclude_token_ids'])
        np.savez_compressed(w.run/'reference_states.npz',hidden=hidden[keep],tokens=flat[keep],packed_positions=np.flatnonzero(keep))
        write(w.run/'reference_identity.json',dict(layer=cfg['layer'],model_revision=cfg['model_revision'],
            token_sha256=cfg['reference_tokens_sha256'],sequences=len(tokens),eligible_tokens=int(keep.sum()),
            prefix_sequences=w.sequence_forwards,additional_full_witness_sequences=cfg['batch_size'],scope=cfg['scope']))
        w.checks['all_sequences_captured']=w.sequence_forwards==len(tokens)
    except Exception:error=traceback.format_exc()
    return w.finish(error)

if __name__=='__main__':raise SystemExit(main())
