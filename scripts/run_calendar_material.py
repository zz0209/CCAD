"""Bounded calendar-variable competence, raw patching and SAE material study.

All authored conditions are retained. SAE donor differences add to the original
recipient residual. This does not fit or evaluate a cross-seed FCC relation.
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', SPARSIFY_DISABLE_TRITON='1')
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.artifacts import sha256, validate_run_directory
from ccad.activation_contract import (HookPointContract, extract_primary_hook_tensor,
                                      replace_primary_hook_tensor)
import numpy as np


def make_prompts(cfg):
    rows, pairs = [], []
    for family in cfg['families']:
        values = family['values']
        for template in family['templates']:
            first = len(rows)
            for i, value in enumerate(values):
                rows.append(dict(id=len(rows), family=family['name'], template=template['id'],
                                 value=value, value_index=i, relation=template['relation'],
                                 expected_index=(i + template['relation']) % len(values),
                                 prefix=template['prefix'], suffix=template['suffix'],
                                 text=template['prefix'] + value + template['suffix'],
                                 label_source='authored controlled development'))
                rows[-1].update({key:template[key] for key in ['role','phase'] if key in template})
            for i in range(len(values)):
                for offset in family['donor_offsets']:
                    pairs.append(dict(pair_id=len(pairs), recipient=first+i,
                                      donor=first+(i+offset) % len(values), offset=offset))
    return rows, pairs


def logprob(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    run = ROOT / 'runs' / cfg['run_id']
    run.mkdir(exist_ok=False)
    start = time.perf_counter()
    write(run/'config.resolved.json', cfg)
    files = []
    for rel in ['scripts/run_calendar_material.py', 'scripts/run_r011s1_raw_hook_asset.py',
                'src/ccad/activation_contract.py', 'src/ccad/artifacts.py'] + (
                ['scripts/calendar_composition.py'] if cfg.get('composition') else []):
        p = ROOT/rel
        dst = run/'source_snapshot'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
        files.append(dict(path=rel, sha256=sha256(p), bytes=p.stat().st_size,
                          snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json', dict(files=files, aggregate_sha256=aggregate(files),
                                      snapshot_root='source_snapshot'))
    write(run/'manifest.json', dict(schema_version='calendar.material.v1', run_id=cfg['run_id'],
          run_parent='F4', purpose=cfg['purpose'], milestone='variable-material-development',
          evidence_level=('authored_conditional_composition_development' if cfg.get('composition') else 'authored_controlled_development_no_FCC_fit'),
          started_utc=datetime.now(timezone.utc).isoformat(), project_root=str(ROOT),
          config_hash=sha256(run/'config.resolved.json'), code_snapshot_hash=aggregate(files),
          source_snapshot_required=True, audit_opened=False, candidate_family_frozen=True,
          mean_constants_source_split='no centering fit; decoder bias cancels in donor differences',
          threshold_source_split='all authored cases and fixed donor offsets; no outcome selection',
          statistics_unit='values within templates and seed comparisons dependent', device='cuda:0',
          seeds=[1,2,3,4,5], resource_lease='gpu-0 resource_manager.run; CPU auxiliary one thread',
          resource_lease_reason=cfg['budget']))
    for name in ['stdout.log', 'stderr.log', 'metrics.raw.jsonl']:
        (run/name).touch()
    write(run/'status.json', dict(status='RUNNING'))
    inputs, metrics, checks, env = [], [], {}, {}
    error, sequence_forwards = None, 0

    def checked(path, expected=None):
        p = Path(path)
        p = p if p.is_absolute() else ROOT/p
        item = entry(p, 'Existing CCAD local model / same-stream SAE / authored config', 'input')
        if expected and item['sha256'] != expected:
            raise ValueError('Input identity changed: '+str(p))
        inputs.append(item)
        return p

    def load(path):
        return json.loads(checked(path).read_text(encoding='utf-8-sig'))

    def progress(stage, **kw):
        obj = dict(stage=stage, seconds=time.perf_counter()-start, **kw)
        write(run/'progress.json', obj)
        with (run/'stdout.log').open('a') as f:
            f.write(json.dumps(obj)+'\n')
        print(json.dumps(obj), flush=True)
        if obj['seconds'] > cfg['budget_seconds']:
            raise TimeoutError('Bounded calendar run budget exceeded')

    try:
        asset = load(cfg['model_asset_config'])
        cpconfig = load(cfg['sae_checkpoints_config'])
        load('.aris/compute/local-r006b1-env-spec.json')
        checked(args.config.resolve())
        sys.path[:0] = [asset['sparsify_source_dir'], asset['sparsify_overlay_dir']]
        import torch
        import transformers
        from sparsify import SparseCoder
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model_dtype = torch.float64 if cfg.get('model_dtype')=='float64' else torch.float32
        array_dtype = np.float64 if model_dtype==torch.float64 else np.float32
        torch.cuda.reset_peak_memory_stats()
        modeldir = Path(asset['model_local_dir'])
        modelcfg = load(modeldir/'config.json')
        for name in ['tokenizer.json', 'tokenizer_config.json']:
            checked(modeldir/name)
        # A single existing 160M weight file is small enough to hash once.
        for p in sorted(modeldir.glob('*.safetensors')):
            checked(p)
        for p in sorted(modeldir.glob('pytorch_model*.bin')):
            checked(p)
        tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir, local_files_only=True)
        rows, pairs = make_prompts(cfg)
        family = {r['name']:r for r in cfg['families']}
        candidate_ids = {}
        for name, spec in family.items():
            words = [tokenizer.encode(' '+v, add_special_tokens=False) for v in spec['values']]
            if any(len(v)!=1 for v in words):
                raise ValueError('Candidate needs multiple tokens: '+name)
            candidate_ids[name] = [x[0] for x in words]
        encoded = []
        for row in rows:
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            prefix = tokenizer.encode(row['prefix'].rstrip(), add_special_tokens=False)
            slot = len(prefix)
            if ids[:slot] != prefix or ids[slot] != candidate_ids[row['family']][row['value_index']]:
                raise ValueError('Marked single-token slot mismatch: '+row['text'])
            for value, cid in zip(family[row['family']]['values'], candidate_ids[row['family']]):
                if tokenizer.encode(row['text']+' '+value, add_special_tokens=False) != ids+[cid]:
                    raise ValueError('Continuation tokenization changes prefix')
            row.update(slot=slot, token_ids=ids, expected=family[row['family']]['values'][row['expected_index']])
            encoded.append(ids)
        for pair in pairs:
            a, b = rows[pair['recipient']], rows[pair['donor']]
            if (a['slot'] != b['slot'] or len(a['token_ids']) != len(b['token_ids']) or
                [i for i,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y] != [a['slot']]):
                raise ValueError('Donor differs outside the one marked value token')
        expected_prompts=sum(len(f['values'])*len(f['templates']) for f in cfg['families'])
        expected_pairs=sum(len(f['values'])*len(f['templates'])*len(f['donor_offsets']) for f in cfg['families'])
        checks.update(all_configured_prompts=len(rows)==expected_prompts, all_configured_pairs=len(pairs)==expected_pairs,
                      one_token_slot_and_continuations=True,
                      layer5_is_middle_of_12=modelcfg['num_hidden_layers']==12 and cfg['sae_layer']==5)
        write(run/'prompts_and_pairs.json', dict(rows=rows, pairs=pairs, candidate_ids=candidate_ids,
              fixed_before_model_forward=True, evidence='authored development; not natural audit'))
        last = np.array([len(x)-1 for x in encoded])
        width, n = int(max(last)+1), len(rows)
        tokens = np.full((n,width), tokenizer.eos_token_id, dtype=np.int64)
        attention = np.zeros_like(tokens)
        for i, ids in enumerate(encoded):
            tokens[i,:len(ids)], attention[i,:len(ids)] = ids, 1
        np.savez_compressed(run/'tokens.npz', tokens=tokens, attention=attention, last=last)
        model = transformers.AutoModelForCausalLM.from_pretrained(modeldir, local_files_only=True,
                    dtype=model_dtype, attn_implementation='eager').eval().to('cuda:0')
        modules = {l:model.get_submodule(f'gpt_neox.layers.{l}') for l in cfg['layers']}
        contracts = {l:HookPointContract(f'gpt_neox.layers.{l}', l, 'resid_post', 768) for l in modules}
        raw = {l:np.empty((n,width,768),dtype=array_dtype) for l in modules}

        def forward(ix, layer=None, deltas=None, replacements=None, capture=False, unpad=False):
            nonlocal sequence_forwards
            if time.perf_counter()-start > cfg['budget_seconds']:
                raise TimeoutError('Calendar compute budget exceeded')
            batch_width = int(last[ix[0]]+1) if unpad else width
            inp = torch.tensor(tokens[ix,:batch_width], device='cuda:0')
            mask = torch.tensor(attention[ix,:batch_width], device='cuda:0')
            pos = torch.tensor(last[ix], device='cuda:0')
            captured, handles = {}, []
            def hook_for(l):
                def hook(m, args, output):
                    h = extract_primary_hook_tensor(output, contracts[l])
                    if capture:
                        captured[l] = h.detach().cpu().numpy().copy()
                    if l != layer:
                        return output
                    h = h.clone()
                    if deltas is not None:
                        h += torch.tensor(deltas[:,:batch_width], device=h.device, dtype=h.dtype)
                    if replacements is not None:
                        data, patch_mask = replacements
                        d = torch.tensor(data[:,:batch_width], device=h.device, dtype=h.dtype)
                        m = torch.tensor(patch_mask[:,:batch_width], device=h.device)
                        h = torch.where(m[:,:,None], d, h)
                    return replace_primary_hook_tensor(output, h, contracts[l])
                return hook
            for l in modules:
                if capture or l==layer:
                    handles.append(modules[l].register_forward_hook(hook_for(l)))
            try:
                with torch.no_grad():
                    h = model.gpt_neox(inp, attention_mask=mask, use_cache=False).last_hidden_state
                    logits = model.get_output_embeddings()(h[torch.arange(len(ix),device='cuda:0'),pos])
                    lp = logprob(logits.cpu().numpy())
            finally:
                for handle in handles:
                    handle.remove()
            sequence_forwards += len(ix)
            return lp, captured

        batches = [np.arange(i,min(i+cfg['batch_size'],n)) for i in range(0,n,cfg['batch_size'])]
        base = []
        for ix in batches:
            lp, cap = forward(ix, capture=True)
            base.append(lp)
            for l in raw:
                raw[l][ix] = cap[l]
        base = np.concatenate(base)
        np.save(run/'baseline_logprobs.npy', base)
        np.savez_compressed(run/'raw_hooks.npz', **{f'layer{l}':h for l,h in raw.items()})
        ix = batches[0]
        noop,_ = forward(ix, layer=5, deltas=np.zeros_like(raw[5][ix]))
        checks['noop_exact'] = bool(np.array_equal(noop,base[ix]))
        padding_errors = []
        for i in sorted({int(np.argmin(last)),int(np.argmax(last))}):
            lp,_ = forward(np.array([i]),unpad=True)
            padding_errors.append(float(np.max(np.abs(lp-base[i]))))
        checks['padding_logprob_max_error_below_1e_4'] = max(padding_errors)<1e-4
        with torch.no_grad():
            public = model(torch.tensor(tokens[:1],device='cuda:0'),
                attention_mask=torch.tensor(attention[:1],device='cuda:0'),use_cache=False).logits[0,last[0]].cpu().numpy()
        sequence_forwards += 1
        public_error = float(np.max(np.abs(logprob(public)-base[0])))
        checks['public_forward_logprob_error_below_1e_4'] = public_error<1e-4
        baseline_rows = []
        for r,lp in zip(rows,base):
            ids = candidate_ids[r['family']]
            probs = np.exp(lp[ids])
            baseline_rows.append(dict(prompt_id=r['id'],family=r['family'],template=r['template'],
                relation=r['relation'],value=r['value'],expected=r['expected'],
                family_prediction=family[r['family']]['values'][int(np.argmax(probs))],
                full_prediction=tokenizer.decode([int(np.argmax(lp))]),
                family_correct=bool(np.argmax(probs)==r['expected_index']),
                full_correct=bool(np.argmax(lp)==ids[r['expected_index']]),
                candidate_probabilities=probs.tolist(),family_mass=float(sum(probs)),
                expected_probability=float(probs[r['expected_index']])))
        write(run/'baseline_results.json',dict(rows=baseline_rows,padding_errors=padding_errors,
              public_logprob_max_error=public_error))
        progress('baseline_captured', prompts=n, pairs=len(pairs), family_correct=sum(r['family_correct'] for r in baseline_rows))
        recipient = np.array([p['recipient'] for p in pairs])
        donor = np.array([p['donor'] for p in pairs])
        pbatches = [np.arange(i,min(i+cfg['batch_size'],len(pairs))) for i in range(0,len(pairs),cfg['batch_size'])]
        patch_masks = {}
        for mode in ['slot','suffix']:
            m = np.zeros((len(pairs),width),dtype=bool)
            for i,rid in enumerate(recipient):
                r = rows[rid]
                m[i,r['slot']:r['slot']+1 if mode=='slot' else last[rid]+1] = True
            patch_masks[mode] = m
        raw_reference = {}

        def record(lp, pair_index, layer, mode, method, seed=None, vector_error=None, reference=None):
            pair = pairs[pair_index]
            ri, di = pair['recipient'], pair['donor']
            r, d = rows[ri], rows[di]
            ids = candidate_ids[r['family']]
            dc, rc = ids[d['expected_index']], ids[r['expected_index']]
            ref = base[di] if reference is None else reference
            kl = lambda a,b: max(0.,float(np.sum(np.exp(a)*(a-b))))
            v = dict(**pair,family=r['family'],template=r['template'],relation=r['relation'],
                recipient_value=r['value'],donor_value=d['value'],expected_after=d['expected'],
                layer=layer,mode=mode,method=method,seed=seed,
                expected_probability=float(np.exp(lp[dc])),original_expected_probability=float(np.exp(lp[rc])),
                family_mass=float(np.exp(lp[ids]).sum()),candidate_probabilities=np.exp(lp[ids]).tolist(),
                family_donor_correct=bool(np.argmax(lp[ids])==d['expected_index']),
                full_donor_correct=bool(np.argmax(lp)==dc),
                donor_margin_change=float((lp[dc]-lp[rc])-(base[ri,dc]-base[ri,rc])),
                intervention_kl=kl(lp,base[ri]),donor_forward_kl=kl(base[di],lp),
                reference_kl=kl(ref,lp),noop_reference_kl=kl(ref,base[ri]),
                reference_margin_change=float((ref[dc]-ref[rc])-(base[ri,dc]-base[ri,rc])),
                vector_relative_squared_error=vector_error)
            metrics.append(v)
            with (run/'metrics.raw.jsonl').open('a') as f:
                f.write(json.dumps(v)+'\n')

        suffix_errors = {}
        for l in cfg['layers']:
            for mode, patch_mask in patch_masks.items():
                outputs = []
                max_error = 0.
                for ix in pbatches:
                    lp,_ = forward(recipient[ix],layer=l,replacements=(raw[l][donor[ix]],patch_mask[ix]))
                    if l==5:
                        outputs.append(lp)
                    for k,j in enumerate(ix):
                        record(lp[k],int(j),l,mode,'raw')
                    if mode=='suffix':
                        max_error = max(max_error,float(np.max(np.abs(lp-base[donor[ix]]))))
                if l==5:
                    raw_reference[mode] = np.concatenate(outputs)
                    np.save(run/f'raw_layer5_{mode}_logprobs.npy',raw_reference[mode])
                if mode=='suffix':
                    suffix_errors[str(l)] = max_error
                progress('raw_intervention',layer=l,mode=mode,rows=len(metrics))
        checks['full_suffix_matches_donor_forward_1e_4'] = max(suffix_errors.values())<1e-4
        # Encode actual nonpadding rows; pad codes remain zero and are never edited.
        keep = np.flatnonzero(attention.ravel())
        hflat = raw[5].reshape(-1,768)
        material = []
        for cp in cpconfig['checkpoints']:
            seed = cp['seed']
            weight = checked(Path(cp['path'])/'sae.safetensors',cp['sha256'])
            checked(weight.parent/'cfg.json')
            sae = SparseCoder.load_from_disk(weight.parent,device='cuda:0').eval()
            z = np.zeros((n*width,asset['num_latents']),dtype=np.float32)
            with torch.no_grad():
                for off in range(0,len(keep),256):
                    k = keep[off:off+256]
                    e = sae.encode(torch.tensor(hflat[k],device='cuda:0',dtype=torch.float32))
                    np.add.at(z,(k[:,None],e.top_indices.cpu().numpy()),e.top_acts.cpu().numpy())
                decoder = sae.W_dec.detach().cpu().numpy()
                reconstruction = (torch.tensor(z[keep],device='cuda:0') @ sae.W_dec).cpu().numpy()
            y = np.zeros_like(hflat)
            y[keep] = reconstruction
            y = y.reshape(n,width,768)
            np.savez_compressed(run/f'codes_seed{seed}.npz',codes=z.reshape(n,width,-1),decoder=decoder)
            # Decoder bias cancels; this variance error is a descriptive material score.
            centered_error = np.sum((hflat[keep]-hflat[keep].mean(0)-reconstruction+reconstruction.mean(0))**2)
            material.append(dict(seed=seed,mean_l0=float(np.count_nonzero(z[keep],axis=1).mean()),
                active_atoms=int(np.count_nonzero(np.any(z[keep]!=0,axis=0))),
                centered_reconstruction_fve=float(1-centered_error/np.sum((hflat[keep]-hflat[keep].mean(0))**2))))
            del sae
            for mode, patch_mask in patch_masks.items():
                for ix in pbatches:
                    delta = (y[donor[ix]]-y[recipient[ix]])*patch_mask[ix,:,None]
                    truth = (raw[5][donor[ix]]-raw[5][recipient[ix]])*patch_mask[ix,:,None]
                    verr = np.sum((delta-truth).astype(float)**2,axis=(1,2))/np.sum(truth.astype(float)**2,axis=(1,2))
                    lp,_ = forward(recipient[ix],layer=5,deltas=delta)
                    for k,j in enumerate(ix):
                        record(lp[k],int(j),5,mode,'sae_delta',seed,float(verr[k]),raw_reference[mode][j])
                progress('sae_intervention',seed=seed,mode=mode,rows=len(metrics))
        write(run/'material_scores.json',dict(rows=material,scope='authored task activations; no convergence claim'))
        write(run/'operation_checks.json',dict(full_suffix_logprob_max_errors=suffix_errors,
              padding_logprob_max_errors=padding_errors,public_logprob_max_error=public_error,
              operations='raw assigns donor states; SAE adds decoded donor-minus-recipient and preserves residual'))
        checks.update(all_operations=len(metrics)==len(pairs)*(len(cfg['layers'])*2+len(cpconfig['checkpoints'])*2),
                      finite_metrics=all(np.isfinite(r['intervention_kl']) for r in metrics))
        if cfg.get('composition'):
            from calendar_composition import run_composition
            extra = run_composition(cfg,run,rows,pairs,raw[5],base,candidate_ids,forward,progress)
            checks.update(extra['checks'])
            metrics.extend(extra['metrics'])
        env = dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
              torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
              peak_vram_bytes=torch.cuda.max_memory_allocated(),cpu_threads=torch.get_num_threads(),
              model_id=asset['model_id'],model_revision=asset['model_revision'],model_layers=modelcfg['num_hidden_layers'],
              sae_source_commit=asset['sparsify_commit'],dtype=f'{model_dtype} model; float32 locked SAE; float64 probability metrics')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (run/'stderr.log').write_text(traceback.format_exc())
    status = 'PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary = dict(status=status,error=error,checks=checks,rows=len(metrics),lm_sequences=sequence_forwards,
          wall_seconds=time.perf_counter()-start,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
          generator_script_path='scripts/run_calendar_material.py',
          generator_script_sha256=sha256(run/'source_snapshot/scripts/run_calendar_material.py'),
          scope=(cfg['composition']['scope'] if cfg.get('composition') else 'All authored cases; baseline competence and five SAE material preservation; no FCC fit or natural confirmation'))
    write(run/'environment.json',env)
    write(run/'inputs.json',dict(inputs=inputs))
    write(run/'metrics.summary.json',summary)
    write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    v = validate_run_directory(run)
    write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True)
    return 0 if status=='PASS' and v.ok else 1


if __name__=='__main__':
    raise SystemExit(main())
