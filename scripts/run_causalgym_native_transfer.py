"""Actual intermediate-layer cross-SAE operations on released CausalGym train.

This is exploratory transfer, with prompt-connected components kept together.
The test split is never opened. All interventions run the complete model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                  MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import numpy as np

from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256, validate_run_directory
from ccad.factor_correspondence import compact_source, source_from_support, ridge, choose_ridge, group_support, assigned_readout
from ccad.native_operation import writable_support, project_native, adaptive_writable_support


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare_panel(data, cfg):
    """Remove reciprocal duplicates and split components sharing any prompt."""
    panel, inventory = [], {}
    for task in cfg['tasks']:
        unique = {}
        for index, row in enumerate(data):
            if row['task'] != task:
                continue
            sides = sorted([(tuple(row[s]), row[s+'_label'], row[s+'_type']) for s in ['base', 'src']])
            key = digest(sides)
            if key not in unique:
                unique[key] = dict(key=key, sides=sides, dataset_indices=[])
            unique[key]['dataset_indices'].append(index)
        pairs = list(unique.values())
        parents = list(range(len(pairs)))
        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        owners = {}
        for i, pair in enumerate(pairs):
            for spans, _, _ in pair['sides']:
                text = ''.join(spans)
                if text in owners:
                    parents[root(i)] = root(owners[text])
                else:
                    owners[text] = i
        components = {}
        for i, pair in enumerate(pairs):
            components.setdefault(root(i), []).append(pair)
        groups = []
        for component in components.values():
            group_id = digest(sorted(pair['key'] for pair in component))
            groups.append((group_id, sorted(component, key=lambda x: x['key'])))
        groups.sort(key=lambda x: digest([cfg['split_salt'], task, x[0]]))
        selected = groups[:cfg['components_per_task']]
        if len(selected) < 10:
            raise ValueError(f'{task}: too few prompt-disjoint components for this pilot: {len(selected)}')
        nfit = int(len(selected) * .6)
        ncal = max(2, int(len(selected) * .15))
        records = []
        for number, (group_id, pairs_in_group) in enumerate(selected):
            split = 'fit' if number < nfit else 'calibration' if number < nfit+ncal else 'held_component_development'
            for pair in pairs_in_group[:cfg['pairs_per_component']]:
                first_id = len(panel)
                for side in range(2):
                    spans, label, kind = pair['sides'][side]
                    other_spans, other_label, other_kind = pair['sides'][1-side]
                    different = [j for j, (a, b) in enumerate(zip(spans, other_spans)) if a != b]
                    if len(spans) != len(other_spans) or not different:
                        raise ValueError('Expected aligned, distinct minimal pairs')
                    panel.append(dict(row_id=len(panel), task=task, component=group_id, pair_key=pair['key'],
                                      side=side, donor_id=first_id+1-side, spans=list(spans), text=''.join(spans),
                                      label=label, donor_label=other_label, input_type=kind, donor_type=other_kind,
                                      sign=1 if other_kind == sorted([kind, other_kind])[1] else -1,
                                      changed_region=max(different), penultimate_region=len(spans)-2,
                                      split=split, dataset_indices=pair['dataset_indices']))
                records.append(dict(pair_key=pair['key'], component=group_id, split=split,
                                    dataset_indices=pair['dataset_indices']))
        inventory[task] = dict(original_rows=sum(r['task']==task for r in data), unique_pairs=len(unique),
                               connected_components=len(groups), selected_components=len(selected),
                               retained_pairs=len(records), selected=records,
                               rule='Hash-ordered prompt-connected components; at most fixed pairs per component; reciprocal directions kept in one component')
    prompt_splits = {}
    for row in panel:
        prompt_splits.setdefault((row['task'], row['text']), set()).add(row['split'])
    if any(len(s) != 1 for s in prompt_splits.values()):
        raise ValueError('A prompt crossed fitting and evaluation components')
    return panel, inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    run = ROOT/'runs'/cfg['run_id']
    run.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc)
    clock_start = time.perf_counter()
    process_start = time.process_time()
    write(run/'config.resolved.json', cfg)
    sources = ['scripts/run_causalgym_native_transfer.py', 'scripts/run_r011s1_raw_hook_asset.py',
               'src/ccad/factor_correspondence.py', 'src/ccad/native_operation.py',
               'src/ccad/activation_contract.py', 'src/ccad/artifacts.py', 'src/ccad/nip_baselines.py']
    if cfg.get('semantic_match_run'):
        sources += ['scripts/prepare_f4_global_matching.py', 'scripts/run_f4_source_reference_causal.py']
    code = []
    for relative in sources:
        path = ROOT/relative
        snapshot = run/'source_snapshot'/relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        code.append(dict(path=relative, sha256=sha256(path), bytes=path.stat().st_size,
                         snapshot_path='source_snapshot/'+relative))
    write(run/'code_hashes.json', dict(files=code, aggregate_sha256=aggregate(code), snapshot_root='source_snapshot'))
    write(run/'manifest.json', dict(schema_version='causalgym.native.transfer.v1', run_id=cfg['run_id'],
          run_parent='FINAL_FIVE_R14', purpose=cfg['purpose'], milestone='external-nonfinal-native-operation',
          evidence_level='controlled_development', started_utc=started.isoformat(),
          started_local=started.astimezone().isoformat(), project_root=str(ROOT),
          config_hash=sha256(run/'config.resolved.json'), code_snapshot_hash=aggregate(code), source_snapshot_required=True,
          git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          automation_id='ccad', trigger='current conversation continuation', audit_opened=False, candidate_family_frozen=True,
          mean_constants_source_split='no intercept; paired differences cancel one common fixed mean',
          threshold_source_split='configuration before external outputs',
          statistics_unit='prompt-connected components, reciprocal pairs and shared SAE seeds; development only',
          device=cfg['device'], seeds=sorted({s for pair in cfg['seed_pairs'] for s in pair}),
          resource_lease='cpu-heavy' if cfg['device']=='cpu' else 'gpu-0', resource_lease_reason=cfg['budget'],
          model=cfg['model_local_dir'], model_revision=cfg['model_revision'], dataset_revision=cfg['dataset_revision'],
          sae_framework='Sparsify 42c064525b1cdd2b97f4a4807e247e89025d552c', protocol_deviations=[]))
    for filename in ['stdout.log', 'stderr.log', 'metrics.raw.jsonl']:
        (run/filename).touch()
    write(run/'status.json', dict(status='RUNNING'))
    inputs, metrics, checks, environment = [], [], {}, {}
    fits, projections = [], []
    sequence_forwards = 0
    token_forwards = 0
    error = None
    def checked(path, source='Pinned existing CCAD asset', boundary='internal'):
        p = Path(path)
        inputs.append(entry(p, source, 'actual input', boundary))
        write(run/'inputs.json', dict(inputs=inputs))
        return p
    def record(row):
        row = dict(run_id=cfg['run_id'], metric_version='causalgym-native-v1', **row)
        with (run/'metrics.raw.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        metrics.append(row)
    def progress(stage, **values):
        message = dict(stage=stage, written_at_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed=time.perf_counter()-clock_start, cpu_seconds=time.process_time()-process_start,
                       sequence_forwards=sequence_forwards, token_forwards=token_forwards, **values)
        write(run/'progress.json', message)
        print(json.dumps(message), flush=True)
    try:
        import torch
        import transformers
        from safetensors import safe_open
        torch.set_num_threads(cfg['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        device = cfg['device']
        environment = dict(python=sys.executable, python_version=platform.python_version(), os=platform.platform(),
                           numpy=np.__version__, torch=torch.__version__, transformers=transformers.__version__,
                           cpu_threads=torch.get_num_threads(), cuda_runtime=torch.version.cuda,
                           gpu='not_used' if device=='cpu' else torch.cuda.get_device_name(),
                           sae_framework='Sparsify 42c0645', device=device)
        write(run/'environment.json', environment)
        # The upstream SparseCoder class evaluates a CUDA capability decorator
        # during import. Load its unchanged, device-agnostic encoding kernel
        # directly so a CPU run never initializes a CUDA context.
        encoder_kernel = checked(Path(cfg['sparsify_source'])/'sparsify/fused_encoder.py',
                                 'Sparsify 42c0645 fused_encoder actually executed', 'MIT')
        checked(Path(cfg['sparsify_source'])/'sparsify/sparse_coder.py', 'Sparsify 42c0645 encode preprocessing reference', 'MIT')
        encode_kernel = runpy.run_path(str(encoder_kernel))['fused_encoder']
        dataset = checked(Path(cfg['dataset_dir'])/'train.json', 'aryaman/causalgym '+cfg['dataset_revision'], 'MIT task data')
        checked(Path(cfg['dataset_dir'])/'README.md', 'Official pinned dataset card', 'MIT')
        panel, inventory = prepare_panel(json.loads(dataset.read_text()), cfg)
        write(run/'selection_inventory.json', inventory)
        checked(ROOT/'.aris/compute/local-r006b1-env-spec.json')
        modeldir = Path(cfg['model_local_dir'])
        for filename in ['config.json', 'tokenizer.json', 'model.safetensors']:
            checked(modeldir/filename, 'EleutherAI/pythia-1b-deduped '+cfg['model_revision'], 'Apache-2.0')
        tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir, local_files_only=True, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.AutoModelForCausalLM.from_pretrained(modeldir, local_files_only=True, trust_remote_code=False,
                    dtype=torch.float32, attn_implementation='eager').eval().to(device)
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        contract = HookPointContract(f"gpt_neox.layers.{cfg['layer']}", cfg['layer'], 'resid_post', model.config.hidden_size)
        module = model.get_submodule(f"gpt_neox.layers.{cfg['layer']}")
        label_ids = {}
        positions = []
        for row in panel:
            tokens = tokenizer.encode(row['text'], add_special_tokens=False)
            ends, prefix = [], ''
            for span in row['spans']:
                prefix += span
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                if prefix_tokens != tokens[:len(prefix_tokens)]:
                    raise ValueError('Region boundary retokenized a prefix')
                ends.append(len(prefix_tokens)-1)
            for label in [row['label'], row['donor_label']]:
                ids = tokenizer.encode(label, add_special_tokens=False)
                if len(ids) != 1 or tokenizer.encode(row['text']+label, add_special_tokens=False) != tokens+ids:
                    raise ValueError('Continuation is not a single concatenative token')
                label_ids[label] = ids[0]
            sites = [ends[row['changed_region']], ends[row['penultimate_region']]]
            if any(site < 0 or site >= len(tokens)-1 for site in sites):
                raise ValueError('Configured nonfinal site is empty or final')
            positions.append(sites)
            row.update(tokens=tokens, region_ends=ends, positions=sites)
        positions = np.asarray(positions, dtype=int)
        write(run/'panel.json', dict(rows=panel, label_ids=label_ids, sites=cfg['sites'], dataset_split='train'))
        checks['all_sites_nonfinal'] = True
        checks['prompt_disjoint_components'] = True
        donors = np.asarray([r['donor_id'] for r in panel])
        dim = model.config.hidden_size
        hidden = np.empty((len(panel), len(cfg['sites']), dim), dtype=np.float32)
        def forward(row_ids, deltas=None, slot=0, capture=False, gradient=False, differentiable=False):
            nonlocal sequence_forwards, token_forwards
            if time.perf_counter()-clock_start > cfg['budget_seconds']:
                raise TimeoutError('Measured external-task wall budget exhausted')
            enc = tokenizer([panel[i]['text'] for i in row_ids], add_special_tokens=False, padding=True, return_tensors='pt').to(device)
            ix = torch.arange(len(row_ids), device=device)
            last = enc.attention_mask.sum(1)-1
            captured = []
            leaf = torch.zeros((len(row_ids), dim), device=device, requires_grad=gradient)
            def hook(mod, args, output):
                h = extract_primary_hook_tensor(output, contract)
                if capture:
                    captured.append(h[ix[:, None], torch.as_tensor(positions[row_ids], device=device)].detach().cpu().numpy().copy())
                if deltas is not None:
                    h = h.clone()
                    update = torch.as_tensor(deltas, device=device, dtype=h.dtype)
                    if gradient:
                        update = update+leaf
                    h[ix, torch.as_tensor(positions[row_ids, slot], device=device)] += update
                    return replace_primary_hook_tensor(output, h, contract)
                return output
            handle = module.register_forward_hook(hook)
            try:
                with torch.set_grad_enabled(gradient or differentiable):
                    out = model.gpt_neox(**enc, use_cache=False).last_hidden_state
                    logits = model.get_output_embeddings()(out[ix, last])
                    live_logprobs = torch.log_softmax(logits, dim=-1)
                    logprobs = live_logprobs.detach().cpu().numpy().astype(np.float64)
                    if gradient:
                        donor_labels = torch.as_tensor([label_ids[panel[i]['donor_label']] for i in row_ids], device=device)
                        base_labels = torch.as_tensor([label_ids[panel[i]['label']] for i in row_ids], device=device)
                        objective = (logits[ix, donor_labels]-logits[ix, base_labels]).sum()
                        derivative = torch.autograd.grad(objective, leaf)[0].detach().cpu().numpy().astype(np.float64)
            finally:
                handle.remove()
            sequence_forwards += len(row_ids)
            token_forwards += int(enc.attention_mask.sum())
            return (live_logprobs if differentiable else logprobs), derivative if gradient else captured[0] if captured else None
        baseline = []
        bs = cfg['batch_size']
        for offset in range(0, len(panel), bs):
            ids = np.arange(offset, min(offset+bs, len(panel)))
            lp, cap = forward(ids, capture=True)
            baseline.append(lp)
            hidden[ids] = cap
            if offset % (bs*4) == 0:
                progress('CAPTURE', completed=offset+len(ids), total=len(panel))
        baseline = np.concatenate(baseline)
        np.savez_compressed(run/'raw_cache.npz', hidden=hidden, positions=positions, baseline_logprobs=baseline.astype(np.float32))
        probe_ids = np.arange(min(bs, len(panel)))
        unchanged, cap = forward(probe_ids, np.zeros((len(probe_ids), dim), np.float32), capture=True)
        checks['same_batch_noop_exact'] = bool(np.array_equal(unchanged, baseline[probe_ids]))
        checks['captured_positions_exact'] = bool(np.array_equal(cap, hidden[probe_ids]))
        for i, row in enumerate(panel):
            a, b = label_ids[row['label']], label_ids[row['donor_label']]
            record(dict(kind='baseline', task=row['task'], component=row['component'], row_id=i, pair_key=row['pair_key'],
                        split=row['split'], correct=bool(baseline[i, a]>baseline[i, b]),
                        base_oriented_margin=float(baseline[i, a]-baseline[i, b])))
        assets = {}
        for seed in sorted({s for pair in cfg['seed_pairs'] for s in pair}):
            checkpoint = Path(cfg['sae_root'])/f'seed_{seed}'
            checked(checkpoint/'sae.safetensors')
            checked(checkpoint/'cfg.json')
            saecfg = json.loads((checkpoint/'cfg.json').read_text())
            if saecfg['transcode'] or saecfg['skip_connection']:
                raise ValueError('This native consumer requires the ordinary SAE checkpoint class')
            with safe_open(checkpoint/'sae.safetensors', framework='pt', device=device) as weights_file:
                enc_weight = weights_file.get_tensor('encoder.weight')
                enc_bias = weights_file.get_tensor('encoder.bias')
                dec_bias = weights_file.get_tensor('b_dec')
                dec_weight = weights_file.get_tensor('W_dec')
            with torch.no_grad():
                flat = torch.as_tensor(hidden.reshape(-1, dim), device=device)
                act, indices, _ = encode_kernel(flat-dec_bias, enc_weight, enc_bias, saecfg['k'], saecfg['activation'])
                codes = torch.zeros((len(flat), enc_weight.shape[0]), device=device).scatter_(1, indices, act).cpu().numpy().reshape(len(panel), len(cfg['sites']), -1)
                decoder = dec_weight.cpu().numpy().astype(np.float64)
            np.savez_compressed(run/f'seed{seed}_codes.npz', codes=codes)
            assets[seed] = dict(codes=codes.astype(np.float64), decoder=decoder)
            del enc_weight, enc_bias, dec_bias, dec_weight, flat, act, indices
        progress('MATERIAL_READY', rows=len(panel), tasks=len(cfg['tasks']))
        semantic_matches = None
        global_matches = {}
        if cfg.get('semantic_match_run'):
            semantic_parent = ROOT/cfg['semantic_match_run']
            if json.loads(checked(semantic_parent/'status.json').read_text())['status']!='PASS':
                raise ValueError('Natural-context retrieval did not complete')
            semantic_matches = json.loads(checked(semantic_parent/'matches.json').read_text())
            checked(semantic_parent/'config.resolved.json')
            from prepare_f4_global_matching import full_assignment
            for source_seed,target_seed in cfg['seed_pairs']:
                if (source_seed,target_seed)!=(semantic_matches['source_seed'],semantic_matches['target_seed']):
                    raise ValueError('Natural retrieval belongs to another dictionary pair')
                mapping,scale,diag = full_assignment(assets[source_seed]['decoder'],assets[target_seed]['decoder'])
                global_matches[source_seed,target_seed] = mapping
                np.savez_compressed(run/f'global_assignment_s{source_seed}_t{target_seed}.npz',target_indices=mapping,geometric_scale=scale)
                fits.append(dict(kind='full_dictionary_hungarian',source_seed=source_seed,target_seed=target_seed,**diag))
        def measure(task, site, source_seed, target_seed, method, row_ids, delta, reference=None, source_delta=None, **budget):
            slot = cfg['sites'].index(site)
            outputs = []
            for offset in range(0, len(row_ids), bs):
                local = np.arange(offset, min(offset+bs, len(row_ids)))
                batch_ids = row_ids[local]
                lp, _ = forward(batch_ids, delta[local].astype(np.float32), slot)
                outputs.append(lp)
                for j, local_id in enumerate(local):
                    i = int(row_ids[local_id])
                    row = panel[i]
                    a, b = label_ids[row['label']], label_ids[row['donor_label']]
                    margin = float(lp[j, b]-lp[j, a])
                    base_margin = float(baseline[i, b]-baseline[i, a])
                    ref = None if reference is None else reference[local_id]
                    kl = None if ref is None else max(0., float(np.sum(np.exp(ref)*(ref-lp[j]))))
                    record(dict(kind='intervention', task=task, site=site, source_seed=source_seed, target_seed=target_seed,
                                method=method, row_id=i, pair_key=row['pair_key'], component=row['component'], split=row['split'],
                                donor_oriented_margin=margin, donor_oriented_change=margin-base_margin,
                                donor_label_correct=bool(margin>0), base_correct=bool(base_margin<0),
                                kl_to_source=kl, edit_norm=float(np.linalg.norm(delta[local_id])),
                                source_squared_error=None if source_delta is None else float(np.sum((delta[local_id]-source_delta[local_id])**2)),
                                source_energy=None if source_delta is None else float(np.sum(source_delta[local_id]**2)), **budget))
            return np.concatenate(outputs)
        for task in cfg['tasks']:
            task_ids = np.asarray([i for i, row in enumerate(panel) if row['task']==task])
            fit = np.asarray([panel[i]['split']=='fit' for i in task_ids])
            cal = np.asarray([panel[i]['split']=='calibration' for i in task_ids])
            discovery = fit | cal
            evaluation = np.flatnonzero(~discovery)
            eval_ids = task_ids[evaluation]
            sign = np.asarray([panel[i]['sign'] for i in task_ids])
            for slot, site in enumerate(cfg['sites']):
                dh = (hidden[donors[task_ids], slot]-hidden[task_ids, slot]).astype(np.float64)
                raw_lp = measure(task, site, 0, 0, 'raw_donor', eval_ids, dh[evaluation])
                noop_lp = baseline[eval_ids]
                for source_seed, target_seed in cfg['seed_pairs']:
                    s, t = assets[source_seed], assets[target_seed]
                    dzs = s['codes'][donors[task_ids], slot]-s['codes'][task_ids, slot]
                    dzt = t['codes'][donors[task_ids], slot]-t['codes'][task_ids, slot]
                    # The source operation is frozen from source codes and labels before target fitting.
                    geometric_source = compact_source(dzs, s['decoder'], discovery, sign, cfg['source_budget'])
                    source = geometric_source
                    full_source = dzs@s['decoder']
                    source_selection = dict(method='geometric oriented contribution', source_only=True)
                    if cfg.get('source_selector') == 'integrated_gradient':
                        training_rows = np.flatnonzero(discovery)
                        mean_gradient = np.zeros((len(training_rows), dim), dtype=np.float64)
                        for point in cfg['ig_points']:
                            for offset in range(0, len(training_rows), bs):
                                local = np.arange(offset, min(offset+bs, len(training_rows)))
                                ids = training_rows[local]
                                _, grad = forward(task_ids[ids], point*full_source[ids], slot, gradient=True)
                                mean_gradient[local] += grad/len(cfg['ig_points'])
                        attribution = dzs[discovery]*(mean_gradient@s['decoder'].T)
                        complete_changes = []
                        for offset in range(0, len(training_rows), bs):
                            ids = training_rows[offset:offset+bs]
                            path_lp, _ = forward(task_ids[ids], full_source[ids], slot)
                            for j, local_id in enumerate(ids):
                                row_id = task_ids[local_id]
                                a, b = label_ids[panel[row_id]['label']], label_ids[panel[row_id]['donor_label']]
                                complete_changes.append(float((path_lp[j,b]-path_lp[j,a])-(baseline[row_id,b]-baseline[row_id,a])))
                        completeness_rmse = float(np.sqrt(np.mean((attribution.sum(axis=1)-np.asarray(complete_changes))**2)))
                        checks[f'finite_source_attribution_{task}_{site}_{source_seed}'] = bool(np.isfinite(attribution).all())
                        score = np.mean(np.abs(attribution), axis=0)
                        selected_source = np.argsort(-score, kind='stable')[:cfg['source_budget']]
                        source = source_from_support(dzs, s['decoder'], selected_source, discovery)
                        source_selection = dict(method='mean absolute source-code integrated-gradient attribution',
                                                source_only=True, points=cfg['ig_points'], source_budget=cfg['source_budget'],
                                                objective='donor minus base next-token logit along complete-source-SAE native donor path',
                                                endpoint_supervision='true task label pair on discovery components only',
                                                midpoint_completeness_rmse=completeness_rmse,
                                                complete_source_effect_mean=float(np.mean(complete_changes)),
                                                scores=score[selected_source].tolist())
                        np.savez_compressed(run/f'{task}_{site}_s{source_seed}_source_attribution.npz',
                                            training_row_ids=task_ids[training_rows], mean_gradient=mean_gradient,
                                            per_member_attribution=attribution, complete_source_effect=np.asarray(complete_changes),
                                            scores=score, selected=selected_source)
                    y = source['coordinates']
                    teacher = y@source['basis'].T
                    ref = measure(task, site, source_seed, target_seed, 'source_native', eval_ids, teacher[evaluation],
                                  reference=raw_lp, source_delta=teacher[evaluation], read_budget=cfg['source_budget'],
                                  write_budget=cfg['source_budget'], reference_identity='raw_donor')
                    if cfg.get('source_selector') == 'integrated_gradient':
                        for name, delta in [('source_geometric_same_budget', geometric_source['coordinates']@geometric_source['basis'].T),
                                            ('source_full_sae_donor', full_source),
                                            ('source_native_atom', dzs[:,source['support'][:1]]@s['decoder'][source['support'][:1]])]:
                            measure(task, site, source_seed, target_seed, name, eval_ids, delta[evaluation],
                                    reference=raw_lp, source_delta=teacher[evaluation], reference_identity='raw_donor')
                    # Score no-op against the same source response without an unnecessary model replay.
                    for j, i in enumerate(eval_ids):
                        row = panel[i]
                        a, b = label_ids[row['label']], label_ids[row['donor_label']]
                        record(dict(kind='intervention', task=task, site=site, source_seed=source_seed, target_seed=target_seed,
                                    method='noop', row_id=int(i), pair_key=row['pair_key'], component=row['component'], split=row['split'],
                                    donor_oriented_margin=float(noop_lp[j,b]-noop_lp[j,a]), donor_oriented_change=0.,
                                    donor_label_correct=bool(noop_lp[j,b]>noop_lp[j,a]), base_correct=bool(noop_lp[j,b]<noop_lp[j,a]),
                                    kl_to_source=max(0.,float(np.sum(np.exp(ref[j])*(ref[j]-noop_lp[j])))), edit_norm=0.,
                                    source_squared_error=float(np.sum(teacher[evaluation[j]]**2)), source_energy=float(np.sum(teacher[evaluation[j]]**2))))
                    predictions, weights, diagnostics = {}, {}, {}
                    alpha, errors = choose_ridge(dzt, y, fit, cal, cfg['ridge_alphas'], 'native_units')
                    full = ridge(dzt[discovery], y[discovery], alpha, 'native_units')
                    predictions['full'] = dzt@full@source['basis'].T
                    weights['full'] = full
                    diagnostics['full'] = dict(alpha=alpha, calibration_mse=errors, read_budget=dzt.shape[1])
                    selected, group_diag = group_support(dzt[discovery], y[discovery], cfg['read_budget'], iterations=cfg['lasso_steps'], scaling='native_units')
                    rms = np.sqrt(np.mean(dzt[discovery]**2, axis=0))
                    dense = np.argsort(-rms*np.linalg.norm(full, axis=1), kind='stable')[:cfg['read_budget']]
                    for name, members in [('fcc', selected), ('dense', dense)]:
                        a, errors = choose_ridge(dzt[:,members], y, fit, cal, cfg['ridge_alphas'], 'native_units')
                        w = ridge(dzt[discovery][:,members], y[discovery], a, 'native_units')
                        predictions[name] = dzt[:,members]@w@source['basis'].T
                        weights[name+'_members'], weights[name+'_weights'] = members, w
                        diagnostics[name] = dict(alpha=a, calibration_mse=errors, members=members.tolist(),
                                                 actual_read_budget=len(members), selection=group_diag if name=='fcc' else 'dense RMS coefficient strength')
                    a, errors = choose_ridge(dh, y, fit, cal, cfg['ridge_alphas'], 'native_units')
                    rw = ridge(dh[discovery], y[discovery], a, 'native_units')
                    predictions['raw'] = dh@rw@source['basis'].T
                    weights['raw_weights'] = rw
                    diagnostics['raw'] = dict(alpha=a, calibration_mse=errors, read_budget=dim)
                    one, one_diag = assigned_readout(dzs[discovery][:,source['support']], dzt[discovery], source['coefficients'])
                    predictions['one_to_one'] = dzt@one@source['basis'].T
                    weights['one_to_one_weights'] = one
                    diagnostics['one_to_one'] = one_diag
                    operation_mean = np.mean(teacher[discovery]*sign[discovery,None], axis=0)
                    predictions['source_class_mean'] = sign[:,None]*operation_mean
                    retrieved_native = {}
                    retrieved_readers = []
                    if semantic_matches is not None:
                        match_lookup = {r['source_member']:r for r in semantic_matches['rows']}
                        if any(int(member) not in match_lookup for member in source['support']):
                            raise ValueError('The frozen source group was not in natural-reference retrieval')
                        matched = [match_lookup[int(member)] for member in source['support']]
                        sd = s['decoder'][source['support']]
                        td = t['decoder']
                        cosine = (sd@td.T)/(np.linalg.norm(sd,axis=1)[:,None]*np.linalg.norm(td,axis=1)[None,:])
                        retrieved = {
                            'semantic_context': sorted({r['target_member'] for r in matched if r['status']=='MATCHED'}),
                            'centroid_context': sorted({r['centroid_target_member'] for r in matched if r['status']=='MATCHED'}),
                            'decoder_nearest': sorted(set(np.argmax(cosine,axis=1).tolist())),
                            'global_hungarian': sorted(set(global_matches[source_seed,target_seed][source['support']].tolist()))}
                        for name,member_list in retrieved.items():
                            members = np.asarray(member_list,dtype=int)
                            if len(members):
                                alpha,errors = choose_ridge(dzt[:,members],y,fit,cal,cfg['ridge_alphas'],'native_units')
                                weight = ridge(dzt[discovery][:,members],y[discovery],alpha,'native_units')
                                prediction = dzt[:,members]@weight@source['basis'].T
                                native = dzt[evaluation][:,members]@td[members]
                            else:
                                alpha,errors = None,[]
                                weight = np.zeros((0,source['rank']))
                                prediction = np.zeros_like(teacher)
                                native = np.zeros_like(teacher[evaluation])
                            predictions[name] = prediction
                            retrieved_readers.append(name)
                            retrieved_native[name+'_copied_native'] = native
                            weights[name+'_members'],weights[name+'_weights'] = members,weight
                            diagnostics[name] = dict(members=members.tolist(),actual_read_budget=len(members),alpha=alpha,
                                calibration_mse=errors,natural_matched_source_members=sum(r['status']=='MATCHED' for r in matched) if 'context' in name else len(source['support']),
                                source_members=len(source['support']),readout='Joint ridge on retrieved union; copying target donor codes is evaluated separately',
                                unmatched_source_ids=[r['source_member'] for r in matched if r['status']!='MATCHED'] if 'context' in name else [],
                                unmatched_policy='Unmatched features add no target retrieval candidate. Joint ridge predicts all source coordinates from the remaining union; only an empty union is zero. This supersedes the matcher metadata proposed per-member zero consumer.',
                                selection='Frozen natural-context nearest match' if 'context' in name else 'Complete target decoder dictionary only')
                    eigenvalues, eigenvectors = np.linalg.eigh(y[discovery].T@y[discovery]/np.sum(discovery))
                    signal = (np.sqrt(np.maximum(eigenvalues,0))[:,None]*eigenvectors.T)@source['basis'].T
                    eligible = np.any(t['codes'][task_ids[discovery],slot]>0, axis=0)
                    writers, writer_diag = writable_support(t['decoder'], signal, cfg['write_budget'], eligible)
                    candidates = {name+'_readout': value[evaluation] for name, value in predictions.items()}
                    candidates.update(retrieved_native)
                    candidates['same_read_members_native'] = dzt[evaluation][:,selected]@t['decoder'][selected]
                    zbase = t['codes'][eval_ids,slot]
                    native_outputs = {}
                    for name in ['fcc', 'full', 'raw', 'source_oracle']+retrieved_readers:
                        desired = teacher[evaluation] if name=='source_oracle' else predictions[name][evaluation]
                        u, diag = project_native(desired, zbase[:,writers], t['decoder'][writers], max_steps=cfg['projection_steps'])
                        candidates[name+'_fixed_native'] = u@t['decoder'][writers]
                        native_outputs[name+'_fixed_increments'] = u
                        projections.append(dict(task=task, site=site, source_seed=source_seed, target_seed=target_seed,
                                                method=name+'_fixed_native', oracle=name=='source_oracle', read_budget='source' if name=='source_oracle' else diagnostics.get(name,{}).get('actual_read_budget',len(selected) if name=='fcc' else dim if name=='raw' else dzt.shape[1]),
                                                write_budget=len(writers), **diag))
                        checks[f'feasible_{task}_{site}_{source_seed}_{target_seed}_{name}_fixed'] = diag['minimum_final_state'] >= -1e-9
                        if cfg['adaptive_writes']:
                            context_members = adaptive_writable_support(desired, zbase, t['decoder'], cfg['write_budget'], eligible, device=device)
                            context_delta, context_u, context_diag = [], [], []
                            for j, members in enumerate(context_members):
                                u, diag = project_native(desired[j:j+1], zbase[j:j+1,members], t['decoder'][members], max_steps=cfg['projection_steps'])
                                context_delta.append(u[0]@t['decoder'][members])
                                context_u.append(u[0])
                                context_diag.append(diag)
                            candidates[name+'_adaptive_native'] = np.asarray(context_delta)
                            native_outputs[name+'_adaptive_members'] = context_members
                            native_outputs[name+'_adaptive_increments'] = np.asarray(context_u)
                            minstate = min(d['minimum_final_state'] for d in context_diag)
                            checks[f'feasible_{task}_{site}_{source_seed}_{target_seed}_{name}_adaptive'] = minstate >= -1e-9
                            projections.append(dict(task=task, site=site, source_seed=source_seed, target_seed=target_seed,
                                                    method=name+'_adaptive_native', oracle=name=='source_oracle',
                                                    candidate_states_scanned=int(eligible.sum()), write_budget=cfg['write_budget'],
                                                    minimum_final_state=minstate,
                                                    max_projected_gradient=max(d['relative_projected_gradient'] for d in context_diag)))
                    prefix = f'{task}_{site}_s{source_seed}_t{target_seed}'
                    np.savez_compressed(run/f'{prefix}_maps.npz', source_members=source['support'], source_basis=source['basis'],
                                        source_decoder=s['decoder'][source['support']], source_coefficients=source['coefficients'],
                                        write_members=writers, eval_ids=eval_ids, teacher_vectors=teacher[evaluation],
                                        **weights, **native_outputs, **{name+'_vectors':value for name,value in candidates.items()})
                    fits.append(dict(task=task, site=site, source_seed=source_seed, target_seed=target_seed,
                                     source_members=source['support'].tolist(), source_rank=source['rank'],
                                     source_selection=source_selection,
                                     source_singular_values=source['discovery_singular_values'].tolist(),
                                     fits=diagnostics, writers=writers.tolist(), writer_selection=writer_diag,
                                     fit_components=len({panel[i]['component'] for i in task_ids[fit]}),
                                     calibration_components=len({panel[i]['component'] for i in task_ids[cal]}),
                                     evaluation_components=len({panel[i]['component'] for i in eval_ids})))
                    write(run/'fit_diagnostics.json', dict(rows=fits))
                    write(run/'projection_diagnostics.json', dict(rows=projections))
                    for name, delta in candidates.items():
                        measure(task, site, source_seed, target_seed, name, eval_ids, delta, reference=ref,
                                source_delta=teacher[evaluation], reference_identity='source_native', oracle=name.startswith('source_oracle'))
                    if cfg.get('writer_training_steps',0) and site in cfg['writer_training_sites']:
                        # Freeze both readout and members; fit only a native writing
                        # matrix against the complete source response. All methods
                        # receive the same output supervision and parameter budget.
                        train_ids = np.flatnonzero(fit)
                        cal_ids = np.flatnonzero(cal)
                        source_training_lp = np.empty((len(task_ids),baseline.shape[1]),np.float32)
                        for offset in range(0,int(discovery.sum()),bs):
                            local_ids = np.flatnonzero(discovery)[offset:offset+bs]
                            source_training_lp[local_ids] = forward(task_ids[local_ids],teacher[local_ids],slot)[0]
                        decoder_tensor = torch.as_tensor(t['decoder'][writers],dtype=torch.float32,device=device)
                        state_tensor = torch.as_tensor(t['codes'][task_ids,slot][:,writers],dtype=torch.float32,device=device)
                        initial = np.linalg.lstsq(t['decoder'][writers].T,source['basis'],rcond=1e-10)[0]
                        initial_tensor = torch.as_tensor(initial,dtype=torch.float32,device=device)
                        response_tensor = torch.as_tensor(source_training_lp,dtype=torch.float32,device=device)
                        selected_writers = {}
                        for name in cfg['writer_training_inputs']:
                            coordinate_np = y if name=='source_oracle' else predictions[name]@source['basis']
                            coordinates = torch.as_tensor(coordinate_np,dtype=torch.float32,device=device)
                            matrix = torch.nn.Parameter(initial_tensor.clone())
                            optimizer = torch.optim.Adam([matrix],lr=cfg['writer_learning_rate'])
                            rng = np.random.default_rng(cfg['writer_random_seed'])
                            best_loss = float('inf'); best_matrix = None; best_step = None; learning_trace = []
                            def native_values(local_ids, coord=None):
                                values = coordinates[local_ids] if coord is None else coord
                                increments = torch.maximum(values@matrix.T,-state_tensor[local_ids])
                                return increments@decoder_tensor
                            for step in range(cfg['writer_training_steps']+1):
                                if step in cfg['writer_validation_steps']:
                                    losses = []
                                    with torch.no_grad():
                                        for offset in range(0,len(cal_ids),bs):
                                            local = cal_ids[offset:offset+bs]
                                            target_lp = forward(task_ids[local],native_values(local),slot)[0]
                                            rlp = source_training_lp[local].astype(np.float64)
                                            losses.extend(np.sum(np.exp(rlp)*(rlp-target_lp),axis=1).tolist())
                                    calibration_loss = float(np.mean(losses))
                                    learning_trace.append(dict(step=step,calibration_kl=calibration_loss))
                                    if calibration_loss < best_loss:
                                        best_loss = calibration_loss; best_step = step
                                        best_matrix = matrix.detach().cpu().numpy().copy()
                                if step == cfg['writer_training_steps']:
                                    break
                                local = rng.choice(train_ids,size=min(bs,len(train_ids)),replace=False)
                                optimizer.zero_grad(set_to_none=True)
                                lp = forward(task_ids[local],native_values(local),slot,differentiable=True)[0]
                                rlp = response_tensor[local]
                                response_loss = (rlp.exp()*(rlp-lp)).sum(dim=1).mean()
                                regularizer = cfg['writer_anchor_penalty']*(matrix-initial_tensor).square().mean()
                                loss = response_loss+regularizer
                                if not torch.isfinite(loss):
                                    raise ValueError('Nonfinite finite-response writing objective')
                                loss.backward(); optimizer.step()
                            selected_writers[name] = best_matrix
                            with torch.no_grad():
                                matrix.copy_(torch.as_tensor(best_matrix,device=device))
                                delta = native_values(evaluation).cpu().numpy().astype(np.float64)
                                final_state = state_tensor[evaluation]+torch.maximum(coordinates[evaluation]@matrix.T,-state_tensor[evaluation])
                            checks[f'finite_writer_feasible_{task}_{site}_{source_seed}_{target_seed}_{name}'] = float(final_state.min())>=-1e-7
                            measure(task,site,source_seed,target_seed,name+'_finite_native',eval_ids,delta,
                                    reference=ref,source_delta=teacher[evaluation],reference_identity='source_native',
                                    oracle=name=='source_oracle',write_budget=len(writers),selected_step=best_step)
                            fits.append(dict(task=task,site=site,source_seed=source_seed,target_seed=target_seed,
                                             method=name+'_finite_native',writer_parameters=int(matrix.numel()),
                                             write_budget=len(writers),steps=cfg['writer_training_steps'],
                                             trace=learning_trace,selected_step=best_step,selected_calibration_kl=best_loss,
                                             training_components='fit only; readout had already refit on fit+calibration',
                                             objective='Complete source-response KL plus declared anchor penalty; native code increment lower bound -z',
                                             heldout_operations='half dose and the two fixed complementary source member masks',
                                             weight_change_norm=float(np.linalg.norm(best_matrix-initial))))
                            write(run/'fit_diagnostics.json',dict(rows=fits))
                            progress('FINITE_WRITER_COMPLETE',task=task,site=site,input_method=name,selected_step=best_step)
                        np.savez_compressed(run/f'{prefix}_finite_writers.npz',initial_matrix=initial,write_members=writers,
                                            **{name+'_matrix':matrix for name,matrix in selected_writers.items()})
                        if cfg.get('writer_component_tests'):
                            masks = {'half_dose':.5*np.eye(source['rank'])}
                            if source['rank'] != len(source['support']):
                                raise ValueError('Source component tests require an identifiable member-coordinate lift')
                            for part in [0,1]:
                                member_mask = np.zeros(len(source['support']))
                                member_mask[part*(len(member_mask)//2):(part+1)*(len(member_mask)//2) if part==0 else len(member_mask)] = 1
                                masks['members_'+str(part)] = np.linalg.pinv(source['coefficients'])@np.diag(member_mask)@source['coefficients']
                            for operation, coordinate_transform in masks.items():
                                operation_teacher = (y[evaluation]@coordinate_transform)@source['basis'].T
                                operation_ref = measure(task,site,source_seed,target_seed,'source_'+operation,eval_ids,operation_teacher,
                                                        source_delta=operation_teacher,reference_identity='self')
                                for name,matrix in selected_writers.items():
                                    coords = y[evaluation] if name=='source_oracle' else predictions[name][evaluation]@source['basis']
                                    coords = coords@coordinate_transform
                                    desired = coords@source['basis'].T
                                    increments = np.maximum(coords@matrix.T,-zbase[:,writers])
                                    finite_delta = increments@t['decoder'][writers]
                                    u,_ = project_native(desired,zbase[:,writers],t['decoder'][writers],max_steps=cfg['projection_steps'])
                                    for variant,delta in [('finite',finite_delta),('fixed',u@t['decoder'][writers]),('readout',desired)]:
                                        measure(task,site,source_seed,target_seed,name+'_'+variant+'_'+operation,eval_ids,delta,
                                                reference=operation_ref,source_delta=operation_teacher,reference_identity='source_'+operation,
                                                oracle=name=='source_oracle',write_budget=len(writers) if variant!='readout' else 0,
                                                operation_seen_by_writer_training=False)
                    progress('DIRECTION_COMPLETE', task=task, site=site, source_seed=source_seed, target_seed=target_seed, methods=len(candidates))
        checks['finite'] = all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v,float))
        checks['unique_rows'] = len(metrics)==len({(row['kind'],row['task'],row.get('site'),row.get('source_seed'),row.get('target_seed'),row.get('method'),row['row_id']) for row in metrics})
        del assets, model
        gc.collect()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (run/'stderr.log').write_text(traceback.format_exc(), encoding='utf-8')
    status = 'PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    # These summaries are reconstructed only from the persisted observations.
    persisted = [json.loads(line) for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
    cells = {}
    for row in persisted:
        key = tuple(row.get(k) for k in ['kind','task','site','source_seed','target_seed','method'])
        cells.setdefault(key, []).append(row)
    cell_rows = []
    for key, rows in cells.items():
        cell = dict(zip(['kind','task','site','source_seed','target_seed','method'],key))
        cell.update(n=len(rows), components=len({r['component'] for r in rows}))
        for field in ['correct','base_correct','donor_label_correct','donor_oriented_change','kl_to_source','source_squared_error','source_energy']:
            values = [r[field] for r in rows if r.get(field) is not None]
            if values:
                cell[field+'_mean'] = float(np.mean(values))
        cell_rows.append(cell)
    summary = dict(status=status, error=error, checks=checks, rows=len(persisted), cells=cell_rows,
                   sequence_forwards=sequence_forwards, token_forwards=token_forwards,
                   wall_seconds=time.perf_counter()-clock_start, process_cpu_seconds=time.process_time()-process_start,
                   scope=cfg['scope'], metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
                   generator_script_path='scripts/run_causalgym_native_transfer.py',
                   generator_script_sha256=sha256(run/'source_snapshot/scripts/run_causalgym_native_transfer.py'))
    write(run/'inputs.json', dict(inputs=inputs))
    write(run/'environment.json', environment)
    write(run/'metrics.summary.json', summary)
    write(run/'stdout.log', summary)
    write(run/'status.json', dict(status=status, error=error, updated_utc=datetime.now(timezone.utc).isoformat()))
    validation = validate_run_directory(run)
    write(run/'contract_validation.json', dict(ok=validation.ok, errors=list(validation.errors)))
    print(json.dumps(dict(status=status, error=error, contract_ok=validation.ok, wall_seconds=summary['wall_seconds'], rows=len(persisted))), flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
