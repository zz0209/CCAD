"""Measure specified old-explanation requests with frozen correspondence maps.

This development bank evaluates finite interventions. It neither refits maps nor
uses evaluation labels to choose the query family or a correspondence.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    work = MultisiteWork(cfg, args.config, [
        'scripts/run_shift_query_bank.py', 'scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py', 'scripts/train_shift_dictionaries.py',
        'src/ccad/artifacts.py'])
    handles, error = [], None
    try:
        import torch
        import transformers
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_device(cfg['device'])
        torch.cuda.reset_peak_memory_stats()
        work.torch, work.device = torch, torch.device(cfg['device'])
        work.environment = dict(python=sys.executable, torch=torch.__version__,
                                transformers=transformers.__version__, numpy=np.__version__,
                                device=torch.cuda.get_device_name(), cpu_threads=2)
        spec = json.loads(work.checked(cfg['request_spec']).read_text())
        panel = json.loads(work.checked(cfg['evaluation_panel']).read_text())
        available = {r['document_sha256']: r for r in panel['rows']}
        rows = [available[key] for key in spec['documents']]
        source = json.loads(work.checked(cfg['source_manifest']).read_text())
        sites = list(source['members'])
        bank = np.load(work.checked(cfg['source_parameters']))
        params = {s: {k: torch.tensor(bank[s+'__'+k], device=work.device)
                      for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
                  for s in sites}
        sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        frozen = np.load(work.checked(Path(cfg['relation_run'])/'relation.npz'))
        targets, relations = {}, {}
        for site in sites:
            state = torch.load(work.checked(Path(cfg['target_directory']) /
                              f'{site}_seed{cfg["target_seed"]}.pt'),
                               map_location=work.device, weights_only=True)
            sae = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
            sae.load_state_dict(state)
            sae.requires_grad_(False)
            targets[site] = sae
            relations[site] = {name: torch.tensor(frozen[site+'__'+name], device=work.device,
                                 dtype=torch.long if name == 'candidates' else torch.float32)
                               for name in ['native', 'geometry', 'geometry_gain', 'raw', 'candidates']}
        probe = np.load(work.checked(Path(cfg['frozen_source_run'])/'probe.npz'))
        weight, bias = (torch.tensor(probe[k], device=work.device) for k in ['weight', 'bias'])
        for name in ['config.json', 'model.safetensors']:
            work.checked(Path(cfg['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            cfg['model_local_dir'], local_files_only=True, dtype=torch.float32,
            attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        queries = spec['queries']
        qweights = {(i, site): torch.tensor(q['weights'][site], device=work.device)
                    for i, q in enumerate(queries) for site in sites}
        restore_weights = {(i, site): torch.tensor(q.get('restore_weights', {}).get(
                              site, [0.]*len(source['members'][site])), device=work.device)
                           for i, q in enumerate(queries) for site in sites}
        active_delete = {key: bool(value.any()) for key, value in qweights.items()}
        active_restore = {key: bool(value.any()) for key, value in restore_weights.items()}
        assert all(not (active_delete[key] and active_restore[key]) for key in qweights)
        method, qi, mask, pooled = 'none', 0, None, None
        capture_reference, references = False, {}

        def field(h, site, q):
            s = params[site]
            if method == 'source':
                z = torch.relu((h-s['center']) @ s['encoder'].T+s['encoder_bias'])
                return (z*q) @ s['decoder']
            t, r = targets[site], relations[site]
            z = t.encode(h)
            if method == 'raw':
                return (z[..., r['candidates']] @ r['raw']*q) @ s['decoder']
            return (z*(r[method] @ q)) @ t.decoder.weight.T

        def hook(site):
            def apply(module, inputs, output):
                nonlocal pooled
                h = output[0] if isinstance(output, tuple) else output
                if capture_reference:
                    if active_restore[qi, site]:
                        references[site] = field(h, site, restore_weights[qi, site]).detach()
                elif method != 'none':
                    if active_delete[qi, site]:
                        h = h-field(h, site, qweights[qi, site])
                    if active_restore[qi, site]:
                        # Code restoration: z'=(1-a)z_current+a*z_clean.
                        # Source and target restorations use their own clean states.
                        h = h+(references[site]-field(h, site, restore_weights[qi, site]))
                if site == 'resid_4':
                    pooled = (h*mask[..., None]).sum(1)/mask.sum(1)[:, None]
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return apply

        for site in sites:
            handles.append(site_module(model, site).register_forward_hook(hook(site)))
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]['tokens']))
        values = np.full((len(cfg['methods']), len(queries), len(rows)), np.nan, np.float32)
        for mi, method in enumerate(cfg['methods']):
            for qi, query in enumerate(queries):
                if method == 'none' and qi:
                    values[mi, qi] = values[mi, 0]
                    continue
                for off in range(0, len(order), cfg['eval_batch_size']):
                    ix = order[off:off+cfg['eval_batch_size']]
                    length = max(len(rows[i]['tokens']) for i in ix)
                    ids = torch.zeros((len(ix), length), device=work.device, dtype=torch.long)
                    mask = torch.zeros_like(ids)
                    for j, i in enumerate(ix):
                        v = rows[i]['tokens']
                        ids[j, :len(v)] = torch.tensor(v, device=work.device)
                        mask[j, :len(v)] = 1
                    with torch.no_grad():
                        if method != 'none' and any(active_restore[qi, site] for site in sites):
                            capture_reference = True
                            references = {}
                            model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                            work.sequence_forwards += len(ix)
                            work.token_forwards += ids.numel()
                            capture_reference = False
                        model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                        result = (pooled @ weight.T+bias).ravel().cpu().numpy()
                    values[mi, qi, ix] = result
                    work.sequence_forwards += len(ix)
                    work.token_forwards += ids.numel()
                    if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                        raise TimeoutError('Bounded query-bank experiment')
                work.record(method=method, operation=query['name'], component=query['name'],
                            mode='fixed_head', seed=cfg['target_seed'], split=spec['evidence_level'],
                            value=float(np.mean(values[mi, qi])), n=len(rows))
                work.progress('QUERY', method=method, query=query['name'],
                              query_number=qi+1, queries=len(queries), documents=len(rows))
            np.savez_compressed(work.run/'responses.npz', logits=values)
            write(work.run/'BANK_INDEX.json', dict(methods=cfg['methods'], queries=queries,
                  documents=spec['documents'], context_split=spec['context_split'],
                  labels=[r['label'] for r in rows], genders=[r['gender'] for r in rows],
                  evidence_level=spec['evidence_level']))
        assert np.isfinite(values).all()
        work.checks.update(finite_complete_bank=True, frozen_relations=True,
                           source_only_query_spec=True, n_requests=len(queries), n_documents=len(rows))
    except Exception:
        error = traceback.format_exc()
    finally:
        for handle in handles:
            handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
