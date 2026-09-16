"""Reuse frozen human-part correspondences in the original SHIFT retraining consumer.

The classifier recipe follows the published MIT notebook: a freshly initialized
linear head, one AdamW epoch on the ambiguous training set. No balanced labels
or target outputs fit the correspondence or select its support.
"""
from pathlib import Path
import argparse
import json
import random
import sys
import time
import traceback

from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module
from run_shift_transfer import input_member_delta
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    sources = ['scripts/run_shift_retraining.py', 'scripts/run_shift_explanation.py',
               'scripts/train_shift_dictionaries.py', 'scripts/run_causalgym_multisite.py',
               'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py',
               'scripts/run_shift_transfer.py']
    work = MultisiteWork(cfg, args.config, sources)
    hooks, error = [], None
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
        sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        source = json.loads(work.checked(cfg['source_manifest']).read_text())
        groups, annotations = source_groups(work.checked(cfg['notebook']), source['members'])
        bank = np.load(work.checked(cfg['source_parameters']))
        params = {s: {k: torch.tensor(bank[s+'__'+k], device=work.device)
                      for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
                  for s in source['members']}
        frozen = np.load(work.checked(Path(cfg['relation_run'])/'relation.npz'))
        targets, relations = {}, {}
        for site in source['members']:
            state = torch.load(work.checked(Path(cfg['target_directory']) /
                              f'{site}_seed{cfg["target_seed"]}.pt'),
                               map_location=work.device, weights_only=True)
            sae = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
            sae.load_state_dict(state)
            sae.requires_grad_(False)
            targets[site] = sae
            relation_names = list(dict.fromkeys(['native', 'geometry', 'geometry_gain', 'raw', 'candidates']+
                                               [m for m in cfg['methods'] if m not in ['source', 'none','raw_reconstruction'] and not m.startswith('input_')]))
            relations[site] = {name: torch.tensor(frozen[site+'__'+name], device=work.device,
                                                  dtype=torch.long if name == 'candidates' else torch.float32)
                               for name in relation_names}
        if cfg.get('fixed_basis_run'):
            basis=np.load(work.checked(Path(cfg['fixed_basis_run'])/'fixed_response_basis.npz'))
            for site in source['members']:
                params[site]['fixed_response_basis']=torch.tensor(basis[site],device=work.device)
        source_run = Path(cfg['frozen_source_run'])
        panel = json.loads(work.checked(cfg.get('training_panel', str(source_run/'panel.json'))).read_text())
        train = [r for r in panel['rows'] if r['split'] == 'train']
        ep = json.loads(work.checked(cfg['evaluation_panel']).read_text())
        evaluation = [r for r in ep['rows'] if r['split'] == cfg['evaluation_split']]
        assert {r['document_sha256'] for r in train}.isdisjoint(r['document_sha256'] for r in evaluation)
        tasks = cfg.get('tasks', [dict(name='original', negative=None, positive=None, orientation=0)])
        multi_task = tasks[0]['negative'] is not None
        if not multi_task:
            assert all(r['label'] == r['gender'] for r in train)
        write(work.run/'panel.json', dict(train_documents=[r['document_sha256'] for r in train],
                                          rows=evaluation, groups=groups, annotations=annotations))
        for name in ['config.json', 'model.safetensors']:
            work.checked(Path(cfg['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            cfg['model_local_dir'], local_files_only=True, dtype=torch.float32,
            attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        method, query, mask, pooled = 'none', 'full', None, None
        requests = {q: list(groups) if q == 'full' else q.split('+') for q in cfg['queries']}
        qweights = {(q, site): torch.tensor(
            [any(i in groups[g].get(site, []) for g in gs) for i in source['members'][site]],
            device=work.device, dtype=torch.float32)
            for q, gs in requests.items() for site in source['members']}

        def hook(site):
            def apply(module, inputs, output):
                nonlocal pooled
                h = output[0] if isinstance(output, tuple) else output
                s, q = params[site], qweights[query, site]
                if method == 'source':
                    z = torch.relu((h-s['center']) @ s['encoder'].T+s['encoder_bias'])
                    h = h-(z*q) @ s['decoder']
                elif method.startswith('input_') or method=='raw_reconstruction':
                    if bool(q.any()):
                        delta,_=input_member_delta(h,targets[site],s,q,method,
                                                   cfg['members_per_source']*len(q),mask)
                        h=h+delta
                elif method != 'none':
                    t, r = targets[site], relations[site]
                    z = t.encode(h)
                    if method == 'raw':
                        h = h-(z[..., r['candidates']] @ r['raw']*q) @ s['decoder']
                    else:
                        h = h-(z*(r[method] @ q)) @ t.decoder.weight.T
                if site == 'resid_4':
                    pooled = (h*mask[..., None]).sum(1)/mask.sum(1)[:, None]
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return apply
        for site in source['members']:
            hooks.append(site_module(model, site).register_forward_hook(hook(site)))

        def collect(rows, split):
            nonlocal mask
            output = np.empty((len(rows), 512), np.float32)
            order = sorted(range(len(rows)), key=lambda i: len(rows[i]['tokens']))
            for start in range(0, len(order), cfg['eval_batch_size']):
                ix = order[start:start+cfg['eval_batch_size']]
                length = max(len(rows[i]['tokens']) for i in ix)
                ids = torch.zeros((len(ix), length), device=work.device, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, i in enumerate(ix):
                    v = rows[i]['tokens']
                    ids[j, :len(v)] = torch.tensor(v, device=work.device)
                    mask[j, :len(v)] = 1
                with torch.no_grad():
                    model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                output[ix] = pooled.cpu().numpy()
                work.sequence_forwards += len(ix)
                work.token_forwards += ids.numel()
                if start % 4096 == 0:
                    work.progress('COLLECT', method=method, query=query, split=split,
                                  done=start+len(ix), total=len(rows))
                if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                    raise TimeoutError('Bounded SHIFT consumer experiment')
            return output

        old_probe = np.load(work.checked(source_run/'probe.npz'))
        task_indices = {}
        for task in tasks:
            name = task['name']
            if multi_task:
                negative, positive, orientation = task['negative'], task['positive'], task['orientation']
                negative_rows = [i for i,r in enumerate(train) if r['profession'] == negative and r['gender'] == orientation]
                positive_rows = [i for i,r in enumerate(train) if r['profession'] == positive and r['gender'] == 1-orientation]
                count = min(len(negative_rows), len(positive_rows))
                ti = negative_rows[:count]+positive_rows[:count]
                random.Random(cfg['probe_seeds'][0]).shuffle(ti)
                ei = [i for i,r in enumerate(evaluation) if r['profession'] in [negative, positive]]
                tl = [int(train[i]['profession'] == positive) for i in ti]
                erows = [dict(evaluation[i], label=int(evaluation[i]['profession'] == positive)) for i in ei]
            else:
                ti, ei = list(range(len(train))), list(range(len(evaluation)))
                tl, erows = [r['label'] for r in train], evaluation
            assert ti and ei
            task_indices[name] = (ti, ei, torch.tensor(tl, device=work.device, dtype=torch.float32), erows)
        write(work.run/'task_membership.json', {name: dict(train_indices=x[0], evaluation_indices=x[1])
                                                for name,x in task_indices.items()})
        clean_heads = {}
        summary = {}
        cache_paths = [Path(p) for p in cfg.get('feature_cache_runs', [])]
        if cfg.get('feature_cache_run'):
            cache_paths.append(Path(cfg['feature_cache_run']))
        for query in cfg['queries']:
            for method in cfg['methods']:
                if method == 'none' and query != 'full':
                    continue
                key = method+'__'+query
                cache=next((p for p in cache_paths if (p/(key+'__train.npy')).exists()
                            and (p/(key+'__evaluation.npy')).exists()),None)
                cached=cache is not None
                if cached:
                    previous=json.loads(work.checked(cache/'config.resolved.json').read_text())
                    for field in ['source_parameters','target_directory','target_seed','relation_run','evaluation_split']:
                        assert str(previous[field]).replace('\\','/')==str(cfg[field]).replace('\\','/'),field
                    membership=json.loads(work.checked(cache/'panel.json').read_text())
                    assert membership['train_documents']==[r['document_sha256'] for r in train]
                    assert membership['rows']==evaluation
                    if method.startswith('input_'):
                        assert previous['members_per_source']==cfg['members_per_source']
                        assert previous['fixed_basis_run']==cfg['fixed_basis_run']
                    x=np.load(work.checked(cache/(key+'__train.npy')))
                    y=np.load(work.checked(cache/(key+'__evaluation.npy')))
                    assert x.shape==(len(train),512) and y.shape==(len(evaluation),512)
                elif method == 'none' and not multi_task:
                    x = np.load(work.checked(source_run/'train_pooled.npz'))['hidden']
                else:
                    x = collect(train, 'train')
                if not cached:
                    y = collect(evaluation, cfg['evaluation_split'])
                    np.save(work.run/(key+'__train.npy'), x)
                    np.save(work.run/(key+'__evaluation.npy'), y)
                for task in tasks:
                    task_name = task['name']
                    ti, ei, labels, erows = task_indices[task_name]
                    xx = torch.tensor(x[ti], device=work.device)
                    for probe_seed in cfg['probe_seeds']:
                        torch.manual_seed(probe_seed)
                        head = torch.nn.Linear(512, 1, device=work.device)
                        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['probe_lr'])
                        for start in range(0, len(ti), cfg['probe_batch_size']):
                            values = head(xx[start:start+cfg['probe_batch_size']]).squeeze(-1)
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                values, labels[start:start+cfg['probe_batch_size']])
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        weight, bias = head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy()
                        if method == 'none':
                            clean_heads[task_name, probe_seed] = (weight, bias)
                        reference_weight, reference_bias = clean_heads[task_name, probe_seed] if multi_task else (old_probe['weight'], old_probe['bias'])
                        logits = (y[ei] @ weight.T+bias).ravel()
                        frozen_logits = (y[ei] @ reference_weight.T+reference_bias).ravel()
                        suffix = f'__{task_name}' if multi_task else ''
                        np.savez_compressed(work.run/f'{key}{suffix}__probe{probe_seed}.npz',
                                            weight=weight, bias=bias, logits=logits, frozen_logits=frozen_logits)
                        results = {}
                        for state, values in [('retrained', logits), ('frozen', frozen_logits)]:
                            correct = (values > 0) == np.array([r['label'] for r in erows])
                            group = {f'{p}/{g}': float(correct[[r['label'] == p and r['gender'] == g
                                                               for r in erows]].mean())
                                     for p in [0, 1] for g in [0, 1]}
                            results[state] = dict(profession=float(correct.mean()), worst_group=min(group.values()), groups=group)
                            for r, value in zip(erows, values):
                                work.record(method=method, operation=query, classifier=state,
                                            task=task_name,
                                            row_id=r['document_sha256'], mode=state, seed=probe_seed,
                                            probe_seed=probe_seed, target_seed=cfg['target_seed'],
                                            component=r['document_sha256'], label=r['label'], gender=r['gender'],
                                            logit=float(value), prediction=int(value > 0), split=cfg['evaluation_split'])
                        summary[f'{key}{suffix}__probe{probe_seed}'] = results
                        work.progress('RESULT', method=method, query=query, task=task_name, probe_seed=probe_seed, results=results)
                        write(work.run/'RETRAINING_RESULTS.json', summary)
                    del xx
        work.checks.update(disjoint_documents=True, ambiguous_training_only=True,
                           frozen_correspondences=True, completed_cells=len(summary))
    except Exception:
        error = traceback.format_exc()
    finally:
        for handle in hooks:
            handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
