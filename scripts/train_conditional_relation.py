import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module
from train_shift_program import project_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    work = MultisiteWork(cfg, args.config, ['scripts/train_conditional_relation.py',
        'scripts/train_shift_program.py', 'scripts/train_shift_dictionaries.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
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
            transformers=transformers.__version__, numpy=np.__version__, cpu_threads=2)
        sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        spec = json.loads(work.checked(cfg['request_spec']).read_text())
        panel = json.loads(work.checked(cfg['evaluation_panel']).read_text())
        source = json.loads(work.checked(cfg['source_manifest']).read_text())
        sites = list(source['members'])
        sb = np.load(work.checked(cfg['source_parameters']))
        source_params = {s: {k: torch.tensor(sb[s+'__'+k], device=work.device)
                            for k in ['encoder', 'encoder_bias', 'decoder', 'center']} for s in sites}
        old = np.load(work.checked(Path(cfg['relation_run'])/'relation.npz'))
        targets, initial, coefficients, support, original_support = {}, {}, {}, {}, {}
        candidate_support, activation_energy = {}, {}
        for site in sites:
            state = torch.load(work.checked(Path(cfg['target_directory'])/f'{site}_seed{cfg["target_seed"]}.pt'),
                               map_location=work.device, weights_only=True)
            target = AutoEncoderTopK(512, len(state['encoder.weight']), int(state['k'])).to(work.device)
            target.load_state_dict(state)
            target.requires_grad_(False)
            targets[site] = target
            initial[site] = torch.tensor(old[site+'__native'], device=work.device, dtype=torch.float32)
            support[site] = initial[site].sum(1) > 0
            original_support[site] = support[site].clone()
            candidate_support[site] = torch.zeros_like(support[site])
            candidate_support[site][torch.tensor(old[site+'__candidates'], device=work.device, dtype=torch.long)] = True
            activation_energy[site] = torch.zeros(len(support[site]), device=work.device)
            coefficients[site] = torch.nn.Parameter(initial[site].clone())
        for name in ['config.json', 'model.safetensors']:
            work.checked(Path(cfg['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        probe = np.load(work.checked(Path(cfg['frozen_source_run'])/'probe.npz'))
        pw, pb = [torch.tensor(probe[k], device=work.device) for k in ['weight', 'bias']]
        queries = spec['queries']
        lookup = {q['name']: i for i, q in enumerate(queries)}
        delete = {(i, s): torch.tensor(q['weights'][s], device=work.device) for i, q in enumerate(queries) for s in sites}
        restore = {(i, s): torch.tensor(q.get('restore_weights', {}).get(s, [0.]*len(source['members'][s])),
                                       device=work.device) for i, q in enumerate(queries) for s in sites}
        active_delete = {key: bool(value.any()) for key, value in delete.items()}
        active_restore = {key: bool(value.any()) for key, value in restore.items()}
        mode, qi, capture, mask, pooled, references = 'source', 0, False, None, None, {}

        def field(h, site, q):
            if mode == 'source':
                sp = source_params[site]
                z = torch.relu((h-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                return (z*q)@sp['decoder']
            target = targets[site]
            return (target.encode(h)*(coefficients[site]@q))@target.decoder.weight.T

        def hook(site):
            def apply(module, inputs, output):
                nonlocal pooled
                h = output[0] if isinstance(output, tuple) else output
                if mode == 'statistics':
                    activation_energy[site].add_((targets[site].encode(h).square()*mask[..., None]).sum((0, 1)))
                elif capture:
                    if active_restore[qi, site]:
                        references[site] = field(h, site, restore[qi, site])
                else:
                    if active_delete[qi, site]:
                        h = h-field(h, site, delete[qi, site])
                    if active_restore[qi, site]:
                        h = h+references[site]-field(h, site, restore[qi, site])
                if site == 'resid_4':
                    pooled = (h*mask[..., None]).sum(1)/mask.sum(1)[:, None]
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return apply

        for site in sites:
            handles.append(site_module(model, site).register_forward_hook(hook(site)))

        def forward(ids, index):
            nonlocal qi, capture, references
            qi = index
            if any(active_restore[qi, site] for site in sites):
                capture, references = True, {}
                model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                work.sequence_forwards += len(ids)
                work.token_forwards += ids.numel()
                capture = False
            model.gpt_neox(ids, attention_mask=mask, use_cache=False)
            work.sequence_forwards += len(ids)
            work.token_forwards += ids.numel()
            return (pooled@pw.T+pb).ravel()

        training_edges = [i for i, q in enumerate(queries) if q['role'] == 'restore'
                          and int(hashlib.sha256(q['name'].encode()).hexdigest(), 16) % 2 == 0]
        held_edges = [i for i, q in enumerate(queries) if q['role'] == 'restore' and i not in training_edges]
        rows = sorted((r for r in panel['rows'] if r['split'] == 'dev' and r['document_sha256'] not in spec['documents']),
                      key=lambda r: r['document_sha256'])
        assert rows and training_edges and held_edges
        rng = np.random.default_rng(cfg['training_seed'])
        edge_order = np.resize(rng.permutation(training_edges), cfg['steps'])
        row_order = rng.integers(len(rows), size=(cfg['steps'], cfg['batch_sequences']))
        write(work.run/'FIT_MEMBERSHIP.json', dict(documents=[r['document_sha256'] for r in rows],
            training_requests=[queries[i]['name'] for i in training_edges],
            held_requests=[queries[i]['name'] for i in held_edges], source_labels_used=False,
            edge_order=edge_order.tolist(), row_order=row_order.tolist(),
            support='Original geometry candidate pool with original final row count.' if cfg.get('reselect_step')
                    else 'Existing native nonzero target rows, fixed for both variants.'))
        if cfg.get('reselect_step'):
            mode = 'statistics'
            with torch.no_grad():
                for off in range(0, len(rows), cfg['batch_sequences']):
                    selected = rows[off:off+cfg['batch_sequences']]
                    length = max(len(r['tokens']) for r in selected)
                    ids = torch.zeros((len(selected), length), dtype=torch.long, device=work.device)
                    mask = torch.zeros_like(ids)
                    for j, row in enumerate(selected):
                        ids[j, :len(row['tokens'])] = torch.tensor(row['tokens'], device=work.device)
                        mask[j, :len(row['tokens'])] = 1
                    model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                    work.sequence_forwards += len(ids)
                    work.token_forwards += ids.numel()
            work.progress('MEMBER_ENERGY', documents=len(rows))
        for variant in cfg['variants']:
            with torch.no_grad():
                for site in sites:
                    coefficients[site].copy_(initial[site])
                    support[site] = (candidate_support[site] if cfg.get('reselect_step') else original_support[site]).clone()
            optimizer = torch.optim.AdamW(list(coefficients.values()), lr=cfg['relation_lr'], weight_decay=0.)
            for step, edge in enumerate(edge_order):
                selected = [rows[i] for i in row_order[step]]
                length = max(len(r['tokens']) for r in selected)
                ids = torch.zeros((len(selected), length), dtype=torch.long, device=work.device)
                mask = torch.zeros_like(ids)
                for j, row in enumerate(selected):
                    ids[j, :len(row['tokens'])] = torch.tensor(row['tokens'], device=work.device)
                    mask[j, :len(row['tokens'])] = 1
                first = lookup[queries[edge]['ablation_reference']]
                second = edge if variant in ['conditional', 'contrast'] else lookup['delete_'+queries[edge]['restore_node']]
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                errors = []
                for index in [first, second]:
                    mode = 'source'
                    with torch.no_grad():
                        teacher = forward(ids, index)
                    mode = 'native'
                    predicted = forward(ids, index)
                    difference = predicted-teacher
                    if variant == 'contrast':
                        errors.append(difference)
                    else:
                        loss = difference.square().mean()/2
                        loss.backward()
                        total += float(loss.detach())
                if variant == 'contrast':
                    loss = (errors[0].square().mean()+(errors[1]-errors[0]).square().mean())/2
                    loss.backward()
                    total = float(loss.detach())
                torch.nn.utils.clip_grad_norm_(list(coefficients.values()), 1.)
                optimizer.step()
                with torch.no_grad():
                    for site, a in coefficients.items():
                        a[~support[site]] = 0
                        project_rows(a)
                        if step+1 == cfg.get('reselect_step'):
                            allowance = int(original_support[site].sum())
                            score = a.square().sum(1)*activation_energy[site]*targets[site].decoder.weight.square().sum(0)
                            score[~candidate_support[site]] = -torch.inf
                            indices = score.topk(allowance).indices
                            support[site] = torch.zeros_like(support[site])
                            support[site][indices] = True
                            a[~support[site]] = 0
                if not np.isfinite(total):
                    raise ValueError('Nonfinite conditional training loss')
                if (step+1) % 32 == 0 or step+1 == cfg['steps']:
                    work.record(kind='training', task='conditional_relation', component=variant,
                                row_id=step+1, method=variant, step=step+1, loss=total,
                                seed=cfg['target_seed'], split='development')
                    work.progress('TRAIN', method=variant, step=step+1, steps=cfg['steps'], loss=total)
                    torch.save(dict(variant=variant, step=step+1,
                        coefficients={s: a.detach().cpu() for s, a in coefficients.items()},
                        optimizer=optimizer.state_dict()), work.run/f'{variant}_checkpoint.pt')
                if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                    raise TimeoutError('Conditional-relation training budget exceeded')
            dest = work.run/variant
            dest.mkdir()
            export = {key: old[key].copy() for key in old.files}
            for site, a in coefficients.items():
                assert float(a.detach().min()) >= 0 and float(a.detach().sum(1).max()) <= 1.00001
                assert not bool(a.detach()[~support[site]].any())
                assert int(support[site].sum()) == int(original_support[site].sum())
                export[site+'__native'] = a.detach().cpu().numpy()
            np.savez_compressed(dest/'relation.npz', **export)
        work.checks.update(frozen_dictionary=True, final_member_allowance=True,
            matched_optimizer_steps=True, held_compositions=len(held_edges), training_compositions=len(training_edges))
    except Exception:
        error = traceback.format_exc()
    finally:
        for handle in handles:
            handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
