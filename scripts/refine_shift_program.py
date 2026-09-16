"""Refine a frozen correspondence through its executed semantic-part programs.

The teacher is the published source explanation. Natural-text contexts and
source classifier responses supply supervision; no new biography-task data enter.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    work = MultisiteWork(cfg, args.config, [
        'scripts/refine_shift_program.py', 'scripts/run_shift_transfer.py',
        'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py',
        'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
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
        groups, _ = source_groups(work.checked(cfg['notebook']), source['members'])
        group_names = list(groups)
        assert len(group_names) == 3
        bank = np.load(work.checked(cfg['source_parameters']))
        params = {s: {k: torch.tensor(bank[s+'__'+k], device=work.device)
                       for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
                  for s in source['members']}
        frozen = np.load(work.checked(Path(cfg['relation_run'])/'relation.npz'))
        export = {k: frozen[k].copy() for k in frozen.files}
        targets, base, support, partition, lift = {}, {}, {}, {}, {}
        for site, members in source['members'].items():
            state = torch.load(work.checked(Path(cfg['target_directory']) /
                               f'{site}_seed{cfg["target_seed"]}.pt'),
                               map_location=work.device, weights_only=True)
            sae = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
            sae.load_state_dict(state)
            sae.requires_grad_(False)
            targets[site] = sae
            a = frozen[site+'__'+cfg.get('base_method', 'native')]
            p = np.array([[i in groups[g].get(site, []) for g in group_names] for i in members], float)
            assert np.all(p.sum(1) == 1)
            ix = np.flatnonzero(a.sum(1) > 0)
            support[site] = torch.tensor(ix, device=work.device)
            partition[site] = torch.tensor(p, device=work.device, dtype=torch.float32)
            base[site] = torch.tensor(a[ix] @ p, device=work.device, dtype=torch.float32)
            # Only semantic-part requests identify the fitted columns. This lift
            # preserves the old within-part proportions for storage compatibility.
            weights = np.zeros((len(ix), 3, len(members)), float)
            for g in range(3):
                js = np.flatnonzero(p[:, g])
                if not len(js):
                    continue
                vals = a[ix][:, js]
                den = vals.sum(1, keepdims=True)
                weights[:, g, js] = np.divide(vals, den, out=np.full_like(vals, 1/len(js)), where=den > 0)
            lift[site] = weights
        for name in ['config.json', 'model.safetensors']:
            work.checked(Path(cfg['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            cfg['model_local_dir'], local_files_only=True, dtype=torch.float32,
            attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        old_probe = np.load(work.checked(Path(cfg['frozen_source_run'])/'probe.npz'))
        pw = torch.tensor(old_probe['weight'].ravel(), device=work.device)
        tokens = np.memmap(work.checked(cfg['paired_tokens']), dtype='<u2', mode='r').reshape(-1, 128)
        ids = torch.tensor(np.array(tokens[:cfg['fit_sequences']], dtype='int64'), device=work.device)
        requests = torch.tensor([[bool(i & (1 << g)) for g in range(3)] for i in range(1, 8)],
                                device=work.device, dtype=torch.float32)
        mode, pooled, q, coefficients = 'none', None, requests[-1], {}

        def hook(site):
            def apply(module, inputs, output):
                nonlocal pooled
                h = output[0] if isinstance(output, tuple) else output
                if mode == 'source':
                    s = params[site]
                    z = torch.relu((h-s['center']) @ s['encoder'].T+s['encoder_bias'])
                    h = h-(z*(partition[site] @ q)) @ s['decoder']
                elif mode == 'target':
                    z = targets[site].encode(h)[..., support[site]]
                    h = h-(z*(coefficients[site] @ q)) @ targets[site].decoder.weight[:, support[site]].T
                if site == 'resid_4':
                    pooled = h.mean(1)
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return apply
        for site in source['members']:
            hooks.append(site_module(model, site).register_forward_hook(hook(site)))

        def forward(index):
            model.gpt_neox(ids[index], use_cache=False)
            work.sequence_forwards += len(index)
            work.token_forwards += len(index)*128
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Bounded executed-program refinement')
            return pooled

        teacher = []
        with torch.no_grad():
            for mode in ['none', 'source']:
                for ri in ([-1] if mode == 'none' else range(7)):
                    q = requests[ri]
                    values = torch.cat([forward(range(i, min(i+8, len(ids)))) for i in range(0, len(ids), 8)])
                    if mode == 'none':
                        clean = values
                    else:
                        teacher.append(values)
        teacher = torch.stack(teacher)
        ntrain = cfg['refine_train_sequences']
        delta = teacher[:, :ntrain]-clean[:ntrain]
        physical_scale = delta.square().mean((1, 2)).clamp_min(1e-4)
        response_scale = (delta @ pw).square().mean(1).clamp_min(1e-4)
        np.savez_compressed(work.run/'teacher.npz', hidden=teacher.cpu().numpy(), clean=clean.cpu().numpy(),
                            requests=requests.cpu().numpy(), physical_scale=physical_scale.cpu().numpy(),
                            response_scale=response_scale.cpu().numpy())
        history, fitted = {}, {}
        mode = 'target'
        for variant in ['program_gain', 'program']:
            parameters = {s: torch.nn.Parameter(torch.ones(3, device=work.device) if variant == 'program_gain'
                                                 else base[s].clone()) for s in base}
            optimizer = torch.optim.Adam(parameters.values(), lr=cfg['learning_rate'])
            rng = np.random.default_rng(cfg['optimization_seed'])

            def current():
                result = {}
                for s, parameter in parameters.items():
                    value = (base[s]*parameter.clamp_min(0) if variant == 'program_gain' else parameter.clamp_min(0))
                    # Empty source groups at a site remain empty.
                    value = value*(partition[s].sum(0) > 0)
                    result[s] = value/value.sum(1, keepdim=True).clamp_min(1)
                return result

            def loss_for(y, ri, index):
                diff = y-teacher[ri, index]
                physical = diff.square().mean()/physical_scale[ri]
                response = (diff @ pw).square().mean()/response_scale[ri]
                return .5*physical+.5*response

            best, best_step, saved = float('inf'), None, None
            trace = []
            for step in range(cfg['optimization_steps']+1):
                coefficients = current()
                if step % cfg['validation_every'] == 0:
                    with torch.no_grad():
                        scores = []
                        for ri in range(7):
                            q = requests[ri]
                            for i in range(ntrain, len(ids), 8):
                                index = list(range(i, min(i+8, len(ids))))
                                scores.append(float(loss_for(forward(index), ri, index)))
                    score = float(np.mean(scores))
                    trace.append(dict(step=step, validation_loss=score))
                    work.record(kind='natural_validation', task='semantic_program', row_id=step,
                                component=step, mode='refinement', method=variant, seed=cfg['optimization_seed'],
                                target_seed=cfg['target_seed'], operation='all_seven', split='natural_calibration', loss=score)
                    if score < best:
                        best, best_step = score, step
                        saved = {s: v.detach().cpu().numpy().copy() for s,v in coefficients.items()}
                    work.progress('REFINE', variant=variant, step=step, loss=score, best=best)
                if step == cfg['optimization_steps']:
                    break
                ri = int(rng.integers(7))
                q = requests[ri]
                index = rng.choice(ntrain, cfg['refine_batch_size'], replace=False).tolist()
                optimizer.zero_grad()
                loss = loss_for(forward(index), ri, index)
                loss.backward()
                optimizer.step()
            export_name = cfg.get('variant_prefix', '')+variant
            history[export_name] = dict(best=best, best_step=best_step, trace=trace)
            fitted[export_name] = saved
            for site, value in saved.items():
                a = np.zeros_like(frozen[site+'__native'])
                a[support[site].cpu().numpy()] = np.einsum('rg,rgm->rm', value, lift[site])
                assert np.max(a.sum(1)) <= 1.00001
                export[site+'__'+export_name] = a
                export[site+'__'+export_name+'_groups'] = value
            write(work.run/'REFINEMENT_RESULTS.json', history)
            np.savez_compressed(work.run/'relation.npz', **export)
        work.checks.update(frozen_support=True, source_responses_only=True, complete_variants=len(fitted) == 2)
    except Exception:
        error = traceback.format_exc()
    finally:
        for handle in hooks:
            handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
