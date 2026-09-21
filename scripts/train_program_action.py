import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from ccad.intervention_transport import project_capacity
from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    current = json.loads(args.config.read_text())
    c = json.loads(Path(current['base_config']).read_text()) | current
    w = MultisiteWork(c, args.config, ['scripts/train_program_action.py',
        'scripts/run_shift_explanation.py', 'scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/intervention_transport.py',
        'src/ccad/artifacts.py'])
    error = None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats()
        w.torch, w.device = torch, torch.device(c['device'])
        w.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__)
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py',
                  'Pinned TopK implementation', 'MIT')
        manifest = json.loads(w.checked(c['source_manifest']).read_text())
        groups, _ = source_groups(w.checked(c['notebook']), manifest['members'])
        bank = np.load(w.checked(c['source_parameters']))
        old = np.load(w.checked(Path(c['relation_run'])/'relation.npz'))
        cache = Path(c['state_run'])/'state_cache'
        membership = json.loads(w.checked(Path(c['state_run'])/'membership.json').read_text())
        write(w.run/'membership.json', membership)
        write(w.run/'natural_membership.json', json.loads(w.checked(
            Path(c['state_run'])/'natural_membership.json').read_text()))
        exports = {v: {} for v in c['action_variants']}
        checkpoints, diagnostics = [], []
        for site, members in manifest['members'].items():
            initial = torch.load(w.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt'),
                                 map_location='cpu', weights_only=True)
            target = AutoEncoderTopK(512, len(initial['encoder.weight']), int(initial['k'])).to(w.device)
            sp = {key: torch.tensor(bank[site+'__'+key], device=w.device)
                  for key in ['encoder', 'encoder_bias', 'decoder', 'center']}
            partition = torch.tensor([[float(i in groups[g].get(site, [])) for g in groups]
                                      for i in members], device=w.device)
            assert bool((partition.sum(1) == 1).all())
            template = torch.tensor(old[site+'__native'], device=w.device, dtype=torch.float32)
            indices = (template.sum(1) > 0).nonzero().flatten()
            assert 0 < len(indices) <= c['members_per_source']*len(members)
            natural = torch.load(w.checked(cache/'natural'/f'{site}.pt'), weights_only=True).to(w.device)
            states = {name: torch.load(w.checked(cache/'fit'/name/f'{site}.pt'), weights_only=True).to(w.device)
                      for name in ['clean', 'full', *groups]}
            evaluation = torch.load(w.checked(cache/'evaluation'/'clean'/f'{site}.pt'),
                                    weights_only=True).to(w.device)
            # 同一位置的全部拟合状态共同确定尺度，保留源作用为零的批次。
            with torch.no_grad():
                calibration_energy, calibration_count = 0., 0
                for values in states.values():
                    for x in values.split(c['state_batch']):
                        zs = torch.relu((x-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                        fields = torch.einsum('bi,ip,id->bpd', zs, partition, sp['decoder'])
                        calibration_energy += float(fields.square().sum())
                        calibration_count += len(x)
                action_scale = calibration_energy/calibration_count
                assert action_scale > 0 and np.isfinite(action_scale)

            def actions(x, coefficients):
                zs = torch.relu((x-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                desired = torch.einsum('bi,ip,id->bpd', zs, partition, sp['decoder'])
                z = target.encode(x)
                predicted = torch.einsum('bj,jp,dj->bpd', z[:, indices],
                                         coefficients@partition, target.decoder.weight[:, indices])
                return desired, predicted, z

            for variant in c['action_variants']:
                target.load_state_dict(initial)
                target.requires_grad_(True)
                coefficients = torch.nn.Parameter(template[indices].clone())
                parameters = [*target.parameters(), coefficients]
                optimizer = torch.optim.AdamW(parameters, lr=c['learning_rate'], weight_decay=0.)
                generator = torch.Generator(device=w.device).manual_seed(c['training_seed'])
                for step in range(c['steps']):
                    ni = torch.randint(len(natural), (c['state_batch'],), device=w.device, generator=generator)
                    bi = torch.randint(len(states['clean']), (c['state_batch'],), device=w.device, generator=generator)
                    query = list(states)[step % len(states)]
                    x, nx = states[query][bi], natural[ni]
                    desired, predicted, z = actions(x, coefficients)
                    energy = action_scale*len(x)
                    difference = predicted-desired
                    action = (difference.sum(1).square().sum() if variant == 'whole'
                              else difference.square().sum())/energy
                    reconstruction = .5*((target(nx)-nx).square().mean()/nx.square().mean().clamp_min(1e-8)
                        +(target.decode(z)-x).square().mean()/x.square().mean().clamp_min(1e-8))
                    loss = action+c['reconstruction_weight']*reconstruction
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite local action objective')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.)
                    optimizer.step()
                    with torch.no_grad():
                        coefficients.copy_(-project_capacity(-coefficients.clamp_min(0),
                            torch.ones(len(indices), device=w.device)))
                        target.decoder.weight.div_(target.decoder.weight.norm(dim=0).clamp_min(1e-10))
                    if (step+1) % c['log_every'] == 0 or step+1 == c['steps']:
                        w.record(kind='training', task=site, component=query, row_id=step+1, method=variant,
                                 step=step+1, loss=float(loss.detach()), action=float(action.detach()),
                                 reconstruction=float(reconstruction.detach()))
                        w.progress('ACTION_TRAIN', site=site, variant=variant, step=step+1,
                                   total=c['steps'], action=float(action.detach()),
                                   reconstruction=float(reconstruction.detach()))
                    if (step+1) % c['checkpoint_every'] == 0 or step+1 == c['steps']:
                        folder = w.run/variant/f'step_{step+1}'
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder/f'{site}_seed{c["target_seed"]}.pt'
                        torch.save(target.state_dict(), path)
                        matrix = torch.zeros_like(template)
                        matrix[indices] = coefficients.detach()
                        if step+1 not in exports[variant]:
                            exports[variant][step+1] = {key: old[key].copy() for key in old.files}
                        exports[variant][step+1][site+'__native'] = matrix.cpu().numpy()
                        np.savez_compressed(folder/'relation.npz', **exports[variant][step+1])
                        checkpoints.append(dict(site=site, method=variant, step=step+1, path=str(path)))
                        write(w.run/'checkpoints.json', checkpoints)
                    if time.perf_counter()-w.wall_start > c['budget_seconds']:
                        raise TimeoutError('Local action driver budget reached')
                target.requires_grad_(False)
                with torch.no_grad():
                    numerator = denominator = reconstruction_error = 0.
                    counts = 0
                    for x in evaluation.split(c['state_batch']):
                        desired, predicted, z = actions(x, coefficients)
                        numerator += float((predicted-desired).square().sum())
                        denominator += float(desired.square().sum())
                        reconstruction_error += float((target.decode(z)-x).square().sum())
                        counts += int((z > 0).sum())
                    diagnostics.append(dict(site=site, method=variant, part_error=numerator,
                        source_energy=denominator, reconstruction_error=reconstruction_error,
                        clean_variance=float((evaluation-evaluation.mean(0)).square().sum()),
                        l0=counts/len(evaluation), selected_members=len(indices),
                        minimum_coefficient=float(coefficients.min()),
                        maximum_capacity=float(coefficients.sum(1).max()),
                        action_scale=action_scale))
                    write(w.run/'local_quality.json', diagnostics)
                del optimizer, coefficients, parameters
            del target, natural, states, evaluation
            torch.cuda.empty_cache()
        w.checks.update(all_sites=len(diagnostics) == len(manifest['members'])*len(c['action_variants']),
            nonnegative=all(r['minimum_coefficient'] >= 0 for r in diagnostics),
            capacity=all(r['maximum_capacity'] <= 1.00001 for r in diagnostics),
            no_task_outputs_used=True, evaluation_documents_excluded=True)
    except Exception:
        error = traceback.format_exc()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
