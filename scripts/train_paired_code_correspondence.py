from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup

from ccad.artifacts import sha256
from run_causalgym_multisite import MultisiteWork, write
from train_grammar_material_support import load_config
from train_grammar_functional_blocks import variance_statistics
from train_shift_program import project_rows


PARTS = ('verb', 'number', 'gender')


def prepare(work, cfg, api):
    index = json.loads(work.checked(Path(cfg['cache_run'])/'cache_index.json').read_text())
    banks = {}
    for name in PARTS:
        with np.load(work.checked(index['arrivals'][name]['stage2'])) as saved:
            banks[name] = {key: saved[key] for key in saved.files}
    hidden = banks['verb']['hidden']
    assert all(np.array_equal(hidden, bank['hidden']) for bank in banks.values())
    members = np.concatenate([banks[name]['source_members'] for name in PARTS])
    part_ids = np.repeat(np.arange(3), 64)
    order = np.argsort(members)
    members, part_ids = members[order], part_ids[order]
    assert len(np.unique(members)) == 192
    source = api.load_dictionary(work.checked(cfg['source_checkpoint']), work.device)
    source.requires_grad_(False)
    x = torch.tensor(hidden, device=work.device)
    with torch.no_grad():
        # 源teacher沿原AutoEncoderTopK编码，目标两臂共用普通FP32编码。
        zs = source.encode(x)[:, members]
        decoder = source.decoder.weight[:, members].T.detach()
        teacher_error = {}
        for pi, name in enumerate(PARTS):
            teacher = -(zs[:, part_ids == pi] @ decoder[part_ids == pi])
            teacher_error[name] = float((teacher-torch.tensor(banks[name]['teacher'], device=work.device)).abs().max())
        assert max(teacher_error.values()) < 1e-4, teacher_error
    with np.load(work.checked(cfg['natural_states'])) as saved:
        natural = saved['hidden'].copy()
    natural_train, natural_quality = natural[:-1024], natural[-1024:]
    rng = np.random.default_rng(cfg['training_seed'])
    state_ids = np.empty((512, 32), np.int64)
    requests = np.zeros((512, 32, 192), np.float32)
    families = np.tile(np.repeat([0, 1], 16), (512, 1))
    source_codes = zs.cpu().numpy()
    for step in range(512):
        state_ids[step] = rng.integers(len(hidden), size=32)
        for j in range(16):
            requests[step, j, part_ids == ((step*16+j) % 3)] = 1
        for j in range(16, 32):
            active = np.flatnonzero(source_codes[state_ids[step, j]] > 0)
            eligible = active if j < 24 and len(active) else np.arange(192)
            requests[step, j, rng.choice(eligible)] = 1
    natural_ids = rng.integers(len(natural_train), size=(512, 256))
    selected = zs[torch.tensor(state_ids.reshape(-1), device=work.device)]
    q = torch.tensor(requests.reshape(-1, 192), device=work.device)
    with torch.no_grad():
        amplitudes = selected*q
        displacement = -(amplitudes @ decoder)
        h0 = x[torch.tensor(state_ids.reshape(-1), device=work.device)]
        hs = h0+displacement
        source_code_energy = amplitudes.square().sum(-1).cpu().numpy()
        source_effect_energy = displacement.square().sum(-1).cpu().numpy()
    scales = {}
    for family, name in enumerate(('group', 'singleton')):
        mask = families.reshape(-1) == family
        scales[name] = dict(code=float(source_code_energy[mask].mean()),
            physical=float(source_effect_energy[mask].mean()),
            zero_effect_count=int((source_effect_energy[mask] == 0).sum()), count=int(mask.sum()))
        assert scales[name]['code'] > 0 and scales[name]['physical'] > 0
    paired_statistics = variance_statistics(np.concatenate([h0.cpu().numpy(), hs.cpu().numpy()]))
    natural_statistics = variance_statistics(natural_train)
    np.savez_compressed(work.run/'training_table.npz', state_ids=state_ids, requests=requests,
        families=families, natural_ids=natural_ids, source_members=members, source_part=part_ids,
        source_code_energy=source_code_energy.reshape(512, 32),
        source_effect_energy=source_effect_energy.reshape(512, 32))
    table_identity = hashlib.sha256()
    for value in (state_ids, requests, families, natural_ids, members, part_ids):
        table_identity.update(value.tobytes())
    table_hash = table_identity.hexdigest()
    # 评价mask仅由源成员身份和固定随机数决定。
    evaluation_rng = np.random.default_rng(20260924)
    subset_queries = np.zeros((8, 192), np.float32)
    for row in subset_queries:
        for pi in range(3):
            row[evaluation_rng.choice(np.flatnonzero(part_ids == pi), 32, replace=False)] = 1
    np.savez_compressed(work.run/'evaluation_subsets.npz', requests=subset_queries,
        source_members=members, source_part=part_ids)
    write(work.run/'source_information.json', dict(source_members=members.tolist(),
        source_part=part_ids.tolist(), part_names=list(PARTS), teacher_replay_max_error=teacher_error,
        relation_scales=scales, natural_statistics=natural_statistics, paired_statistics=paired_statistics,
        training_table_sha256=sha256(work.run/'training_table.npz'),
        training_table_array_sha256=table_hash,
        information='Exposed192 R26 fit prefix states; fixed source teacher; no target responses',
        source_checkpoint_sha256=sha256(Path(cfg['source_checkpoint']))))
    work.progress('SOURCE_TABLE_READY', requests=512*32, teacher_replay_max_error=teacher_error)
    result = dict(hidden=x, hs=hs.reshape(512, 32, -1), displacement=displacement.reshape(512, 32, -1),
        requests=q.reshape(512, 32, 192), state_ids=state_ids, natural_ids=natural_ids,
        natural=torch.tensor(natural_train, device=work.device), quality=natural_quality,
        members=members, part_ids=part_ids, source_decoder=decoder, scales=scales,
        table_hash=table_hash,
        natural_scale=natural_statistics['mean_square_variance'], paired_scale=paired_statistics['mean_square_variance'])
    del source, selected, amplitudes, h0
    return result


def initial_relation(ae, source_decoder):
    cosine = F.normalize(ae.decoder.weight.detach().double().T, dim=1) @ F.normalize(source_decoder.double(), dim=1).T
    # stable排序让完全相等的cosine使用较小目标成员ID。
    ids = torch.argsort(cosine, dim=0, descending=True, stable=True)[:2]
    values = cosine.gather(0, ids).clamp_min(0)
    values /= values.sum(0).clamp_min(1e-30)
    a = torch.zeros_like(cosine).scatter(0, ids, values).float()
    a /= a.sum(1, keepdim=True).clamp_min(1)
    return a


def measure_quality(ae, states, api, device):
    total_error, l0 = 0., 0
    alive = torch.zeros(8192, dtype=torch.bool, device=device)
    with torch.no_grad():
        for start in range(0, len(states), 256):
            x = torch.tensor(states[start:start+256], device=device)
            z = api.encode(ae, x, {}, 'global')
            total_error += float((ae.decode(z)-x).double().square().sum())
            l0 += int((z > 0).sum())
            alive |= (z > 0).any(0)
    variance = variance_statistics(states)['centered_sum_squares']
    return dict(fve=1-total_error/variance, squared_error=total_error, variance=variance,
        l0=l0/len(states), alive=int(alive.sum()), dead=int((~alive).sum()),
        decoder_norm_max_error=float((ae.decoder.weight.norm(dim=0)-1).abs().max()))


def train(work, cfg, api):
    data = prepare(work, cfg, api)
    final = []
    encoded_states, backwards = 0, 0
    for seed in cfg['target_seeds']:
        initial_path = work.checked(cfg['target_checkpoint_template'].format(seed=seed))
        initial = api.load_dictionary(initial_path, work.device)
        with torch.no_grad():
            a0 = initial_relation(initial, data['source_decoder'])
        np.savez_compressed(work.run/f'initial_relation_s{seed}.npz', A=a0.cpu().numpy(),
            source_members=data['members'], source_part=data['part_ids'])
        del initial
        for method in ('physical', 'code'):
            cell = work.run/f'seed{seed}'/method
            cell.mkdir(parents=True)
            ae = api.load_dictionary(initial_path, work.device)
            trainable, frozen = api.configure_trainable(ae, {}, [])
            a = torch.nn.Parameter(a0.clone())
            optimizer = torch.optim.Adam([dict(params=[p for p in ae.parameters() if p.requires_grad], lr=cfg['learning_rate']),
                dict(params=[a], lr=cfg['relation_learning_rate'])])
            scheduler = get_constant_schedule_with_warmup(optimizer, cfg['warmup_steps'])
            metadata = dict(target_seed=seed, method=method, initial_checkpoint=str(initial_path),
                initial_sha256=sha256(initial_path), training_table_sha256=data['table_hash'],
                source_information=str(work.run/'source_information.json'), b_dec_fixed=True,
                encoder='ordinary FP32 linear relu global TopK64', both_encoder_branches_receive_gradient=True,
                parameter_count=sum(p.numel() for p in ae.parameters() if p.requires_grad)+a.numel())
            write(cell/'config.json', dict(cfg, cell=metadata))
            write(cell/'source_identity.json', json.loads((work.run/'code_hashes.json').read_text()))
            write(cell/'environment.json', work.environment)
            start_step = 0
            previous = Path(cfg['resume_run'])/f'seed{seed}'/method if cfg.get('resume_run') else None
            if previous is not None and (previous/'resume_state.pt').exists():
                state = torch.load(previous/'resume_state.pt', map_location=work.device, weights_only=True)
                saved = torch.load(previous/f'step_{state["step"]}.pt', map_location=work.device, weights_only=True)
                assert saved['metadata']['training_table_sha256'] == metadata['training_table_sha256']
                ae.load_state_dict(saved['dictionary'])
                a.data.copy_(saved['A'])
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                start_step = state['step']
                if start_step == cfg['steps']:
                    checkpoint = previous/f'step_{start_step}.pt'
                    quality = json.loads((previous/f'quality_{start_step}.json').read_text())
                    final.append(dict(metadata, path=str(checkpoint), step=start_step,
                        sha256=sha256(checkpoint), quality=quality, reused_from=str(previous)))
                    write(work.run/'final_checkpoints.json', dict(checkpoints=final))
                    work.record(kind='quality', task='paired_code', component=f'seed{seed}_{method}',
                        row_id=start_step, method=method, seed=seed, target_seed=seed, reused_from=str(previous), **quality)
                    del ae, a, optimizer, scheduler
                    continue
                assert start_step < cfg['steps']
            for step in range(start_step, cfg['steps']):
                api.normalize_decoder(ae, trainable)
                x = data['hidden'][data['state_ids'][step]]
                hs, q, displacement = data['hs'][step], data['requests'][step], data['displacement'][step]
                nx = data['natural'][data['natural_ids'][step]]
                optimizer.zero_grad(set_to_none=True)
                z = api.encode(ae, x, {}, 'global')
                edited_z = api.encode(ae, hs, {}, 'global')
                aq = q @ a.T
                e = edited_z-(1-aq)*z
                physical_error = -(z*aq) @ ae.decoder.weight.T-displacement
                code_per_row = e.square().sum(-1)
                physical_per_row = physical_error.square().sum(-1)
                with torch.no_grad():
                    decoded_error_energy = float((e.detach() @ ae.decoder.weight.detach().T).square().sum(-1).mean())
                code_loss = sum(code_per_row[sl].mean()/data['scales'][name]['code']
                    for name, sl in [('group', slice(0, 16)), ('singleton', slice(16, 32))])/2
                physical_loss = sum(physical_per_row[sl].mean()/data['scales'][name]['physical']
                    for name, sl in [('group', slice(0, 16)), ('singleton', slice(16, 32))])/2
                nz = api.encode(ae, nx, {}, 'global')
                natural_loss = (ae.decode(nz)-nx).square().mean()/data['natural_scale']
                paired_loss = ((ae.decode(z)-x).square().mean()+(ae.decode(edited_z)-hs).square().mean())/(2*data['paired_scale'])
                loss = (.5*physical_loss if method == 'physical' else .25*physical_loss+.25*code_loss)+.25*natural_loss+.25*paired_loss
                assert torch.isfinite(loss)
                if step == start_step:
                    z.retain_grad()
                    edited_z.retain_grad()
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in [*ae.parameters(), a] if p.grad is not None)
                if step == start_step:
                    write(cell/'gradient_check.json', dict(clean_branch=float(z.grad.norm()),
                        paired_branch=float(edited_z.grad.norm()), A=float(a.grad.norm()),
                        encoder=float(ae.encoder.weight.grad.norm()), decoder=float(ae.decoder.weight.grad.norm())))
                    assert z.grad.norm() > 0 and edited_z.grad.norm() > 0 and a.grad.norm() > 0
                api.constrain_gradients(ae, trainable)
                optimizer.step()
                scheduler.step()
                project_rows(a)
                api.normalize_decoder(ae, trainable)
                assert torch.equal(ae.b_dec, frozen['b_dec'])
                assert a.min() >= 0 and a.sum(1).max() <= 1.00001
                item = dict(step=step+1, loss=float(loss.detach()), physical_loss=float(physical_loss.detach()),
                    code_loss=float(code_loss.detach()), natural_fvu=float(natural_loss.detach()),
                    paired_fvu=float(paired_loss.detach()), code_error_energy=float(code_per_row.detach().mean()),
                    decoded_code_error_energy=decoded_error_energy,
                    physical_error_energy=float(physical_per_row.detach().mean()), A_row_sum_max=float(a.detach().sum(1).max()),
                    zero_source_rows=int((displacement.square().sum(-1) == 0).sum()))
                with (cell/'loss_trace.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(item)+'\n')
                encoded_states += 320
                backwards += 1
                if step+1 in cfg['checkpoint_steps']:
                    checkpoint = cell/f'step_{step+1}.pt'
                    torch.save(dict(dictionary={k: v.detach().cpu() for k, v in ae.state_dict().items()},
                        A=a.detach().cpu(), source_members=torch.tensor(data['members']),
                        source_part=torch.tensor(data['part_ids']), metadata=dict(metadata, step=step+1)), checkpoint)
                    temporary = cell/'resume_state.pt.tmp'
                    torch.save(dict(step=step+1, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), temporary)
                    os.replace(temporary, cell/'resume_state.pt')
                    quality = measure_quality(ae, data['quality'], api, work.device)
                    write(cell/f'quality_{step+1}.json', quality)
                    work.record(kind='quality', task='paired_code', component=f'seed{seed}_{method}',
                        row_id=step+1, method=method, seed=seed, target_seed=seed, **quality)
                    record = dict(metadata, path=str(checkpoint), step=step+1, sha256=sha256(checkpoint), quality=quality)
                    if step+1 == cfg['steps']:
                        final.append(record)
                        write(work.run/'final_checkpoints.json', dict(checkpoints=final))
                if (step+1) % 32 == 0 or step+1 in cfg['checkpoint_steps']:
                    work.progress('TRAINING', target_seed=seed, method=method, step=step+1,
                        total_steps=cfg['steps'], loss=item['loss'], peak_cuda_bytes=torch.cuda.max_memory_allocated())
                if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                    raise TimeoutError('Training run exceeded allocated driver time')
            del ae, a, optimizer, scheduler
            torch.cuda.empty_cache()
    write(work.run/'execution_cost.json', dict(encoded_training_states=encoded_states,
        sae_backward_calls=backwards, model_sequence_forwards=0, model_token_forwards=0,
        bytes=sum(p.stat().st_size for p in work.run.rglob('*') if p.is_file())))
    work.checks.update(all_cells_complete=len(final) == 2*len(cfg['target_seeds']),
        fixed_b_dec=True, feasible_A=True, both_encoder_branches=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sys.path[:0] = [cfg['dictionary_overlay_dir'], cfg['dictionary_source_dir']]
    import ccad.incremental_function_blocks as api
    Path(cfg['run_storage_root']).mkdir(parents=True, exist_ok=True)
    work = MultisiteWork(cfg, args.config, ['scripts/train_paired_code_correspondence.py',
        'src/ccad/incremental_function_blocks.py', 'scripts/train_shift_program.py',
        'scripts/train_grammar_functional_blocks.py', 'scripts/train_grammar_material_support.py',
        'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
    error = None
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
            dtype='float32', autocast=False, matmul_precision='highest', cpu_threads=4)
        work.checked(Path(cfg['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py', 'AutoEncoderTopK', 'MIT')
        train(work, cfg, api)
    except Exception:
        error = traceback.format_exc()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
