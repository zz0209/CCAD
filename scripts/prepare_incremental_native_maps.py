from pathlib import Path
import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch
import transformers

from ccad.artifacts import sha256
from run_causalgym_multisite import MultisiteWork, write
from train_grammar_material_support import load_config
from fit_component_correspondence import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sys.path[:0] = [cfg['dictionary_overlay_dir'], cfg['dictionary_source_dir']]
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    work = MultisiteWork(cfg, args.config, ['scripts/prepare_incremental_native_maps.py',
        'scripts/fit_component_correspondence.py', 'scripts/train_grammar_material_support.py',
        'scripts/run_causalgym_multisite.py', 'src/ccad/artifacts.py'])
    error = None
    try:
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, torch=torch.__version__, transformers=transformers.__version__,
            dtype='float32', matmul_precision='highest', autocast=False, cpu_threads=2)
        source_run = Path(cfg['source_run'])
        original_cfg_path = work.checked(source_run/'config.resolved.json')
        original_cfg = json.loads(original_cfg_path.read_text())
        original_solver = work.checked(source_run/'source_snapshot/scripts/fit_component_correspondence.py')
        assert sha256(original_solver) == sha256(Path('scripts/fit_component_correspondence.py'))
        assert original_cfg['fit_steps'] == 800 and original_cfg['ridge_fraction'] == .001
        assert original_cfg['target_pool'] == 512 and original_cfg['target_members_per_component'] == 64
        panel_path = work.checked(source_run/'panel.json')
        rows = [row for row in json.loads(panel_path.read_text())['rows'] if row['split'] == 'fit']
        assert len(rows) == 384
        assert all(sum(row['task'] == task for row in rows) == 128 for task in original_cfg['source_tasks'])
        write(work.run/'fit_panel.json', dict(rows=rows, original_panel_sha256=sha256(panel_path)))
        hidden_path = work.run/'fit_hidden.npz'
        if cfg.get('reuse_hidden_run'):
            prior = Path(cfg['reuse_hidden_run'])
            previous_panel = json.loads(work.checked(prior/'fit_panel.json').read_text())
            assert previous_panel['rows'] == rows
            prior_config = json.loads(work.checked(prior/'config.resolved.json').read_text())
            assert prior_config['model_revision'] == cfg['model_revision']
            with np.load(work.checked(prior/'fit_hidden.npz'), allow_pickle=False) as data:
                hidden = data['hidden'].copy()
            hidden_path = prior/'fit_hidden.npz'
        else:
            for filename in ('config.json', 'tokenizer.json', 'model.safetensors'):
                work.checked(Path(cfg['model_local_dir'])/filename)
            model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'], local_files_only=True,
                dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
            model.requires_grad_(False)
            model.config.use_cache = False
            tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'], local_files_only=True)
            module = model.get_submodule(cfg['hook_module_path'])
            hidden = np.empty((384, 1024), np.float32)
            positions, captured = None, None
            def hook(module, arguments, output):
                nonlocal captured
                values = output[0] if isinstance(output, tuple) else output
                captured = values[torch.arange(len(values), device=work.device), positions].detach()
            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad(), torch.autocast(device_type=work.device.type, enabled=False):
                    for start in range(0, 384, 16):
                        batch = rows[start:start+16]
                        tokens = torch.full((32, 64), tokenizer.eos_token_id, dtype=torch.long, device=work.device)
                        for i, row in enumerate(batch):
                            for j, name in enumerate(('good', 'bad')):
                                assert len(row[name]) <= 64
                                tokens[2*i+j, :len(row[name])] = torch.tensor(row[name], device=work.device)
                        positions = torch.tensor([row['position'] for row in batch], device=work.device).repeat_interleave(2)
                        model.transformer(tokens, use_cache=False)
                        assert float((captured[::2]-captured[1::2]).abs().max()) < .001
                        hidden[start:start+16] = captured[::2].cpu().numpy()
                        work.sequence_forwards += 32
                        work.token_forwards += tokens.numel()
                        work.progress('ORIGINAL_FIT_CAPTURE', pairs=start+16, total_pairs=384)
                        if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                            raise TimeoutError('Native reference preparation budget exceeded')
            finally:
                handle.remove()
            np.savez_compressed(hidden_path, hidden=hidden)
            del model
            torch.cuda.empty_cache()
        states = torch.tensor(hidden, device=work.device)
        def load_sae(path):
            state = torch.load(work.checked(path), map_location=work.device, weights_only=True)
            ae = AutoEncoderTopK(1024, 8192, 64).to(work.device)
            ae.load_state_dict(state)
            ae.eval().requires_grad_(False)
            return ae
        def codes(ae):
            with torch.no_grad():
                return torch.cat([ae.encode(states[i:i+256]) for i in range(0, len(states), 256)]).double()
        source_path = cfg['source_checkpoint']
        source = load_sae(source_path)
        gate_path = work.checked(source_run/'topk_s1_source.npz')
        sg = torch.tensor(np.load(gate_path)['gate'], device=work.device)
        sp = torch.where(sg.sum(1) > 0)[0]
        assert len(sp) == 192
        zs, ds = codes(source), source.decoder.weight.T.double()
        se = zs[:, sp].square().mean(0)*ds[sp].square().sum(1)
        index = dict(source_seed=1, source_checkpoint_sha256=sha256(Path(source_path)),
            source_gate_sha256=sha256(gate_path), fit_panel_sha256=sha256(panel_path),
            fit_hidden=str(hidden_path), fit_hidden_sha256=sha256(hidden_path), fit_rows=384,
            steps=800, ridge_fraction=.001, target_pool=512, members_per_component=64, maps=[])
        for seed in cfg['native_target_seeds']:
            target_path = Path(cfg['target_checkpoint_template'].format(seed=seed))
            target = load_sae(target_path)
            zt, dt = codes(target), target.decoder.weight.T.double()
            te = zt.square().mean(0)*dt.square().sum(1)
            cross = (zs[:, sp].T@zt/384)*(ds[sp]@dt.T)
            aff = cross/se.sqrt().clamp_min(1e-9)[:, None]/te.sqrt().clamp_min(1e-9)[None, :]
            tp = torch.argsort(aff.abs().max(0).values, descending=True, stable=True)[:512]
            x, d = zt[:, tp], dt[tp]
            K = (x.T@x/384)*(d@d.T)
            B = cross[:, tp].T@sg[sp].double()
            full, full_info = fit(K, B, steps=800, ridge_fraction=.001, capacity=True)
            allowed = torch.zeros_like(full, dtype=torch.bool)
            for component in range(3):
                allowed[torch.argsort(full[:, component].square()*K.diag(), descending=True, stable=True)[:64], component] = True
            partition, diagnostics = fit(K, B, steps=800, ridge_fraction=.001, allowed=allowed, capacity=True)
            output = work.run/f'topk_s1_t{seed}_map.npz'
            np.savez_compressed(output, source_members=sp.cpu().numpy(), source_gate=sg[sp].cpu().numpy(),
                target_members=tp.cpu().numpy(), partition64=partition.float().cpu().numpy())
            item = dict(source_seed=1, target_seed=seed, path=str(output), sha256=sha256(output),
                target_checkpoint_sha256=sha256(target_path), full_fit=full_info, constrained_fit=diagnostics)
            if seed == 2:
                with np.load(work.checked(source_run/'topk_s1_t2_map.npz'), allow_pickle=False) as previous:
                    same_pool = np.array_equal(previous['target_members'], tp.cpu().numpy())
                    old_dense = np.zeros((8192, 3), np.float64)
                    new_dense = np.zeros_like(old_dense)
                    old_dense[previous['target_members']] = previous['partition64']
                    new_dense[tp.cpu().numpy()] = partition.float().cpu().numpy()
                    difference = new_dense-old_dense
                    item['original_reproduction'] = dict(same_target_pool=same_pool,
                        same_source_members=np.array_equal(previous['source_members'], sp.cpu().numpy()),
                        maximum_absolute_partition_difference=float(np.abs(difference).max()),
                        rms_partition_difference=float(np.sqrt(np.mean(difference**2))),
                        changed_positive_support=int(np.count_nonzero((new_dense > 0) != (old_dense > 0))))
                    write(work.run/'original_s1_t2_reproduction.json', item['original_reproduction'])
            index['maps'].append(item)
            write(work.run/'native_map_index.json', index)
            work.record(kind='native_fit', task='R26_original_fit', component=f's1_t{seed}',
                mode='partition64', method='native', seed=1, target_seed=seed, row_id=0,
                fit_rows=384, maximum_row_sum=diagnostics['maximum_row_sum'])
            work.progress('NATIVE_REFERENCE_SAVED', target_seed=seed, maps=len(index['maps']))
            del target, zt, dt
        work.checks.update(all_maps=len(index['maps']) == len(cfg['native_target_seeds']),
                           original_fit_rows=True, original_solver_identity=True, source1_only=True)
    except Exception:
        error = traceback.format_exc()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
