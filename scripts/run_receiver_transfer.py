from pathlib import Path
import argparse
import json
import platform
import sys
import time
import traceback

import numpy as np
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module
import torch
import transformers


OPERATIONS = ['none', 'A', 'B_plural', 'B_singular', 'receiver_plural',
              'receiver_singular', 'receiver_both', 'restore_plural',
              'restore_singular', 'restore_both', 'receiver_zero', 'restore_zero']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    work = MultisiteWork(cfg, args.config, ['scripts/run_receiver_transfer.py',
        'scripts/train_shift_dictionaries.py', 'scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py'])
    handles = []
    error = None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        device = torch.device(cfg['device'])
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        work.torch, work.device = torch, device
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            matmul_precision='highest', cpu_threads=2)
        panel = json.loads(work.checked(cfg['panel']).read_text())
        rows = panel['rows']
        write(work.run/'panel.json', panel)
        bank = np.load(work.checked(cfg['source_parameters'], 'Published SFC feature parameters', 'MIT'))
        params = {site: {key: torch.tensor(bank[site+'__'+key], device=device)
            for key in ['encoder', 'encoder_bias', 'decoder', 'center']}
            for site in ['resid_3', 'attn_4']}
        assert bank['resid_3__ids'].tolist() == [18529]
        assert bank['attn_4__ids'].tolist() == [3982, 31148]
        for name in ['config.json', 'model.safetensors', 'tokenizer.json']:
            work.checked(Path(cfg['model_local_dir'])/name, 'Pinned Pythia70M', 'Apache-2.0')
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(device)
        model.requires_grad_(False)
        model.config.use_cache = False
        n = len(rows)
        logits = np.full((len(OPERATIONS), n, 2), np.nan, dtype=np.float32)
        arrays = {key: np.full(shape, np.nan, dtype=np.float32) for key, shape in {
            'h0': (n, 512), 'hA': (n, 512), 'b_codes0': (n, 2),
            'b_codesA': (n, 2), 'b_delta': (n, 2, 512), 'a_code': (n, 1)}.items()}
        mode = 'none'
        selected = []
        cache = {}
        positions_a = positions_b = row_index = None
        checks = dict(baseline_state_max_error=0., intervened_state_max_error=0.,
            receiver_zero_logit_error=0., restore_zero_logit_error=0.)

        def encode(site, value):
            p = params[site]
            return torch.relu((value-p['center'])@p['encoder'].T+p['encoder_bias'])

        def sender_hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if mode == 'A' or mode.startswith('restore_'):
                z = encode('resid_3', h[row_index, positions_a])
                if mode == 'A':
                    cache['a_code'] = z.detach().clone()
                updated = h.clone()
                updated[row_index, positions_a] -= z@params['resid_3']['decoder']
                return (updated, *output[1:]) if isinstance(output, tuple) else updated
            return output

        def receiver_hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            current = h[row_index, positions_b]
            if mode == 'none':
                cache['h0'] = current.detach().clone()
                cache['b_codes0'] = encode('attn_4', current).detach().clone()
                return output
            if mode == 'A':
                cache['hA'] = current.detach().clone()
                cache['b_codesA'] = encode('attn_4', current).detach().clone()
                cache['b_delta'] = (cache['b_codesA']-cache['b_codes0']).unsqueeze(-1)*params['attn_4']['decoder']
                return output
            background = 'hA' if mode.startswith('restore_') else 'h0'
            check_name = 'intervened_state_max_error' if background == 'hA' else 'baseline_state_max_error'
            checks[check_name] = max(checks[check_name], float((current-cache[background]).abs().max()))
            assert torch.allclose(current, cache[background], atol=2e-5, rtol=0)
            component = mode.split('_')[-1]
            if component == 'zero':
                delta = torch.zeros_like(current)
            elif mode.startswith('B_'):
                j = ['plural', 'singular'].index(component)
                delta = -cache['b_codes0'][:, j:j+1]*params['attn_4']['decoder'][j]
            else:
                delta = cache['b_delta'].sum(1) if component == 'both' else cache['b_delta'][:, ['plural', 'singular'].index(component)]
                if mode.startswith('restore_'):
                    delta = -delta
            updated = h.clone()
            updated[row_index, positions_b] += delta
            return (updated, *output[1:]) if isinstance(output, tuple) else updated

        handles.append(site_module(model, 'resid_3').register_forward_hook(sender_hook))
        handles.append(site_module(model, 'attn_4').register_forward_hook(receiver_hook))
        destination = work.run/'batches'
        destination.mkdir()
        write(work.run/'protocol.json', dict(operations=OPERATIONS,
            sender=dict(site='resid_3', ids=[18529], position='unique that token', action='zero deletion'),
            receiver=dict(site='attn_4', ids=[3982, 31148], position='last token'),
            contribution='delta_B = (z_B(hA)-z_B(h0)) D_B',
            receiver_operation='baseline attn4 last hidden + delta_B',
            restore_operation='pure A attn4 last hidden - delta_B',
            scope='Finite total transmission through a receiver group, with intermediate computations free',
            margin='recipient answer logit minus donor answer logit'))
        for start in range(0, n, cfg['batch_size']):
            selected = list(range(start, min(start+cfg['batch_size'], n)))
            rr = [rows[i] for i in selected]
            width = max(len(r['input_ids']) for r in rr)
            ids = torch.zeros((len(rr), width), device=device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, row in enumerate(rr):
                ids[j, :len(row['input_ids'])] = torch.tensor(row['input_ids'], device=device)
                mask[j, :len(row['input_ids'])] = 1
            positions_a = torch.tensor([r['a_position'] for r in rr], device=device)
            positions_b = torch.tensor([r['b_position'] for r in rr], device=device)
            row_index = torch.arange(len(rr), device=device)
            answer_ids = torch.tensor([[r['answer_id'], r['donor_answer_id']] for r in rr], device=device)
            assert torch.equal(positions_b, mask.sum(1)-1)
            cache = {}
            with torch.no_grad():
                for oi, mode in enumerate(OPERATIONS):
                    hidden = model.gpt_neox(ids, attention_mask=mask, use_cache=False).last_hidden_state
                    last = hidden[row_index, positions_b]
                    scores = model.get_output_embeddings()(last)
                    values = scores.gather(1, answer_ids).cpu().numpy()
                    assert np.isfinite(values).all()
                    logits[oi, selected] = values
                    work.sequence_forwards += len(rr)
                    work.token_forwards += int(mask.sum())
            for key in arrays:
                arrays[key][selected] = cache[key].cpu().numpy()
            checks['receiver_zero_logit_error'] = max(checks['receiver_zero_logit_error'],
                float(np.abs(logits[OPERATIONS.index('receiver_zero'), selected]-logits[0, selected]).max()))
            checks['restore_zero_logit_error'] = max(checks['restore_zero_logit_error'],
                float(np.abs(logits[OPERATIONS.index('restore_zero'), selected]-logits[1, selected]).max()))
            np.savez_compressed(destination/f'batch_{start:05d}.npz', row_indices=np.array(selected),
                answer_logits=logits[:, selected], **{k: v[selected] for k, v in arrays.items()})
            for oi, operation in enumerate(OPERATIONS):
                for i in selected:
                    row = rows[i]
                    work.record(kind='receiver_response', task=row['structure'], row_id=row['row_id'],
                        component=row['source_pair_sha256'], operation=operation, method='source',
                        seed=0, split='development', recipient_logit=float(logits[oi, i, 0]),
                        donor_logit=float(logits[oi, i, 1]), margin=float(logits[oi, i, 0]-logits[oi, i, 1]))
            work.progress('RECEIVER_TRANSFER', completed_rows=selected[-1]+1, total_rows=n)
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Receiver transfer budget exceeded; completed batches retained')
        assert all(value <= 2e-5 for value in checks.values())
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(work.run/'responses.npz', operations=np.array(OPERATIONS),
            answer_logits=logits, margin=logits[:, :, 0]-logits[:, :, 1], **arrays)
        write(work.run/'checks.json', checks)
        work.checks.update(shared_sender=True, baseline_receiver_identity=True,
            restore_background_identity=True, all_rows_complete=True, source_ids_frozen=True)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    for handle in handles:
        handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
