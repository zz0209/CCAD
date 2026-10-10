import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, write

DESCRIPTION = ('Select the grammatical explanation of an SAE with the R26 contrast rule: first-order credit of each '
               'feature to the sentence margin at the shared prefix, minus its mean positive credit on the other two '
               'tasks, keeping the top features of each task among those that win it.')
SOURCES = ['scripts/select_grammar_explanation.py', 'scripts/run_causalgym_multisite.py',
           'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py']


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    w = MultisiteWork(c, args.config, SOURCES)
    error = None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        device = torch.device(c['device'])
        torch.cuda.set_device(device)
        w.torch, w.device = torch, device
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        for name in c.get('model_files', ['config.json', 'model.safetensors']):
            w.checked(Path(c['model_dir'])/name, c['model_label'])
        model = transformers.AutoModelForCausalLM.from_pretrained(
            c['model_dir'], local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(device)
        model.requires_grad_(False)
        state = torch.load(w.checked(c['sae_checkpoint'], c['sae_label']), map_location=device, weights_only=True)
        sae = AutoEncoderTopK(model.config.hidden_size, state['encoder.weight'].shape[0], int(state['k'])).to(device)
        sae.load_state_dict(state)
        sae.eval().requires_grad_(False)
        w.environment = dict(python=sys.executable, torch=torch.__version__, transformers=transformers.__version__)
        panel = json.loads(w.checked(c['panel'], 'R26 grammar panel').read_text())
        rows = [r for r in panel['rows'] if r['split'] == 'source_selection' and r['task'] in c['tasks']]
        module = model.get_submodule(c['hook_module_path'])
        dim = model.config.hidden_size
        hidden, grads = [], []
        batch = c['gradient_batch_pairs']
        for off in range(0, len(rows), batch):
            rr = rows[off:off+batch]
            padded = rr+[rr[0]]*(batch-len(rr))
            ids = torch.full((2*batch, c['max_length']), c['eos_token_id'], device=device, dtype=torch.long)
            lengths, positions = [], []
            for i, r in enumerate(padded):
                for j, key in enumerate(['good', 'bad']):
                    ids[2*i+j, :len(r[key])] = torch.tensor(r[key], device=device)
                    lengths.append(len(r[key]))
                    positions.append(r['position'])
            idx = torch.arange(2*batch, device=device)
            pos = torch.tensor(positions, device=device)
            cache = {}

            def hook(m, i, out):
                h = out[0] if isinstance(out, tuple) else out
                leaf = h.detach().requires_grad_(True)
                cache['leaf'] = leaf
                return (leaf,)+out[1:] if isinstance(out, tuple) else leaf
            handle = module.register_forward_hook(hook)
            with torch.enable_grad():
                logits = model(ids, use_cache=False).logits
                per = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
                mask = torch.arange(c['max_length']-1, device=device)[None, :] < (torch.tensor(lengths, device=device)-1)[:, None]
                total = (per*mask).double().sum(1).reshape(-1, 2)
                margin = total[:, 0]-total[:, 1]
                grad = torch.autograd.grad(margin.sum(), cache['leaf'])[0][idx, pos].reshape(batch, 2, dim).sum(1)
            handle.remove()
            h = cache['leaf'].detach()[idx, pos].reshape(batch, 2, dim)
            assert float((h[:, 0]-h[:, 1]).abs().max()) < 1e-3
            hidden.append(h[:len(rr), 0])
            grads.append(grad[:len(rr)])
            w.progress('GRADIENTS', items=min(off+batch, len(rows)), total=len(rows))
        hidden, grads = torch.cat(hidden), torch.cat(grads)
        with torch.no_grad():
            codes = torch.cat([sae.encode(hidden[i:i+256]) for i in range(0, len(hidden), 256)])
            credit = codes*(grads@sae.decoder.weight)
            means = torch.stack([credit[[i for i, r in enumerate(rows) if r['task'] == task]].mean(0) for task in c['tasks']], 1)
            score = means-(means.clamp_min(0).sum(1, keepdim=True)-means.clamp_min(0))/2
            values, winner = score.max(1)
            gate = torch.zeros_like(means)
            selected = []
            for k in range(len(c['tasks'])):
                eligible = torch.where((winner == k) & (values > 0))[0]
                order = torch.argsort(score[eligible, k], descending=True, stable=True)
                ix = eligible[order[:c['members_per_component']]]
                gate[ix, k] = 1
                selected.append(dict(task=c['tasks'][k], members=ix.cpu().tolist(), eligible=len(eligible),
                                     selection_score=score[ix, k].cpu().tolist(), own_contribution=means[ix, k].cpu().tolist()))
                w.records([dict(kind='selection', task=c['tasks'][k], row_id=int(i), component=f'{k}:{int(i)}', method='contrast',
                                mode='select', seed=c['seeds'][0], target_seed=c['seeds'][0], operation='select',
                                score=float(score[i, k])) for i in ix.tolist()])
        np.savez_compressed(w.run/'source.npz', gate=gate.cpu().numpy(), mean_credit=means.cpu().numpy(),
                            selectivity_score=score.cpu().numpy())
        write(w.run/'source_selection.json', dict(components=selected, rows=len(rows)))
        if c.get('reference_gate'):
            reference = np.load(w.checked(c['reference_gate'], 'Reference selection'))['gate']
            write(w.run/'reference_check.json', dict(identical=bool(np.array_equal(reference, gate.cpu().numpy())),
                                                      overlap=int((reference*gate.cpu().numpy()).sum()),
                                                      members=int(gate.sum())))
        w.checks['full_components'] = all(len(s['members']) == c['members_per_component'] for s in selected)
    except Exception:
        error = traceback.format_exc()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
