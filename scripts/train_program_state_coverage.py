import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    c = json.loads(Path(c['base_config']).read_text()) | c
    files = ['scripts/train_program_state_coverage.py', 'scripts/run_shift_explanation.py',
             'scripts/train_shift_dictionaries.py', 'scripts/run_causalgym_multisite.py',
             'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py']
    w = MultisiteWork(c, args.config, files)
    handles, error = [], None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_device(c['device'])
        torch.cuda.reset_peak_memory_stats()
        w.torch, w.device = torch, torch.device(c['device'])
        w.environment = dict(python=sys.executable, torch=torch.__version__,
                             numpy=np.__version__, transformers=transformers.__version__)
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py',
                  'Pinned TopK encoder and normalized decoder', 'MIT')
        manifest = json.loads(w.checked(c['source_manifest']).read_text())
        groups, _ = source_groups(w.checked(c['notebook']), manifest['members'])
        sites = list(manifest['members'])
        bank = np.load(w.checked(c['source_parameters']))
        source = {s: {k: torch.tensor(bank[s+'__'+k], device=w.device)
                       for k in ['encoder', 'encoder_bias', 'decoder', 'center']} for s in sites}
        panel = json.loads(w.checked(Path(c['frozen_source_run'])/'panel.json').read_text())
        dev = []
        for y in [0, 1]:
            for g in [0, 1]:
                cell = sorted((r for r in panel['rows'] if r['split'] == 'dev' and
                               r['label'] == y and r['gender'] == g), key=lambda r:r['document_sha256'])
                dev.extend(cell[:c['development_per_group']])
        excluded = {r['document_sha256'] for r in dev}
        fit = sorted((r for r in panel['rows'] if r['split'] == 'dev' and
                      r['document_sha256'] not in excluded), key=lambda r:r['document_sha256'])[:c['fit_documents']]
        assert len(fit) == c['fit_documents'] and len(excluded) == len(dev)
        assert not excluded.intersection(r['document_sha256'] for r in fit)
        write(w.run/'membership.json', dict(fit_documents=[r['document_sha256'] for r in fit],
              development_documents=[r['document_sha256'] for r in dev], evidence='exposed_development',
              training_queries=['clean', 'full', *groups], evaluation_queries=c['queries']))
        token_path = w.checked(c['paired_tokens'])
        token_bank = np.memmap(token_path, dtype='<u2', mode='r').reshape(-1, 128)
        offset = c['natural_offset']
        natural_rows = [dict(tokens=token_bank[i].astype('int64').tolist())
                        for i in range(offset, offset+c['natural_sequences'])]
        write(w.run/'natural_membership.json', dict(offset=offset, count=len(natural_rows),
              fit_end=offset+len(natural_rows), relation_sequence_range=[0, c['fit_sequences']]))
        assert offset >= c['fit_sequences']
        for name in ['config.json', 'model.safetensors', 'tokenizer.json']:
            w.checked(Path(c['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').to(w.device).eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        mode, mask, observed = 'clean', None, {}
        q = {s: torch.zeros(len(manifest['members'][s]), device=w.device) for s in sites}

        def hook(site):
            def apply(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                if mode != 'clean':
                    sp = source[site]
                    z = torch.relu((h-sp['center'])@sp['encoder'].T+sp['encoder_bias'])
                    h = h-(z*q[site])@sp['decoder']
                # 同时保存当前位置自身干预及前序干预传播后的实际状态。
                observed[site] = h.detach()
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return apply

        for s in sites:
            handles.append(site_module(model, s).register_forward_hook(hook(s)))

        def capture(rows, query, folder):
            nonlocal mode, mask
            mode = query
            selected = list(groups) if query == 'full' else query.split('+')
            for s in sites:
                q[s] = torch.tensor([query != 'clean' and any(i in groups[g].get(s, [])
                    for g in selected) for i in manifest['members'][s]], device=w.device, dtype=torch.float32)
            chunks = {s: [] for s in sites}
            for off in range(0, len(rows), c['capture_batch']):
                rr = rows[off:off+c['capture_batch']]
                length = max(len(r['tokens']) for r in rr)
                ids = torch.zeros((len(rr), length), device=w.device, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, row in enumerate(rr):
                    ids[j, :len(row['tokens'])] = torch.tensor(row['tokens'], device=w.device)
                    mask[j, :len(row['tokens'])] = 1
                with torch.no_grad():
                    model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                for s in sites:
                    chunks[s].append(observed[s][mask.bool()].cpu())
                w.sequence_forwards += len(rr)
                w.token_forwards += ids.numel()
                w.progress('CAPTURE', split=folder.name, query=query, done=min(off+len(rr), len(rows)), total=len(rows))
            folder.mkdir(parents=True, exist_ok=True)
            for s in sites:
                values = torch.cat(chunks[s])
                assert values.shape[1] == 512 and bool(torch.isfinite(values).all())
                torch.save(values, folder/f'{s}.pt')

        cache = w.run/'state_cache'
        capture(natural_rows, 'clean', cache/'natural')
        for query in ['clean', 'full', *groups]:
            capture(fit, query, cache/'fit'/query)
        for query in ['clean', *c['queries']]:
            capture(dev, query, cache/'evaluation'/query)
        for handle in handles:
            handle.remove()
        handles.clear()
        del model, observed
        torch.cuda.empty_cache()
        quality, checkpoints = [], []
        for site in sites:
            initial = torch.load(w.checked(Path(c['target_directory'])/f'{site}_seed{c["target_seed"]}.pt'),
                                 map_location='cpu', weights_only=True)
            target = AutoEncoderTopK(512, len(initial['encoder.weight']), int(initial['k'])).to(w.device)
            natural = torch.load(cache/'natural'/f'{site}.pt', weights_only=True).to(w.device)
            states = {name: torch.load(cache/'fit'/name/f'{site}.pt', weights_only=True).to(w.device)
                      for name in ['clean', 'full', *groups]}
            assert len({tuple(x.shape) for x in states.values()}) == 1
            reference = torch.load(cache/'evaluation'/'clean'/f'{site}.pt', weights_only=True).to(w.device)

            def measure(variant):
                target.requires_grad_(False)
                with torch.no_grad():
                    rr = torch.cat([target(x) for x in reference.split(c['state_batch'])])
                    variance = (reference-reference.mean(0)).square().sum()
                    for query in ['clean', *c['queries']]:
                        x = torch.load(cache/'evaluation'/query/f'{site}.pt', weights_only=True).to(w.device)
                        recon, count = [], 0
                        alive = torch.zeros(len(initial['encoder.weight']), device=w.device, dtype=torch.bool)
                        for xx in x.split(c['state_batch']):
                            z = target.encode(xx)
                            recon.append(target.decode(z))
                            count += int((z > 0).sum())
                            alive |= (z > 0).any(0)
                        rec = torch.cat(recon)
                        change = x-reference
                        item = dict(method=variant, site=site, query=query, states=len(x),
                            reconstruction_error=float((rec-x).square().sum()),
                            change_error=float((rec-rr-change).square().sum()),
                            change_energy=float(change.square().sum()),
                            clean_variance=float(variance), fve=float(1-(rec-x).square().sum()/variance),
                            l0=count/len(x), alive=int(alive.sum()),
                            decoder_norm_error=float((target.decoder.weight.norm(dim=0)-1).abs().max()))
                        quality.append(item)
                write(w.run/'representation_quality.json', quality)

            target.load_state_dict(initial)
            measure('initial')
            for variant in c['variants']:
                target.load_state_dict(initial)
                target.requires_grad_(True)
                optimizer = torch.optim.AdamW(target.parameters(), lr=c['learning_rate'], weight_decay=0.)
                generator = torch.Generator(device=w.device).manual_seed(c['training_seed'])
                for step in range(c['steps']):
                    ni = torch.randint(len(natural), (c['state_batch'],), device=w.device, generator=generator)
                    bi = torch.randint(len(states['clean']), (c['state_batch'],), device=w.device, generator=generator)
                    query = 'clean' if variant == 'natural' else 'full' if variant == 'whole' else list(groups)[step % len(groups)]
                    x, y = natural[ni], states[query][bi]
                    loss = .5*((target(x)-x).square().mean()/x.square().mean().clamp_min(1e-8)
                              +(target(y)-y).square().mean()/states['clean'][bi].square().mean().clamp_min(1e-8))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite state reconstruction loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(target.parameters(), 1.)
                    optimizer.step()
                    with torch.no_grad():
                        target.decoder.weight.div_(target.decoder.weight.norm(dim=0).clamp_min(1e-10))
                    if (step+1) % c['log_every'] == 0 or step+1 == c['steps']:
                        w.record(kind='training', task=site, component=query, row_id=step+1, method=variant,
                                 step=step+1, loss=float(loss.detach()))
                        w.progress('TRAIN', site=site, variant=variant, step=step+1, total=c['steps'],
                                   loss=float(loss.detach()), peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    if (step+1) % c['checkpoint_every'] == 0 or step+1 == c['steps']:
                        folder = w.run/variant/f'step_{step+1}'
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder/f'{site}_seed{c["target_seed"]}.pt'
                        torch.save(target.state_dict(), path)
                        checkpoints.append(dict(variant=variant, site=site, step=step+1, path=str(path)))
                        write(w.run/'checkpoints.json', checkpoints)
                    if time.perf_counter()-w.wall_start > c['budget_seconds']:
                        raise TimeoutError('State coverage allocated driver time reached')
                measure(variant)
                del optimizer
            del target, natural, states, reference
            torch.cuda.empty_cache()
        w.checks.update(all_sites=len({r['site'] for r in quality}) == len(sites),
                        all_methods=len({r['method'] for r in quality}) == len(c['variants'])+1,
                        disjoint_documents=True, all_finite=all(np.isfinite(r['fve']) for r in quality))
    except Exception:
        error = traceback.format_exc()
    finally:
        for handle in handles:
            handle.remove()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
