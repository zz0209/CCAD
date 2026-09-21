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
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--documents', type=int, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.base_config.read_text())
    cfg.update(run_id=args.run_id, diagnostic_documents=args.documents,
               generator_script='scripts/diagnose_conditional_gating.py',
               purpose='Measure TopK membership changes under source conditional states.',
               budget_seconds=180, scope='Local field diagnostic on exposed source intervention states.')
    work = MultisiteWork(cfg, args.base_config, ['scripts/diagnose_conditional_gating.py',
        'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
        'scripts/train_shift_dictionaries.py', 'src/ccad/artifacts.py'])
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
        spec = json.loads(work.checked(cfg['request_spec']).read_text())
        panel = json.loads(work.checked(cfg['evaluation_panel']).read_text())
        source = json.loads(work.checked(cfg['source_manifest']).read_text())
        available = {r['document_sha256']: r for r in panel['rows']}
        rows = [available[k] for k in spec['documents'][:args.documents]]
        assert len(rows) == args.documents
        sites = list(source['members'])
        sb = np.load(work.checked(cfg['source_parameters']))
        params = {s: {k: torch.tensor(sb[s+'__'+k], device=work.device)
                      for k in ['encoder', 'encoder_bias', 'decoder', 'center']} for s in sites}
        old = np.load(work.checked(Path(cfg['relation_run'])/'relation.npz'))
        sys.path.extend([cfg['dictionary_source_dir'], cfg['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        targets, relations = {}, {}
        for site in sites:
            state = torch.load(work.checked(Path(cfg['target_directory'])/f'{site}_seed{cfg["target_seed"]}.pt'),
                               map_location=work.device, weights_only=True)
            target = AutoEncoderTopK(512, len(state['encoder.weight']), int(state['k'])).to(work.device)
            target.load_state_dict(state)
            target.requires_grad_(False)
            targets[site] = target
            relations[site] = torch.tensor(old[site+'__native'], device=work.device, dtype=torch.float32)
        for name in ['config.json', 'model.safetensors']:
            work.checked(Path(cfg['model_local_dir'])/name)
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        queries = [q for q in spec['queries'] if q['role'] == 'delete']
        assert len(queries) == len(spec['nodes'])
        qweights = {(q['name'], s): torch.tensor(q['weights'][s], device=work.device)
                    for q in queries for s in sites}
        nodes_by_site = {s: [n for n in spec['nodes'] if n['site'] == s] for s in sites}
        clean, current, batch, mask, cache = True, None, [], None, {}

        def source_code(h, site):
            p = params[site]
            return torch.relu((h-p['center'])@p['encoder'].T+p['encoder_bias'])

        def hook(site):
            def apply(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                if clean:
                    cache[site] = (targets[site].encode(h), source_code(h, site))
                    return output
                early = next(n for n in spec['nodes'] if 'delete_'+n['name'] == current['name'])
                if sites.index(site) > early['site_index']:
                    target, r = targets[site], relations[site]
                    z0, s0 = cache[site]
                    z1, _, _, relu1 = target.encode(h, return_topk=True)
                    frozen = relu1*(z0 > 0)
                    s1 = source_code(h, site)
                    for node in nodes_by_site[site]:
                        q = qweights['delete_'+node['name'], site]
                        a = r@q
                        actual = ((z1-z0)*a)@target.decoder.weight.T
                        fixed = ((frozen-z0)*a)@target.decoder.weight.T
                        source_change = ((s1-s0)*q)@params[site]['decoder']
                        metrics = dict(actual_error=(actual-source_change).square().sum(-1),
                            fixed_gate_error=(fixed-source_change).square().sum(-1),
                            source_change=source_change.square().sum(-1),
                            gate_change=(actual-fixed).square().sum(-1),
                            target_change=actual.square().sum(-1),
                            changed_members=((z1 > 0) != (z0 > 0)).sum(-1).float()/int(target.k))
                        for j, row in enumerate(batch):
                            work.record(component=node['name'], condition=early['name'],
                                row_id=f'{early["name"]}/{node["name"]}/{row["document_sha256"]}',
                                document=row['document_sha256'], tokens=int(mask[j].sum()),
                                seed=cfg['target_seed'], split='exposed_development',
                                **{k: float((v[j]*mask[j]).sum()) for k, v in metrics.items()})
                q = qweights[current['name'], site]
                if bool(q.any()):
                    h = h-(source_code(h, site)*q)@params[site]['decoder']
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return apply

        for site in sites:
            handles.append(site_module(model, site).register_forward_hook(hook(site)))
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset+4]
            length = max(len(r['tokens']) for r in batch)
            ids = torch.zeros((len(batch), length), dtype=torch.long, device=work.device)
            mask = torch.zeros_like(ids)
            for j, row in enumerate(batch):
                ids[j, :len(row['tokens'])] = torch.tensor(row['tokens'], device=work.device)
                mask[j, :len(row['tokens'])] = 1
            with torch.no_grad():
                clean, cache = True, {}
                model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                work.sequence_forwards += len(batch)
                work.token_forwards += ids.numel()
                clean = False
                for current in queries:
                    model.gpt_neox(ids, attention_mask=mask, use_cache=False)
                    work.sequence_forwards += len(batch)
                    work.token_forwards += ids.numel()
            work.progress('SOURCE_STATES', documents=min(offset+4, len(rows)), total=len(rows))
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Gating diagnostic budget exceeded')
        assert len(work.metrics) == 198*len(rows)
        sums = {key: sum(r[key] for r in work.metrics) for key in
                ['actual_error', 'fixed_gate_error', 'source_change', 'gate_change', 'target_change']}
        assert all(np.isfinite(v) and v >= 0 for v in sums.values()) and sums['source_change'] > 0
        write(work.run/'GATING_ANALYSIS.json', dict(evidence_level='exposed_development',
            documents=[r['document_sha256'] for r in rows], records=len(work.metrics), sums=sums,
            actual_change_nrmse=float(np.sqrt(sums['actual_error']/sums['source_change'])),
            fixed_gate_change_nrmse=float(np.sqrt(sums['fixed_gate_error']/sums['source_change'])),
            diagnostic='Local source-induced states; frozen masks are a counterfactual diagnostic, not target execution.'))
        work.checks.update(all_pairs_measured=True, finite_statistics=True)
    except Exception:
        error = traceback.format_exc()
    finally:
        for handle in handles:
            handle.remove()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
