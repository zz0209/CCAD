from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
import transformers

from ccad.intervention_transport import transport_delta, refine_columns
from check_request_geometry import identity
from train_shift_dictionaries import site_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--documents', type=int, choices=[2, 8], required=True)
    args = parser.parse_args()
    args.run.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    c = json.loads(args.config.read_text())
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    membership_path = Path(c['run_storage_root']) / c['run_id'] / 'membership.json'
    membership = json.loads(membership_path.read_text())
    buckets = {}
    for row in membership['human_rows']:
        buckets.setdefault((row['profession'], row['gender']), row)
    rows = list(buckets.values())[:args.documents]
    if len(rows) != args.documents:
        raise ValueError('Missing profession/gender cells')
    sites = membership['site_order']
    bank = np.load(c['source_parameters'])
    source = {s: {k: torch.from_numpy(bank[s + '__' + k])
                  for k in ['center', 'encoder', 'encoder_bias', 'decoder']} for s in sites}
    roots = torch.load(c['source_metric_cache'], map_location='cpu', weights_only=True)
    conditional = torch.load(c['conditional_profile_cache'], map_location='cpu', weights_only=True)
    checkpoint = Path(c['target_directory']) / f'embed_seed{c["target_seed"]}.pt'
    sd = torch.load(checkpoint, map_location='cpu', weights_only=True)
    target = AutoEncoderTopK(512, len(sd['encoder.weight']), int(sd['k']))
    target.load_state_dict(sd)
    target.requires_grad_(False)
    model = transformers.AutoModelForCausalLM.from_pretrained(c['model_local_dir'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval()
    model.requires_grad_(False)
    probe_path = Path(c['frozen_source_run']) / 'probe.npz'
    probe = np.load(probe_path)
    weight, bias = torch.from_numpy(probe['weight']), torch.from_numpy(probe['bias'])
    query = {}
    observed = {}
    edit = None
    clean = True

    def hook(site):
        def apply(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if not clean:
                if site == 'embed' and torch.is_grad_enabled():
                    h = h.detach().requires_grad_(True)
                sp = source[site]
                activation = torch.relu((h - sp['center']) @ sp['encoder'].T + sp['encoder_bias'])
                h = h - (activation * query[site]) @ sp['decoder']
                if site == 'embed' and edit is not None:
                    h = h + edit
            if site in ['embed', 'resid_4']:
                observed[site] = h
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return apply

    handles = [site_module(model, s).register_forward_hook(hook(s)) for s in sites]

    def forward(ids):
        model.gpt_neox(ids, use_cache=False)
        pool = observed['resid_4'].mean(1)
        return (pool @ weight.T + bias).reshape(())

    cache = {}
    results = []
    names = ['full', 'pronouns', 'names', 'associated_words']
    if args.documents == 2:
        names = ['full', 'pronouns']
    for row_index, row in enumerate(rows):
        ids = torch.tensor([row['tokens']], dtype=torch.long)
        clean = True
        with torch.no_grad():
            clean_response = float(forward(ids))
            h = torch.nn.functional.embedding(ids, model.gpt_neox.embed_in.weight)
            for method in ['global', 'conditional']:
                for token in torch.unique(ids).tolist():
                    if (method, token) in cache:
                        continue
                    state = h[ids == token][:1]
                    sp = source['embed']
                    zs = torch.relu((state - sp['center']) @ sp['encoder'].T + sp['encoder_bias'])
                    if not (zs > 0).any():
                        cache[method, token] = torch.zeros((512, zs.shape[-1]))
                        continue
                    _, _, columns = transport_delta(state, target, sp, torch.ones(zs.shape[-1]),
                                                    target.encoder.weight, 2 * zs.shape[-1], False)
                    root = conditional[token] if method == 'conditional' and token in conditional else roots['human']['embed']
                    columns = refine_columns(state, target, sp, columns, 2 * zs.shape[-1], 128,
                                             metric_root=root, accelerate=True)
                    cache[method, token] = (target.decoder.weight @ columns[0]
                                           + sp['decoder'].T * zs[0])
        clean = False
        for name in names:
            query = {s: torch.tensor(v, dtype=torch.float32) for s, v in membership['human_queries'][name].items()}
            edit = None
            response = forward(ids)
            gradient = torch.autograd.grad(response, observed['embed'])[0].detach()
            source_response = float(response.detach())
            for method in ['global', 'conditional']:
                edit = torch.stack([cache[method, t] @ query['embed'] for t in row['tokens']])[None]
                contribution = (gradient * edit).sum(-1)
                linear = float(contribution.sum())
                with torch.no_grad():
                    target_response = float(forward(ids))
                    full_edit = edit
                    epsilon = 0.001
                    edit = epsilon * full_edit
                    positive_response = float(forward(ids))
                    edit = -epsilon * full_edit
                    negative_response = float(forward(ids))
                    edit = full_edit
                finite_derivative = (positive_response - negative_response) / (2 * epsilon)
                item = dict(document_sha256=row['document_sha256'], profession=row['profession'], gender=row['gender'],
                            request=name, method=method, clean_response=clean_response,
                            source_response=source_response, target_response=target_response,
                            actual_error=target_response-source_response, linear_error=linear,
                            finite_derivative=finite_derivative, derivative_epsilon=epsilon,
                            diagonal_energy=float(contribution.square().sum()),
                            joint_energy=linear ** 2, local_error_energy=float(edit.square().sum()))
                results.append(item)
            progress = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                            completed_documents=row_index, current_document=row_index + 1, request=name,
                            total_documents=len(rows), wall_seconds=time.perf_counter() - clock,
                            records=results)
            (args.run / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            print(json.dumps({k: v for k, v in progress.items() if k != 'records'}), flush=True)
    for handle in handles:
        handle.remove()
    summary = {}
    for method in ['global', 'conditional']:
        entries = [r for r in results if r['method'] == method]
        actual = np.array([r['actual_error'] for r in entries])
        predicted = np.array([r['linear_error'] for r in entries])
        diagonal = np.array([r['diagonal_energy'] for r in entries])
        finite = np.array([r['finite_derivative'] for r in entries])
        summary[method] = dict(actual_rms=float(np.sqrt(np.mean(actual ** 2))),
            relative_linear_error=float(np.linalg.norm(predicted - actual) / np.linalg.norm(actual)),
            pearson_actual_linear=float(np.corrcoef(actual, predicted)[0, 1]),
            joint_to_diagonal_energy=float(np.sum(predicted ** 2) / np.sum(diagonal)),
            relative_derivative_check=float(np.linalg.norm(finite-predicted)/np.linalg.norm(predicted)),
            maximum_derivative_difference=float(np.abs(finite-predicted).max()))
    inputs = [args.config, membership_path, c['source_parameters'], c['source_metric_cache'],
              c['conditional_profile_cache'], checkpoint, probe_path, __file__, 'src/ccad/intervention_transport.py',
              Path(c['model_local_dir']) / 'config.json', Path(c['model_local_dir']) / 'model.safetensors']
    result = dict(run_id=args.run.name, started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.perf_counter() - clock, python=sys.executable, torch=torch.__version__,
                  scope='Exposed development biographies; target execution at embedding followed by the source program',
                  inputs=[identity(p) for p in inputs], summary=summary, records=results)
    (args.run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
