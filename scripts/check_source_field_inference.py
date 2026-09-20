from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

from ccad.intervention_transport import transport_delta, refine_columns
from ccad.source_field_inference import infer_source_fields
from check_request_geometry import identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    c = json.loads(args.config.read_text())
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    natural = torch.load(c['natural_cache'], map_location='cpu', weights_only=True)
    roots = torch.load(c['source_metric_cache'], map_location='cpu', weights_only=True)
    bank = np.load(c['source_parameters'])
    paths = [args.config, c['natural_cache'], c['source_metric_cache'], c['source_parameters'],
             __file__, 'src/ccad/source_field_inference.py', 'src/ccad/request_inference.py',
             'src/ccad/intervention_transport.py']
    results = []
    with torch.no_grad():
        for site in ['embed', 'mlp_0', 'resid_0']:
            source = {k: torch.from_numpy(bank[site + '__' + k])
                      for k in ['center', 'encoder', 'encoder_bias', 'decoder']}
            h = natural[site]
            zs = torch.relu((h - source['center']) @ source['encoder'].T + source['encoder_bias'])
            indices = (zs.sum(-1) > 0).nonzero().flatten()[:32]
            h, zs = h[indices], zs[indices]
            path = Path(c['target_directory']) / f'{site}_seed{c["target_seed"]}.pt'
            paths.append(path)
            sd = torch.load(path, map_location='cpu', weights_only=True)
            target = AutoEncoderTopK(512, len(sd['encoder.weight']), int(sd['k']))
            target.load_state_dict(sd)
            target.requires_grad_(False)
            p = zs.shape[-1]
            _, _, initial = transport_delta(h, target, source, torch.ones(p), target.encoder.weight, 2 * p, False)
            expected = refine_columns(h, target, source, initial, 2 * p, 128,
                                      metric_root=roots['human'][site], accelerate=True)
            actual = infer_source_fields(h, target, source['decoder'], -zs, 2 * p, roots['human'][site])
            difference = float((actual - expected).abs().max())
            assert torch.equal(actual, expected), (site, difference)
            results.append(dict(dataset='human_deletion', site=site, states=indices.tolist(),
                                maximum_difference=difference))
            print(json.dumps(results[-1]), flush=True)
        training = Path('runs/REFORM_R32_qwen_l23_topk_five_seed_16m_v1_20260914')
        tc = json.loads((training / 'config.resolved.json').read_text())
        snapshots = json.loads((training / 'checkpoints.json').read_text())['checkpoints']
        reader = Path('runs/REFORM_R45_bank_ridge_control_fit_v1_20260915/readout_s1_t2.npz')
        cache = Path('runs/REFORM_R32_qwen_counterfactual_source_v1_20260914/states.npz')
        extra = Path('runs/REFORM_R38_qwen_member_fields_five_v1_20260914/source_view_states.npz')
        panel = Path('runs/REFORM_R38_qwen_member_fields_five_v1_20260914/panel.json')
        paths.extend([training / 'config.resolved.json', training / 'checkpoints.json', reader, cache, extra, panel])
        payload = np.load(reader)
        pairs = payload['fit_pairs'][:8]
        hidden = torch.from_numpy(np.concatenate([np.load(cache)['hidden'], np.load(extra)['hidden']]))
        rows = json.loads(panel.read_text())['rows']
        assert all(rows[int(i)]['split'] == 'fit' for i in pairs.ravel())
        saes = {}
        for seed in [1, 2]:
            snapshot = next(s for s in snapshots if s['seed'] == seed and s['objective'] == 'topk' and s['step'] == 16384)
            paths.append(snapshot['path'])
            sd = torch.load(snapshot['path'], map_location='cpu', weights_only=True)
            ae = AutoEncoderTopK(hidden.shape[-1], tc['dict_size'], tc['k'])
            ae.load_state_dict(sd)
            saes[seed] = ae.eval().requires_grad_(False)
        h = hidden[pairs[:, 0], 0]
        donor = hidden[pairs[:, 1], 0]
        source_difference = saes[1].encode(donor) - saes[1].encode(h)
        gate = torch.from_numpy(payload['source_gate'])
        roles = torch.tensor([1 - rows[int(i)]['template'] for i in pairs[:, 0]])
        for index, operation in enumerate(['unit', 'tens']):
            si = torch.from_numpy(payload[f'{operation}_source_indices']).long()
            amplitudes = source_difference[:, si] * gate[roles][:, si, index]
            decoder = saes[1].decoder.weight.T[si]
            coefficients = infer_source_fields(h, saes[2], decoder, amplitudes, 64)
            capacity = saes[2].encode(h)
            violation = float(((-coefficients).clamp_min(0).sum(-1) - capacity).clamp_min(0).max())
            assert violation < 2e-5
            support = (coefficients.abs().sum(-1) > 0).sum(-1)
            assert support.max() <= 64
            expected_field = (source_difference[:, si] * gate[roles][:, si, index]) @ decoder
            field = (decoder.T[None] * amplitudes[:, None]).sum(-1)
            assert torch.allclose(field, expected_field, atol=2e-5, rtol=2e-5)
            results.append(dict(dataset='qwen_donor_replacement', operation=operation, pairs=pairs.tolist(),
                                negative_amplitudes=int((amplitudes < 0).sum()), positive_amplitudes=int((amplitudes > 0).sum()),
                                maximum_capacity_violation=violation, support=support.tolist(),
                                relative_field_error=float(((coefficients.sum(-1) @ saes[2].decoder.weight.T - field).square().sum() / field.square().sum()).sqrt())))
            print(json.dumps(results[-1]), flush=True)
    args.output.write_text(json.dumps(dict(run_id='SOURCE_FIELD_INTERFACE_20260920', started_at_utc=started,
        ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - clock,
        evidence='Implementation checks on retained training states; no new functional confirmation',
        results=results, inputs=[identity(p) for p in paths]), indent=2) + '\n')


if __name__ == '__main__':
    main()
