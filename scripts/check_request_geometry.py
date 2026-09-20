from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch

from ccad.intervention_transport import transport_delta, refine_columns
from ccad.request_inference import request_second_moment, solve_request_columns
from run_shift_explanation import source_groups


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


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
    manifest = json.loads(Path(c['source_manifest']).read_text())
    grouping, _ = source_groups(Path(c['notebook']), manifest['members'])
    rows = []
    states = {}
    paths = [args.config, c['natural_cache'], c['source_metric_cache'], c['source_parameters'],
             c['source_manifest'], c['notebook'], __file__, 'src/ccad/request_inference.py',
             'src/ccad/intervention_transport.py']
    with torch.no_grad():
        for site in ['embed', 'mlp_0', 'resid_0']:
            source = {k: torch.from_numpy(bank[site + '__' + k])
                      for k in ['center', 'encoder', 'encoder_bias', 'decoder']}
            all_h = natural[site]
            all_zs = torch.relu((all_h - source['center']) @ source['encoder'].T + source['encoder_bias'])
            selected = (all_zs.sum(-1) > 0).nonzero().flatten()[:32]
            states[site] = selected.tolist()
            h = all_h[selected]
            zs = all_zs[selected]
            path = Path(c['target_directory']) / f'{site}_seed{c["target_seed"]}.pt'
            paths.append(path)
            sd = torch.load(path, map_location='cpu', weights_only=True)
            target = AutoEncoderTopK(512, len(sd['encoder.weight']), int(sd['k']))
            target.load_state_dict(sd)
            target.requires_grad_(False)
            p = zs.shape[-1]
            indicators = torch.tensor([[float(m in grouping[g].get(site, [])) for g in grouping]
                                       for m in manifest['members'][site]])
            qs = dict(columns=torch.eye(p), semantic=request_second_moment(indicators, 0),
                      mixed=request_second_moment(indicators, 0.1))
            evaluation = dict(semantic=qs['semantic'], members=request_second_moment(indicators, 1),
                              columns=torch.eye(p))
            _, _, columns = transport_delta(h, target, source, torch.ones(p), target.encoder.weight, 2 * p, False)
            candidates = -(target.encoder.weight @ source['decoder'].T)[None] * zs[:, None, :]
            scores = candidates.abs().sum(-1) * target.decoder.weight.norm(dim=0)
            ids = scores.topk(2 * p, dim=-1).indices
            initial = columns.gather(1, ids[..., None].expand(-1, -1, p))
            d = target.decoder.weight.T[ids]
            y = -source['decoder'].T[None] * zs[:, None, :]
            capacity = target.encode(h).gather(1, ids)
            root = roots['human'][site]
            weighted_d, weighted_y = d @ root, root.T @ y
            gram = weighted_d @ weighted_d.transpose(-1, -2)
            rhs = weighted_d @ weighted_y
            for steps in [128, 512]:
                for name, moment in qs.items():
                    a = solve_request_columns(gram, rhs, capacity, initial, moment, steps)
                    consumption = (-a).clamp_min(0).sum(-1)
                    assert (consumption <= capacity + 2e-5).all()
                    error = d.transpose(-1, -2) @ a - y
                    weighted_error = root.T @ error
                    row = dict(site=site, method=name, steps=steps, states=len(h),
                               positive_capacity_binding=int(((capacity > 1e-7) & (consumption >= capacity - 1e-6)).sum()),
                               positive_capacity_rows=int((capacity > 1e-7).sum()),
                               coefficient_l1=float(a.abs().sum()))
                    for label, q in evaluation.items():
                        numerator = ((weighted_error @ q) * weighted_error).sum()
                        denominator = ((weighted_y @ q) * weighted_y).sum()
                        row[label + '_relative_mse'] = float(numerator / denominator)
                    if name == 'columns':
                        original = refine_columns(h, target, source, columns, 2 * p, steps,
                                                  metric_root=root, accelerate=True)
                        expected = original.gather(1, ids[..., None].expand(-1, -1, p))
                        row['identity_max_difference'] = float((a - expected).abs().max())
                        print(json.dumps(dict(site=site, steps=steps, identity_max_difference=row['identity_max_difference'])), flush=True)
                        assert torch.allclose(a, expected, atol=2e-5, rtol=2e-5)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    result = dict(run_id='REQUEST_GEOMETRY_20260920', started_at_utc=started,
                  ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - clock,
                  evidence='Development local quadratic objectives on retained natural training states',
                  state_indices=states, torch=torch.__version__, python=sys.executable,
                  inputs=[identity(p) for p in paths], results=rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
