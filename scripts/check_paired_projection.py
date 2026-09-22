import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ccad.paired_projection import paired_coefficients
from run_shift_transfer import input_member_delta


def relative_error(value, reference):
    return float(torch.linalg.matrix_norm(value - reference) / torch.linalg.matrix_norm(reference).clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/rg10_grammar_function_holdout_development.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.config.read_text())
    sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK

    source_state = torch.load(config['source_checkpoint'], weights_only=True, map_location='cpu')
    target_state = torch.load(config['target_checkpoint'], weights_only=True, map_location='cpu')
    dim = source_state['encoder.weight'].shape[1]
    source_ae = AutoEncoderTopK(dim, len(source_state['encoder.weight']), int(source_state['k']))
    target = AutoEncoderTopK(dim, len(target_state['encoder.weight']), int(target_state['k']))
    source_ae.load_state_dict(source_state)
    target.load_state_dict(target_state)
    source_ae.eval().requires_grad_(False)
    target.eval()
    gate_path = ROOT / config['source_run'] / f'topk_s{config["source_seed"]}_source.npz'
    with np.load(gate_path) as asset:
        gate = torch.tensor(asset['gate'])
    members = gate.sum(1).nonzero().flatten()
    source = dict(sae=source_ae, member_ids=members, decoder=source_ae.decoder.weight[:, members].T)
    natural_path = ROOT / config['natural_states']
    with np.load(natural_path) as asset:
        available = asset['hidden'][:128]
        state_ids = np.linspace(0, len(available) - 1, 8, dtype=int)
        h = torch.tensor(available[state_ids])
    allowance = config['members_per_source'] * len(members)
    width = min(allowance, int(target.k))
    queries = [torch.ones(len(members)), gate[members] @ torch.tensor([.25, .5, .75]), torch.zeros(len(members))]
    cases = []
    with torch.no_grad():
        z, zs = target.encode(h), source_ae.encode(h)[:, members]
        field = -source['decoder'].T[None] * zs[:, None, :]
        directions = -(target.encoder.weight @ source['decoder'].T)
        score = (zs @ directions.abs().T) * (z > 0) * target.decoder.weight.norm(dim=0)
        original_ids = score.topk(allowance, dim=-1).indices
        selected = (z > 0) * torch.zeros_like(score).scatter(-1, original_ids, 1.)
        ids = selected.topk(width, dim=-1).indices
        retained = selected.gather(1, ids)
        w = target.encoder.weight[ids] * retained[..., None]
        d = target.decoder.weight.T[ids].transpose(-1, -2) * retained[:, None, :]
        cap = z.gather(1, ids) * retained
        assert int((field != 0).sum()) > 0
        # 原 tangent 公式与共同列容量写法使用相同状态、支持和请求。
        old_columns = directions[ids] * zs[:, None, :] * retained[..., None]
        old_negative = (-old_columns).clamp_min(0)
        old_scale = (cap / old_negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
        old_allocated = old_columns.clamp_min(0) - old_negative * old_scale[..., None]
        tangent_errors = []
        for q in queries:
            original, _ = input_member_delta(h, target, source, q, 'input_tangent_budget', allowance,
                                              torch.ones(len(h), dtype=torch.bool))
            reproduced = (d @ (old_allocated @ q)[..., None]).squeeze(-1)
            torch.testing.assert_close(original, reproduced, rtol=3e-5, atol=3e-5)
            tangent_errors.append(float((original - reproduced).abs().max()))
        for method in ('oblique', 'orthogonal'):
            coefficients, diagnostics = paired_coefficients(w, d, field, method)
            padded_w = torch.nn.functional.pad(w, (0, 0, 0, 5))
            padded_d = torch.nn.functional.pad(d, (0, 5))
            padded, padded_diagnostics = paired_coefficients(padded_w, padded_d, field, method)
            torch.testing.assert_close(coefficients, padded[:, :width], rtol=3e-5, atol=3e-5)
            assert int(torch.count_nonzero(padded[:, width:])) == 0
            assert torch.equal(diagnostics['rank'], padded_diagnostics['rank'])
            zero, _ = paired_coefficients(w, d, torch.zeros_like(field), method)
            assert int(torch.count_nonzero(zero)) == 0
            negative = (-coefficients).clamp_min(0)
            scale = (cap / negative.sum(-1).clamp_min(1e-20)).clamp_max(1)
            allocated = coefficients.clamp_min(0) - negative * scale[..., None]
            minimum = float((cap - (-allocated).clamp_min(0).sum(-1)).min())
            assert minimum >= -1e-5
            execution_checks = []
            for q in queries:
                delta = allocated @ q
                assert float((cap + delta).min()) >= -1e-5
                assert int((delta != 0).sum(-1).max()) <= width
                mode = 'input_paired_budget' if method == 'oblique' else 'input_orthogonal_budget'
                actual, counts = input_member_delta(h, target, source, q, mode, allowance,
                                                     torch.ones(len(h), dtype=torch.bool))
                expected_output = (d @ delta[..., None]).squeeze(-1)
                torch.testing.assert_close(actual, expected_output, rtol=3e-5, atol=3e-5)
                assert counts['states'] == len(h) and counts['selected_max'] <= width
                assert counts['minimum_final_code'] >= -1e-5
                execution_checks.append(dict(max_absolute_error=float((actual-expected_output).abs().max()), counts=counts))
            state_checks = []
            for i in range(len(h)):
                wi, di = w[i].double(), d[i].double()
                rtol = float(diagnostics['relative_cutoff'][i])
                b = wi @ di
                local = torch.linalg.pinv(b, atol=0., rtol=rtol) @ wi if method == 'oblique' else torch.linalg.pinv(di, atol=0., rtol=rtol)
                ld = local @ di
                expected = torch.linalg.pinv(b, atol=0., rtol=rtol) @ b if method == 'oblique' else torch.linalg.pinv(di, atol=0., rtol=rtol) @ di
                torch.testing.assert_close(ld, expected, rtol=1e-8, atol=1e-8)
                projection = di @ local
                idempotence = relative_error(projection @ projection, projection)
                assert idempotence < 1e-8
                rank = int(diagnostics['rank'][i])
                if rank == int(retained[i].sum()):
                    torch.testing.assert_close(ld, torch.diag(retained[i].double()), rtol=1e-8, atol=1e-8)
                state_checks.append(dict(state_id=int(state_ids[i]), rank=rank,
                                         selected_members=int(retained[i].sum()),
                                         condition_retained=float(diagnostics['condition_retained'][i]) if rank else None,
                                         ld_identity_error=relative_error(ld, torch.diag(retained[i].double())),
                                         projection_idempotence_error=idempotence))
            result = dict(method=method, minimum_capacity_margin=minimum,
                          zero_padding_max_absolute_difference=float((coefficients - padded[:, :width]).abs().max()),
                          inverse_identity_relative_error_max=float(diagnostics['inverse_identity_relative_error'].max()),
                          states=state_checks, actual_execution=execution_checks)
            cases.append(result)
            print(json.dumps(result), flush=True)
    # 使用同一真实状态检查两个目标参数的可微路径，参数保持原值。
    gradient_checks = []
    for method in ('oblique', 'orthogonal'):
        target.zero_grad(set_to_none=True)
        wi = target.encoder.weight[ids[0]] * retained[0, :, None]
        di = target.decoder.weight[:, ids[0]] * retained[0, None, :]
        coefficients, _ = paired_coefficients(wi, di, field[0], method)
        loss = coefficients.square().mean()
        loss.backward()
        norms = {name: None if p.grad is None else float(p.grad.norm()) for name, p in target.named_parameters()}
        assert norms['decoder.weight'] > 0
        if method == 'oblique':
            assert norms['encoder.weight'] > 0
        else:
            assert norms['encoder.weight'] is None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in target.parameters())
        gradient_checks.append(dict(method=method, parameter_gradient_norms=norms))
    assert all(torch.equal(value, target_state[name]) for name, value in target.state_dict().items())
    assert all(torch.equal(value, source_state[name]) for name, value in source_ae.state_dict().items())
    result = dict(status='PASS', evidence='Real checkpoint and natural-state numerical validation',
                  started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  driver_seconds=time.perf_counter()-timer, device='cpu', threads=torch.get_num_threads(),
                  python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                  source_checkpoint=config['source_checkpoint'], target_checkpoint=config['target_checkpoint'],
                  natural_states=natural_path.as_posix(), source_members=gate_path.as_posix(),
                  state_selection='Eight evenly spaced indices among the first 128 cached natural states',
                  state_ids=state_ids.tolist(), allowance=allowance, maximum_active_budget=width,
                  source_member_count=len(members), tangent_max_absolute_errors=tangent_errors,
                  cases=cases, gradients=gradient_checks, dictionary_parameters_unchanged=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], driver_seconds=result['driver_seconds'])), flush=True)


if __name__ == '__main__':
    main()
