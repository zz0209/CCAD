import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_shift_transfer import input_member_delta


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/rg10_grammar_function_holdout_development.json')
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
    gate_path = ROOT/config['source_run']/f'topk_s{config["source_seed"]}_source.npz'
    with np.load(gate_path) as asset:
        gate = torch.tensor(asset['gate'])
    members = gate.sum(1).nonzero().flatten()
    assert len(members) == 192 and bool((gate.sum(1) <= 1).all())
    source = dict(sae=source_ae, member_ids=members, decoder=source_ae.decoder.weight[:, members].T)
    natural_path = ROOT/config['natural_states']
    with np.load(natural_path) as asset:
        natural = torch.tensor(asset['hidden'][:128])
    with torch.no_grad():
        effect = source_ae.encode(natural)[:, members]@source['decoder']
        state_ids = effect.square().sum(1).topk(8).indices
    h = natural[state_ids].reshape(2, 4, dim)
    attention = torch.ones(2, 4, dtype=torch.bool)
    attention[-1, -1] = False
    q = torch.ones(len(members))
    rows = []
    for allowance in [16, 192]:
        calls = dict(source=0, target=0)

        def source_call(module, inputs, output):
            calls['source'] += 1

        def target_call(module, inputs, output):
            calls['target'] += 1

        handles = [source_ae.encoder.register_forward_hook(source_call), target.encoder.register_forward_hook(target_call)]
        with torch.no_grad():
            actual, counts = input_member_delta(h, target, source, q, 'input_request_budget', allowance, attention)
        for handle in handles:
            handle.remove()
        assert counts['source_encode_calls'] == calls['source'] == 1
        assert counts['target_encode_calls'] == calls['target'] == 2
        assert counts['states'] == 7 and counts['max_changed_per_state'] <= min(allowance, int(target.k))
        assert torch.equal(actual[~attention], torch.zeros_like(actual[~attention]))
        with torch.no_grad():
            z = target.encode(h)
            v = -(source_ae.encode(h)[..., members]*q)@source['decoder']
            dz = target.encode(h+v)-z
            ids = (dz.abs()*target.decoder.weight.norm(dim=0)).topk(min(allowance, int(target.k)), dim=-1).indices
            dz = dz*torch.zeros_like(dz).scatter(-1, ids, 1.)
            final_code = z+dz
            assert float(final_code.min()) >= 0
            expected = (dz@target.decoder.weight.T)*attention.unsqueeze(-1)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            zero, zero_counts = input_member_delta(h, target, source, torch.zeros_like(q), 'input_request_budget', allowance, attention)
            assert torch.count_nonzero(zero) == zero_counts['changed'] == 0
            # 同一个源作用用重复成员和一半系数表示，检查请求描述的数值一致性。
            doubled = dict(sae=source_ae, member_ids=members.repeat(2), decoder=source['decoder'].repeat(2, 1))
            repeated, _ = input_member_delta(h, target, doubled, q.repeat(2)/2, 'input_request_budget', allowance, attention)
            torch.testing.assert_close(actual, repeated, rtol=2e-5, atol=2e-5)
        rows.append(dict(allowance=allowance, counts=counts, observed_encoder_calls=calls,
                         newly_activated=int(((z == 0) & (dz > 0) & attention.unsqueeze(-1)).sum()),
                         minimum_final_code=float(final_code.min()),
                         catalog_max_absolute_difference=float((actual-repeated).abs().max())))
        print(json.dumps(rows[-1]), flush=True)
    output, _ = input_member_delta(h, target, source, q, 'input_request_budget', 192, attention)
    loss = output.square().mean()
    loss.backward()
    gradient_norms = {name: float(parameter.grad.norm()) for name, parameter in target.named_parameters()}
    assert gradient_norms['encoder.weight'] > 0 and gradient_norms['decoder.weight'] > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in target.parameters())
    assert all(torch.equal(value, target_state[name]) for name, value in target.state_dict().items())
    assert all(torch.equal(value, source_state[name]) for name, value in source_ae.state_dict().items())
    inputs = [args.config, Path(config['source_checkpoint']), Path(config['target_checkpoint']), gate_path, natural_path]
    result = dict(status='PASS', evidence='Real-state numerical smoke; no language-model functional evaluation',
                  started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.perf_counter()-timer, python=sys.executable, torch=torch.__version__,
                  numpy=np.__version__, threads=torch.get_num_threads(), selected_natural_state_ids=state_ids.tolist(),
                  selection='Eight largest source action norms among the first128 saved natural states',
                  cases=rows, gradient_norms=gradient_norms, backward_loss=float(loss.detach()),
                  dictionary_parameters_unchanged=True,
                  inputs=[dict(path=p.as_posix(), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], gradient_norms=gradient_norms)), flush=True)


if __name__ == '__main__':
    main()
