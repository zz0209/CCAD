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

from ccad.program_compilation import compile_program, project_program_rows
from run_shift_transfer import input_member_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/rg08_grammar_program_development.json')
    parser.add_argument('--checkpoint', type=Path, default=Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round08/RG08_GPT2_PROGRAM_DEV_T2_20260921/program_step512.pt'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.config.read_text())
    sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK

    source_state = torch.load(config['source_checkpoint'], map_location='cpu', weights_only=True)
    target_state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['dictionary']
    dim = source_state['encoder.weight'].shape[1]
    source = AutoEncoderTopK(dim, len(source_state['encoder.weight']), int(source_state['k']))
    target = AutoEncoderTopK(dim, len(target_state['encoder.weight']), int(target_state['k']))
    source.load_state_dict(source_state)
    target.load_state_dict(target_state)
    source.eval().requires_grad_(False)
    target.eval().requires_grad_(False)
    source_seed = config.get('source_seed', config['seeds'][0])
    source_path = ROOT / config['source_run'] / f'topk_s{source_seed}_source.npz'
    with np.load(source_path) as asset:
        gates = torch.tensor(asset['gate'])
    members = gates.sum(1).nonzero().flatten()
    source_parameters = dict(sae=source, member_ids=members, decoder=source.decoder.weight[:, members].T)
    natural_path = ROOT / config['natural_states']
    with np.load(natural_path) as asset:
        state_indices = np.linspace(0, min(len(asset['hidden']), 128) - 1, 8, dtype=int)
        hidden = torch.tensor(asset['hidden'][state_indices])
    with torch.no_grad():
        full_codes = target.encode(hidden)
        target_members = (full_codes != 0).any(0).nonzero().flatten()
        codes = full_codes[:, target_members]
        decoder = target.decoder.weight[:, target_members]
        attention = torch.ones(len(hidden), dtype=torch.bool)
        allowance = config['members_per_source'] * len(members)
        fields = torch.stack([input_member_delta(hidden, target, source_parameters, gates[members, part],
                                'input_tangent_budget', allowance, attention)[0] for part in range(gates.shape[1])], dim=1)
    assert float(fields.square().sum()) > 0
    coefficients, diagnostics = compile_program(codes, decoder, fields, steps=512)
    zd, dd, yd, bd = codes.double(), decoder.double(), fields.double(), coefficients.double()
    prediction = torch.einsum('nf,fg,df->ngd', zd, bd, dd)
    direct = (prediction - yd).square().sum() / len(codes)
    gram = (zd.T @ zd) * (dd.T @ dd) / len(codes)
    rhs = torch.einsum('nf,ngf->fg', zd, torch.einsum('ngd,df->ngf', yd, dd)) / len(codes)
    quadratic = yd.square().sum() / len(codes) + (bd * (gram @ bd - 2 * rhs)).sum()
    torch.testing.assert_close(direct, quadratic, rtol=1e-10, atol=1e-9)
    assert diagnostics['objective_final'] < diagnostics['objective_initial']
    proposal = rhs / diagnostics['largest_gram_eigenvalue']
    projected = project_program_rows(proposal)
    torch.testing.assert_close(projected.clamp_min(0), proposal.clamp_min(0), rtol=0, atol=0)
    assert bool((proposal > 0).any()) and bool((proposal < 0).any())
    assert float((-projected).clamp_min(0).sum(-1).max()) <= 1 + 1e-12
    assert float((-bd).clamp_min(0).sum(-1).max()) <= 1 + 2e-7
    minimum = 0.
    maximum_linearity_error = 0.
    queries = list(torch.eye(gates.shape[1], dtype=torch.float64))
    queries.extend([torch.ones(gates.shape[1], dtype=torch.float64), torch.tensor([.25, .5, .75], dtype=torch.float64)])
    for query in queries:
        delta_code = zd * (bd @ query)
        direct_request = delta_code @ dd.T
        combined = (prediction * query[None, :, None]).sum(1)
        torch.testing.assert_close(direct_request, combined, rtol=1e-10, atol=1e-9)
        maximum_linearity_error = max(maximum_linearity_error, float((direct_request - combined).abs().max()))
        minimum = min(minimum, float((zd + delta_code).min()))
    assert minimum >= -1e-5
    assert all(torch.equal(target.state_dict()[key], value) for key, value in target_state.items())
    assert all(torch.equal(source.state_dict()[key], value) for key, value in source_state.items())
    result = dict(status='PASS', started_at_utc=started, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  seconds=time.perf_counter()-timer, config=str(args.config.resolve()), checkpoint=str(args.checkpoint.resolve()),
                  source_checkpoint=config['source_checkpoint'], natural_states=str(natural_path.resolve()),
                  state_indices=state_indices.tolist(), target_members=target_members.tolist(),
                  diagnostics=diagnostics, direct_objective=float(direct), gram_objective=float(quadratic),
                  positive_projection_unchanged=True, minimum_final_code=minimum,
                  maximum_linearity_error=maximum_linearity_error, dictionaries_unchanged=True,
                  scope='CPU numerical check on eight existing natural states and real trained-program fields; no language-model response evaluation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
