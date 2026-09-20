from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / 'src'), 'D:/CCAD_Storage/environments/r005a_dictionary_overlay',
                'D:/CCAD_Storage/references/source/dictionary_learning_60ec6bf'])
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from ccad.intervention_transport import transport_delta, refine_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    config = json.loads((ROOT / 'configs/profile_confirmation_t2_20260920.json').read_text())
    roots = torch.load(config['source_metric_cache'], map_location='cpu', weights_only=True)
    natural = torch.load(config['natural_cache'], map_location='cpu', weights_only=True)
    sd = torch.load(Path(config['target_directory']) / 'resid_4_seed2.pt', map_location='cpu', weights_only=True)
    target = AutoEncoderTopK(512, sd['encoder.weight'].shape[0], int(sd['k']))
    target.load_state_dict(sd)
    target.requires_grad_(False)
    bank = np.load(ROOT / config['grammar_source_parameters'])
    source = {k: torch.tensor(bank[k]) for k in ['center', 'encoder', 'encoder_bias', 'decoder']}
    h = natural['resid_4'][-8:]
    q = torch.ones(4)
    _, _, columns = transport_delta(h, target, source, q, target.encoder.weight, 8, False)
    results = {}
    with torch.no_grad():
        zero = refine_columns(h, target, source, columns, 8, 0, accelerate=True)
        identity = refine_columns(h, target, source, columns, 8, 128, accelerate=True)
        explicit = refine_columns(h, target, source, columns, 8, 128, metric_root=torch.eye(512), accelerate=True)
        assert torch.allclose(identity, explicit, atol=2e-6, rtol=2e-5)
        support = ((-(target.encoder.weight @ source['decoder'].T))[None]
                   * torch.relu((h-source['center']) @ source['encoder'].T+source['encoder_bias'])[:, None])
        ids = (support.abs().sum(-1)*target.decoder.weight.norm(dim=0)).topk(8, dim=-1).indices
        selected = torch.zeros((len(h), len(sd['encoder.weight'])), dtype=torch.bool).scatter(1, ids, True)
        y = -source['decoder'].T[None] * torch.relu((h-source['center']) @ source['encoder'].T+source['encoder_bias'])[:, None]
        z = target.encode(h)
        for name, metric_root in [('identity', torch.eye(512)), ('source_profile', roots['grammar']['resid_4'])]:
            fitted = refine_columns(h, target, source, columns, 8, 128, metric_root=metric_root, accelerate=True)
            e0 = target.decoder.weight @ zero - y
            e1 = target.decoder.weight @ fitted - y
            before = float((metric_root.T @ e0).square().sum())
            after = float((metric_root.T @ e1).square().sum())
            assert after <= before + 1e-4
            assert ((-fitted).clamp_min(0).sum(-1) <= z + 2e-5).all()
            assert torch.count_nonzero(fitted[~selected]) == 0
            for mask in range(16):
                request = torch.tensor([(mask >> i) & 1 for i in range(4)], dtype=h.dtype)
                assert (z + fitted @ request).min() >= -2e-5
            results[name] = dict(initial_objective=before, fitted_objective=after,
                                 max_changed_members=int((fitted != 0).any(-1).sum(-1).max()))
    spectra = {}
    for dataset, site_roots in roots.items():
        spectra[dataset] = {}
        for site, factor in site_roots.items():
            metric = factor.double() @ factor.double().T
            values = ((torch.linalg.eigvalsh(metric) - .05) / .95).clamp_min(0).flip(0)
            shares = values / values.sum()
            spectra[dataset][site] = dict(top1=float(shares[0]), top5=float(shares[:5].sum()),
                rank95=int((shares.cumsum(0) < .95).sum()+1),
                effective_rank=float(1/shares.square().sum()), trace=float(torch.trace(metric)))
    result = dict(checks=results, identity_root_equivalence=True, all_16_grammar_masks_feasible=True,
                  scope='Eight actual held-out natural states, actual target seed2 and public grammatical members; CPU numerical check. Profile spectra descriptive.',
                  source_profile_spectra=spectra)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
