from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

from ccad.intervention_transport import transport_delta, refine_columns, pursuit_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    c = json.loads(args.config.read_text())
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    natural = torch.load(c['natural_cache'], weights_only=True)
    roots = torch.load(c['source_metric_cache'], weights_only=True)
    human = np.load(c['source_parameters'])
    grammar = np.load(c['grammar_source_parameters'])
    output = []
    for dataset, site in [('human', 'embed'), ('grammar', 'resid_4')]:
        bank = human if dataset == 'human' else grammar
        source = {k: torch.tensor(bank[site+'__'+k if dataset == 'human' else k], device='cuda')
                  for k in ['center', 'encoder', 'encoder_bias', 'decoder']}
        h = natural[site].to('cuda')
        zs = torch.relu((h-source['center']) @ source['encoder'].T + source['encoder_bias'])
        rows = (zs.sum(-1) > 0).nonzero().flatten()[:32]
        h = h[rows]
        zs = zs[rows]
        y = -source['decoder'].T[None] * zs[:, None, :]
        initial = torch.load(Path(c['target_directory']) / f'{site}_seed2.pt', weights_only=True)
        path = c['task_adapted_reference'][dataset][site] if dataset == 'human' else c['task_adapted_reference']['grammar']
        trained = torch.load(path, weights_only=True)
        target = AutoEncoderTopK(512, len(initial['encoder.weight']), int(initial['k'])).to('cuda')
        target.requires_grad_(False)
        q = torch.ones(zs.shape[-1], device='cuda')
        for component in ['initial', 'encoder', 'decoder', 'both']:
            sd = {k: (trained[k] if component == 'both' or component == 'encoder' and k.startswith('encoder.')
                      or component == 'decoder' and k.startswith('decoder.') else v) for k, v in initial.items()}
            target.load_state_dict(sd)
            _, _, columns = transport_delta(h, target, source, q, target.encoder.weight, 2*len(q), False)
            z = target.encode(h)
            for metric in ['identity', 'profile']:
                root = torch.eye(512, device='cuda') if metric == 'identity' else roots[dataset][site].to('cuda')
                for method in ['encoder_support', 'pursuit_support']:
                    with torch.no_grad():
                        a = refine_columns(h, target, source, columns, 2*len(q), 128, metric_root=root, accelerate=True) if method == 'encoder_support' else pursuit_columns(h, target, source, 2*len(q), root, 128)
                        error = target.decoder.weight @ a - y
                        weighted = root.T @ error
                        energy = (root.T @ y).square().sum()
                        assert ((-a).clamp_min(0).sum(-1) <= z+2e-5).all()
                        output.append(dict(dataset=dataset, component=component, metric=metric, method=method,
                            relative_weighted_mse=float(weighted.square().sum()/energy),
                            relative_euclidean_mse=float(error.square().sum()/y.square().sum()),
                            changed_members=int((a!=0).any(-1).sum(-1).max()),
                            states=rows.cpu().tolist()))
                        print(json.dumps({k:v for k,v in output[-1].items() if k!='states'}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
