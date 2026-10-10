from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

from ccad.artifacts import sha256
from naacl_grammar_matching_maps import exclusive_top, hungarian

DESCRIPTION = ('Label-free matching relations for the four published infinitive features at resid_4, '
               'in the relation-matrix format read by train_infinitive_program.py.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(c['device'])
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    sb = np.load(c['source_parameters'])
    sp = {k: torch.tensor(sb[k], device=device) for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
    h = torch.tensor(np.load(c['natural_states'])['h'], device=device, dtype=torch.float32)
    zs = torch.relu((h - sp['center']) @ sp['encoder'].T + sp['encoder_bias']).double()
    record = dict(config=c, script_sha256=sha256(Path(__file__)), states=int(len(h)), targets={})
    for seed, directory in c['targets']:
        state = torch.load(Path(directory) / f'resid_4_seed{seed}.pt', map_location=device, weights_only=True)
        target = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(device)
        target.load_state_dict(state)
        with torch.no_grad():
            zt = target.encode(h).double()
        zt_c, zs_c = zt - zt.mean(0), zs - zs.mean(0)
        corr = (zt_c.T @ zs_c) / (zt_c.norm(dim=0)[:, None] * zs_c.norm(dim=0)[None, :]).clamp_min(1e-12)
        corr = corr.cpu().numpy()
        corr[~np.isfinite(corr)] = 0.
        dt = target.decoder.weight.T.detach()
        ds = sp['decoder']
        cos = ((dt / dt.norm(dim=1, keepdim=True)) @ (ds / ds.norm(dim=1, keepdim=True)).T).cpu().numpy()
        relations = {}
        for name, picks in [('hungarian_decoder', hungarian(cos)), ('hungarian_activation', hungarian(corr))]:
            rel = np.zeros(cos.shape, dtype=np.float32)
            rel[picks, np.arange(cos.shape[1])] = 1.
            relations[name] = rel
        rel = np.zeros(cos.shape, dtype=np.float32)
        for j, picked in enumerate(exclusive_top(corr, c['members_per_source'])):
            rel[picked, j] = 1.
        relations['activation_top2'] = rel
        np.savez_compressed(out / f'infinitive_matching_seed{seed}.npz', **relations)
        record['targets'][str(seed)] = dict(
            decoder_cosine=cos[relations['hungarian_decoder'].argmax(0), np.arange(4)].tolist(),
            activation_correlation=corr[relations['hungarian_activation'].argmax(0), np.arange(4)].tolist())
        print(json.dumps(dict(seed=seed, **record['targets'][str(seed)])), flush=True)
    record.update(status='PASS', written_utc=datetime.now(timezone.utc).isoformat())
    (out / 'INFINITIVE_MATCHING.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
