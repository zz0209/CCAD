from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from naacl_grammar_matching_maps import exclusive_top, hungarian

DESCRIPTION = ('Convert public SAEBench Pythia-70M TopK dictionaries at resid_post_layer_4 into the target '
               'format of train_infinitive_program.py, with label-free relations for the infinitive source.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    out = Path(c['output_dir'])
    (out / 'targets').mkdir(parents=True, exist_ok=False)
    sb = np.load(c['source_parameters'])
    sp = {k: torch.tensor(sb[k], dtype=torch.float64) for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
    h = torch.tensor(np.load(c['natural_states'])['h'], dtype=torch.float64)
    zs = torch.relu((h - sp['center']) @ sp['encoder'].T + sp['encoder_bias'])
    zs_c = zs - zs.mean(0)
    mapping = {}
    for index, trainer in enumerate(c['trainers']):
        seed = c['first_id'] + index
        folder = Path(c['saebench_dir']) / trainer
        cfg = json.loads((folder / 'config.json').read_text())['trainer']
        assert cfg['dict_class'] == 'AutoEncoderTopK' and cfg['activation_dim'] == 512
        state = torch.load(folder / 'ae.pt', map_location='cpu', weights_only=True)
        state = dict(state, k=torch.tensor(cfg['k'], dtype=torch.int), threshold=torch.tensor(-1., dtype=torch.float32))
        torch.save(state, out / 'targets' / f'resid_4_seed{seed}.pt')
        enc, bias, dec, bdec = (state[k].double() for k in ['encoder.weight', 'encoder.bias', 'decoder.weight', 'b_dec'])
        pre = torch.relu((h - bdec) @ enc.T + bias)
        top = pre.topk(cfg['k'], dim=-1)
        zt = torch.zeros_like(pre).scatter(-1, top.indices, top.values)
        recon = zt @ dec.T + bdec
        fve = float(1 - (recon - h).square().sum() / (h - h.mean(0)).square().sum())
        zt_c = zt - zt.mean(0)
        corr = ((zt_c.T @ zs_c) / (zt_c.norm(dim=0)[:, None] * zs_c.norm(dim=0)[None, :]).clamp_min(1e-12)).numpy()
        corr[~np.isfinite(corr)] = 0.
        dt = dec.T
        cos = ((dt / dt.norm(dim=1, keepdim=True)) @ (sp['decoder'] / sp['decoder'].norm(dim=1, keepdim=True)).T).numpy()
        geometry = np.zeros(cos.shape, dtype=np.float32)
        for j, picked in enumerate(exclusive_top(cos, c['members_per_source'])):
            geometry[picked, j] = 1.
        relation_dir = out / 'relations' / f'seed{seed}'
        relation_dir.mkdir(parents=True)
        np.savez_compressed(relation_dir / 'relation.npz', native=geometry, geometry=geometry, geometry_gain=geometry,
                            candidates=np.flatnonzero(geometry.sum(1)).astype(np.int64))
        matching = {}
        for name, picks in [('hungarian_decoder', hungarian(cos)), ('hungarian_activation', hungarian(corr))]:
            rel = np.zeros(cos.shape, dtype=np.float32)
            rel[picks, np.arange(cos.shape[1])] = 1.
            matching[name] = rel
        rel = np.zeros(cos.shape, dtype=np.float32)
        for j, picked in enumerate(exclusive_top(corr, c['members_per_source'])):
            rel[picked, j] = 1.
        matching['activation_top2'] = rel
        np.savez_compressed(out / f'infinitive_matching_seed{seed}.npz', **matching)
        # Fraction of each source feature's natural activation retained when read from the target reconstruction.
        zs_rec = torch.relu((recon - sp['center']) @ sp['encoder'].T + sp['encoder_bias'])
        retained = (zs_rec * zs).sum(0) / zs.square().sum(0).clamp_min(1e-12)
        mapping[str(seed)] = dict(trainer=trainer, dict_size=cfg['dict_size'], k=cfg['k'], lr=cfg.get('lr'),
                                  fve_on_fit_states=fve, source_activation_retained=retained.tolist(),
                                  hungarian_decoder_cosine=cos[matching['hungarian_decoder'].argmax(0), np.arange(4)].tolist())
        print(json.dumps({seed: mapping[str(seed)]}), flush=True)
    (out / 'TARGETS.json').write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(),
                                                      config=c, targets=mapping), indent=1))


if __name__ == '__main__':
    main()
