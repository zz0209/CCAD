from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = ('Fraction of each explanation feature\'s natural activation that survives when the source encoder reads '
               'the target reconstruction, for the retrained infinitive and GPT-2 targets, matching the public-target '
               'measurement of naacl_prepare_public_targets.py.')


def topk_recon(state, h):
    enc, bias, dec, bdec = (state[k].double() for k in ['encoder.weight', 'encoder.bias', 'decoder.weight', 'b_dec'])
    pre = torch.relu((h - bdec) @ enc.T + bias)
    top = pre.topk(int(state['k']), dim=-1)
    z = torch.zeros_like(pre).scatter(-1, top.indices, top.values)
    return z, z @ dec.T + bdec


def retention(zs, zs_rec):
    return ((zs_rec * zs).sum(0) / zs.square().sum(0).clamp_min(1e-12)).numpy()


def infinitive():
    sp = np.load(ROOT / 'artifacts/morning_reform_20260916/independent_circuit_reading/source_parts.npz')
    sp = {k: torch.tensor(sp[k], dtype=torch.float64) for k in ['encoder', 'encoder_bias', 'decoder', 'center']}
    h = torch.tensor(np.load(ROOT / 'runs/IR01_infinitive_natural_dev_v1_20260916/fit_states.npz')['h'], dtype=torch.float64)

    def encode(x):
        return torch.relu((x - sp['center']) @ sp['encoder'].T + sp['encoder_bias'])
    zs = encode(h)
    out = {}
    for seed, folder in [(1, 'REFORM_R58_shift_dictionaries_curve_v1_20260915'), (3, 'REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915'),
                         (4, 'REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915'),
                         (5, 'REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915')]:
        state = torch.load(Path('D:/CCAD_Storage/training_curves') / folder / 'step_8192' / f'resid_4_seed{seed}.pt',
                           map_location='cpu', weights_only=True)
        _, recon = topk_recon(state, h)
        out[str(seed)] = retention(zs, encode(recon)).tolist()
    return out


def gpt2():
    root = Path('D:/CCAD_Storage/training_curves/FINAL3_R19_gpt2_dual_objective_five_seed_16m_v1_20260909/step_16384')
    gate = np.load(ROOT / 'runs/REFORM_R26_contrast_components_v2_20260913/topk_s1_source.npz')['gate']
    members = np.flatnonzero(gate.sum(1))
    assert len(members) == 192
    h = torch.tensor(np.load(ROOT / 'runs/FINAL3_R19_gpt2_selector_five_cycle_v2_20260909/natural_discovery_states.npz')['hidden'],
                     dtype=torch.float64)
    src = torch.load(root / 'topk_seed1.pt', map_location='cpu', weights_only=True)
    zs_full, _ = topk_recon(src, h)
    zs = zs_full[:, members]
    out = {}
    for seed in (2, 3, 4, 5):
        state = torch.load(root / f'topk_seed{seed}.pt', map_location='cpu', weights_only=True)
        _, recon = topk_recon(state, h)
        zs_rec, _ = topk_recon(src, recon)
        r = retention(zs, zs_rec[:, members])
        active = zs.square().sum(0).numpy() > 0
        out[str(seed)] = r[active].tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/naacl_sprint_20261010/e4_scale/RETENTION_RETRAINED.json')
    args = parser.parse_args()
    res = dict(infinitive=infinitive(), gpt2=gpt2())
    summary = {k: dict(per_target_mean={s: float(np.mean(v)) for s, v in d.items()},
                       mean=float(np.mean([np.mean(v) for v in d.values()]))) for k, d in res.items()}
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), summary=summary,
                                           per_feature=res), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
