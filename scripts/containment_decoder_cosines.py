from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SAES = Path('D:/CCAD_Storage/training_curves/FINAL3_R19_gpt2_dual_objective_five_seed_16m_v1_20260909/step_16384')
GATES = ROOT / 'runs/REFORM_R26_contrast_components_v2_20260913'
DESCRIPTION = ('Mean decoder cosine of the Hungarian decoder matches of the explained features, for every GPT-2 Medium '
               'source and target SAE pair of X1 and X1c, the atom-stability coordinate of the geometry law.')
PAIRS = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5), (5, 1)]


def decoder(seed):
    state = torch.load(SAES / f'topk_seed{seed}.pt', map_location='cpu', weights_only=True)
    d = state['decoder.weight'].double().numpy().T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/containment/GPT2_DECODER_COSINES.json')
    args = parser.parse_args()
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, pairs={})
    for s, t in PAIRS:
        gate = np.load(GATES / f'topk_s{s}_source.npz')['gate']
        members = np.flatnonzero(gate.sum(1))
        cos = decoder(s)[members] @ decoder(t).T
        rows, cols = linear_sum_assignment(-cos)
        matched = cos[rows, cols]
        out['pairs'][f's{s}_t{t}'] = dict(members=len(members), mean_cosine=float(matched.mean()),
                                          per_part=[float(matched[gate[members].argmax(1)[rows] == k].mean()) for k in range(3)])
        print(s, t, round(float(matched.mean()), 4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
