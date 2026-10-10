from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010')
DESCRIPTION = ('How far correspondence training moves the infinitive target dictionaries, measured by the cosine between '
               'each feature\'s original and adapted decoder and encoder directions.')
TARGETS = {1: 'REFORM_R58_shift_dictionaries_curve_v1_20260915', 3: 'REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915',
           4: 'REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915', 5: 'REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915'}


def cosines(a, b):
    a, b = a.double(), b.double()
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/naacl_sprint_20261010/e2_supervision/DICTIONARY_DRIFT.json')
    args = parser.parse_args()
    out = {}
    for seed, folder in TARGETS.items():
        original = torch.load(Path('D:/CCAD_Storage/training_curves') / folder / 'step_8192' / f'resid_4_seed{seed}.pt',
                              map_location='cpu', weights_only=True)
        row = {}
        for variant in ['tangent_full', 'tangent_mixed']:
            adapted = torch.load(BULK / f'NAACL_E2_infinitive_t{seed}_v1_20261010' / variant / 'dictionary.pt',
                                 map_location='cpu', weights_only=True)
            dec = cosines(original['decoder.weight'], adapted['decoder.weight']).numpy()
            enc = cosines(original['encoder.weight'].T, adapted['encoder.weight'].T).numpy()
            row[variant] = dict(decoder_mean=float(dec.mean()), decoder_min=float(dec.min()),
                                decoder_below_99=float((dec < .99).mean()), encoder_mean=float(enc.mean()),
                                encoder_min=float(enc.min()), encoder_below_99=float((enc < .99).mean()))
        out[str(seed)] = row
        print(seed, json.dumps(row))
    summary = {v: {k: float(np.mean([out[s][v][k] for s in out])) for k in out['1'][v]} for v in ['tangent_full', 'tangent_mixed']}
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), summary=summary, targets=out),
                                      indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
