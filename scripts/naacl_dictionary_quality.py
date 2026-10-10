from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010')
MIRROR = BULK / 'mayo_mirror'
DESCRIPTION = ('Fraction of variance explained on natural text by each target dictionary before and after the '
               'correspondence training that adapts it, for every setting in which dictionaries are adapted.')


def from_list(path, before, complete, part):
    rows = {r['method']: r for r in json.loads(path.read_text())}
    return [rows[before]['fve'], rows[complete]['fve'], rows[part]['fve']]


def shift(run):
    rows = [json.loads(line) for line in (run / 'metrics.raw.jsonl').open(encoding='utf-8')]
    fve = {}
    for r in rows:
        if r.get('kind') == 'quality':
            fve.setdefault(r['method'], []).append(r['fve'])
    # parts_relation trains the membership only, so its dictionaries are the original targets.
    return [float(np.mean(fve['parts_relation'])), float(np.mean(fve['whole'])), float(np.mean(fve['parts']))]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/naacl_sprint_20261010/e2_supervision/DICTIONARY_QUALITY.json')
    args = parser.parse_args()
    settings = {
        'SHIFT, Pythia-70M (embedding, block-0 MLP and residual)': [
            shift(ROOT / 'runs' / f'IR04_shift_program_seed{s}_v1_20260916') for s in (2, 3, 4, 5)],
        'Infinitive, Pythia-70M, retrained': [
            from_list(BULK / f'NAACL_E2_infinitive_t{t}_v1_20261010' / 'quality.json', 'initial', 'tangent_full',
                      'tangent_mixed') for t in (1, 3, 4, 5)],
        'Infinitive, Pythia-70M, SAEBench': [
            from_list(MIRROR / f'NAACL_E4A_public_t{t}_v1_20261010' / 'quality.json', 'initial', 'tangent_full',
                      'tangent_mixed') for t in range(101, 117)],
        'Grammar, GPT-2 Medium': [
            from_list(Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round08') /
                      f'RG08_GPT2_PROGRAM_CONFIRM_T{t}_20260921' / 'natural_quality.json', 'none', 'whole', 'program')
            for t in (2, 3, 4, 5)],
        'Grammar, Qwen3-1.7B': [
            from_list(MIRROR / f'NAACL_E4B_QWEN17_S{s}_T{t}_v4_20261010' / 'natural_quality.json', 'none', 'whole',
                      'program') for s, t in ((1, 2), (2, 1))],
    }
    out = {}
    for name, rows in settings.items():
        a = np.array(rows)
        out[name] = dict(targets=len(rows), before=float(a[:, 0].mean()), complete=float(a[:, 1].mean()),
                         part=float(a[:, 2].mean()), largest_drop=float((a[:, 0] - a[:, 1:].min(1)).max()))
        print(f"{name:<58} before {a[:, 0].mean():.4f}  complete {a[:, 1].mean():.4f}  part {a[:, 2].mean():.4f}  "
              f"largest drop {out[name]['largest_drop']:.4f}")
    args.output.write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), settings=out), indent=1))


if __name__ == '__main__':
    main()
