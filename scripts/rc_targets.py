from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'containment'
READBACK = Path('D:/CCAD_Storage/runs/naacl_sprint_20261010/mayo_mirror/NAACL_E10_GEMMA2B_readback_v1_20261010/READBACK.json')
DESCRIPTION = ('Held-out readers of the Gemma SHIFT correspondences across all seven Gemma Scope targets, on part requests: '
               'the original classifier, the mean of the held-out classifiers and the gender probe, and the mean of the '
               'token-level readers, with the sparsity of each target relative to the source and the decoder cosine of '
               'its Hungarian matches.')
LATER = ['composer_surgeon_o0', 'composer_surgeon_o1', 'model_software_engineer_o0', 'model_software_engineer_o1', 'gender']
MASSES = ['mass_pronoun', 'mass_female', 'mass_male']
# Method groups per analysis kind: development targets use the EXP-2 and C1 maps, confirmation targets the E10 and
# CONF maps; both train the same objectives with the same recipe.
METHODS = {'dev': dict(state='lw_subset_state_sup_hungarian', classifier='e4c_subset_sup_hungarian',
                       classifier_parts='part_sup_hungarian', hungarian='hungarian_decoder', generic='natural_sup_hungarian'),
           'conf': dict(state='conf_subset_state_sup_hungarian', classifier='e10_subset_sup_hungarian',
                        classifier_parts='e10_part_sup_hungarian', hungarian='e10_hungarian_decoder',
                        generic='conf_natural_sup_hungarian')}
TARGET_OF = {'w65k': 'w65k_sfc', 'l0low16k': 'w16k_lowest'}


def l0_ratio(target):
    # Geometric mean over edited sites of the target's average L0 relative to the source's, read from the released
    # dictionary paths, which name the average L0 of each dictionary.
    sites = json.loads((ART / 'gemma_target_dictionaries' / f'{target}.json').read_text())
    l0 = lambda p: int(re.search(r'average_l0_(\d+)', p).group(1))
    return float(np.exp(np.mean([np.log(l0(v['target']) / l0(v['source'])) for v in sites.values()])))


def readers(pt, g):
    get = lambda r: pt[g][f'parts|{r}']
    token = [get('bio_next_token'), np.mean([get(r) for r in MASSES]), get('state'), get('fineweb_next_token')]
    return dict(original=get('original'), held_out=float(np.mean([get(r) for r in LATER])), token=float(np.mean(token)),
                e2=get('E2'))


def collect(sources):
    cos = {k: v['hungarian_cos_mean'] for k, v in json.loads(READBACK.read_text()).items()}
    rows = []
    for kind, path in sources:
        rec = json.loads(Path(path).read_text())
        for run, r in rec['runs'].items():
            key = run.split('_readers_')[1].rsplit('_v', 1)[0]
            target = TARGET_OF.get(key, key)
            rows.append(dict(target=target, kind=kind, run=run, hungarian_cos=cos[target], l0_ratio=l0_ratio(target),
                             methods={m: readers(r['point'], g) for m, g in METHODS[kind].items()}))
    return sorted(rows, key=lambda r: r['l0_ratio'])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dev', type=Path, required=True, help='reader analysis of the development targets')
    parser.add_argument('--conf', type=Path, required=True, help='reader analysis of the confirmation targets')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = collect([('dev', args.dev), ('conf', args.conf)])
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION,
               sources=dict(dev=str(args.dev), conf=str(args.conf)), rows=rows)
    args.output.write_text(json.dumps(out, indent=1))
    for r in rows:
        m = r['methods']
        print(f"{r['target']:<13} {r['kind']:<4} L0 ratio {r['l0_ratio']:.2f} cos {r['hungarian_cos']:.3f} | " +
              ' | '.join(f"{k} {m[k]['original']:.3f}/{m[k]['held_out']:.3f}/{m[k]['token']:.3f}" for k in m))


if __name__ == '__main__':
    main()
