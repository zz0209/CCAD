from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('D:/CCAD_Storage/runs/cross_seed_models')
DESCRIPTION = ('Held-out reader errors of saved Gemma correspondences on development biographies: effect nRMSE for every '
               'reader and request family, E2 over the readers that never supervised a map, and the paired contrasts of '
               'rules DR1 to DR3, with a bootstrap over biographies within profession-by-gender strata and over '
               'training seeds.')
FAMILIES = ['complete', 'parts', 'new', 'random']
DOC_FAMILIES = ['complete', 'parts', 'random']
EXTRA = ['bio_next_token', 'state', 'mass_pronoun', 'mass_female', 'mass_male']
# Readers on which the paired contrasts given with --contrast are reported, all on part requests.
CONTRAST_READERS = ['original', 'E2', 'bio_next_token', 'state', 'fineweb_next_token']


def groups_of(names):
    out = {}
    for name in names:
        out.setdefault(re.sub(r'_s\d+$', '', name), []).append(name)
    return out


def per_bio(z, req, reqs_of, name, readers):
    # Squared error and source energy per biography, summed over the requests of each family, for every reader.
    err, eff = {}, {}
    for fam, reqs in reqs_of.items():
        rows = [req.index(r) for r in reqs]
        for r in readers:
            clean = z[f'bio|none|{r}'][0]
            src = z[f'bio|source|logit_{r}'][rows] - clean
            mine = z[f'bio|{name}|logit_{r}'][rows] - clean
            err[fam, r] = ((mine - src) ** 2).sum(0)
            eff[fam, r] = (src ** 2).sum(0)
        for key, label in [('nt', 'bio_next_token'), ('state', 'state'), ('mass_pronoun', 'mass_pronoun'),
                           ('mass_female', 'mass_female'), ('mass_male', 'mass_male')]:
            err[fam, label] = z[f'bio|{name}|{key}_err'][rows].sum(0)
            eff[fam, label] = z[f'bio|{name}|{key}_eff'][rows].sum(0)
    return err, eff


def per_doc(z, nreq, reqs_of, name):
    # Next-token squared error and source energy per generic-text document, summed over the requests of each family.
    err, eff = {}, {}
    for fam in DOC_FAMILIES:
        rows = [nreq.index(r) for r in reqs_of[fam]]
        err[fam, 'fineweb_next_token'] = z[f'doc|{name}|nt_err'][rows].sum(0)
        eff[fam, 'fineweb_next_token'] = z[f'doc|{name}|nt_eff'][rows].sum(0)
    return err, eff


def analyse(run, draws, rng, doc_rng):
    z = dict(np.load(run / 'responses.npz'))
    requests = json.loads((run / 'requests.json').read_text())['families']
    record = json.loads((run / 'READERS.json').read_text())
    readers = record['readers']
    held = [r for r in readers if r != 'original'] + ['bio_next_token']
    req = list(z['request_names'])
    nreq = list(z['natural_request_names'])
    reqs_of = {f: requests[f] for f in FAMILIES}
    names = [n for n in record['summary'] if n != 'source']
    groups = groups_of(names)
    strata = z['bio_true'] * 2 + z['bio_gender']
    n = len(strata)
    w = np.zeros((draws, n))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        picks = rng.choice(idx, size=(draws, len(idx)))
        for d in range(draws):
            np.add.at(w[d], picks[d], 1.)
    # Documents are resampled with their own generator, so the biography draws do not depend on this addition.
    n_doc = z[f'doc|{names[0]}|nt_err'].shape[1]
    wd = np.zeros((draws, n_doc))
    picks = doc_rng.integers(0, n_doc, (draws, n_doc))
    for d in range(draws):
        np.add.at(wd[d], picks[d], 1.)
    keys = [(f, r) for f in FAMILIES for r in readers + EXTRA]
    doc_keys = [(f, 'fineweb_next_token') for f in DOC_FAMILIES]
    point, boot = {}, {}
    for g, members in groups.items():
        stats = [per_bio(z, req, reqs_of, m, readers) for m in members]
        doc_stats = [per_doc(z, nreq, reqs_of, m) for m in members]
        seed_pick = rng.integers(0, len(members), (draws, len(members)))
        point[g], boot[g] = {}, {}
        for key in keys + doc_keys:
            source, weight = (doc_stats, wd) if key in doc_keys else (stats, w)
            vals = np.stack([np.sqrt(e[key].sum() / f[key].sum()) for e, f in source])
            bvals = np.stack([np.sqrt((weight * e[key]).sum(1) / (weight * f[key]).sum(1)) for e, f in source])
            point[g][key] = float(vals.mean())
            boot[g][key] = bvals[seed_pick, np.arange(draws)[:, None]].mean(1)
        for f in FAMILIES:
            point[g][f, 'E2'] = float(np.mean([point[g][f, r] for r in held]))
            boot[g][f, 'E2'] = np.mean([boot[g][f, r] for r in held], 0)
        point[g]['seeds'] = len(members)
    return point, boot, readers


def contrast(boot_a, boot_b, pa, pb):
    return dict(point=pa - pb, interval=np.quantile(boot_a - boot_b, [.025, .975]).tolist())


def report(name, point, readers, rec):
    print(name)
    width = max(len(g) for g in point)
    cols = ['original'] + [r for r in readers if r != 'original'] + EXTRA + ['fineweb_next_token', 'E2']
    print(' ' * width, '  '.join(c[:10].rjust(10) for c in cols))
    for g, pt in point.items():
        print(g.ljust(width), '  '.join(f"{pt[('parts', c)]:10.3f}" for c in cols))
    for k, v in rec['contrasts'].items():
        print(f"  {k}: {v['point']:+.3f} [{v['interval'][0]:+.3f}, {v['interval'][1]:+.3f}]")
    if 'DR' in rec:
        print('  DR', {k: v for k, v in rec['DR'].items() if not isinstance(v, dict) or 'point' in v})


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--whitening', default=None, help='prefix of the chosen reader-complete maps, lw_ or std_')
    parser.add_argument('--contrast', nargs=2, action='append', default=[], metavar=('A', 'B'),
                        help='paired contrast A minus B on part requests, for each run and for the pooled runs')
    parser.add_argument('--pool', action='store_true', help='also average every map group over the runs, targets fixed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261011)
    doc_rng = np.random.default_rng(20261012)
    out = dict(written_utc=datetime.now(timezone.utc).isoformat(), description=DESCRIPTION, draws=args.draws, runs={})
    pooled = []
    for name in args.runs:
        point, boot, readers = analyse(RUNS / name, args.draws, rng, doc_rng)
        pooled.append((point, boot))
        rec = dict(point={g: {f'{k[0]}|{k[1]}' if isinstance(k, tuple) else k: v for k, v in p.items()} for g, p in point.items()},
                   intervals={g: {f'{k[0]}|{k[1]}': np.quantile(v, [.025, .975]).tolist() for k, v in b.items()}
                              for g, b in boot.items()}, contrasts={})
        has = lambda *gs: all(g in point for g in gs)
        # DR3: on part requests, the classifier-fitted part map against matching and the natural-text map, read by the
        # model's own next-token distribution on biographies.
        if has('part_sup_hungarian', 'hungarian_decoder', 'natural_sup_hungarian'):
            key = ('parts', 'bio_next_token')
            for other in ['hungarian_decoder', 'natural_sup_hungarian']:
                rec['contrasts'][f'DR3 part_sup-{other} parts bio_next_token'] = contrast(
                    boot['part_sup_hungarian'][key], boot[other][key], point['part_sup_hungarian'][key], point[other][key])
        # DR1 and DR2 on the reader-by-request cells, with the chosen whitening.
        p = args.whitening
        if p and has('e4c_subset_sup_hungarian', 'e8a_random_sup_hungarian', f'{p}subset_state_sup_hungarian',
                     f'{p}random_state_sup_hungarian'):
            rc, lf = f'{p}subset_state_sup_hungarian', f'{p}random_state_sup_hungarian'
            P = ('parts', 'original')
            E2 = ('parts', 'E2')
            rec['DR'] = dict(
                P={g: point[g][P] for g in ['e4c_subset_sup_hungarian', 'e8a_random_sup_hungarian', rc, lf]},
                E2={g: point[g][E2] for g in ['e4c_subset_sup_hungarian', 'e8a_random_sup_hungarian', rc, lf,
                                              'hungarian_decoder', 'natural_sup_hungarian'] if g in point},
                cover_ratio=point[lf][P] / point['e4c_subset_sup_hungarian'][P],
                cover_vs_random=contrast(boot[lf][P], boot['e8a_random_sup_hungarian'][P], point[lf][P],
                                         point['e8a_random_sup_hungarian'][P]),
                ann_ratio_random=point[lf][P] / point['e8a_random_sup_hungarian'][P],
                ann_ratio_rc=point[rc][P] / point['e4c_subset_sup_hungarian'][P],
                e2_ratio_rc=point[rc][E2] / point['e4c_subset_sup_hungarian'][E2])
            for g in [rc, lf]:
                for ref in ['hungarian_decoder', 'natural_sup_hungarian']:
                    if has(ref):
                        rec['contrasts'][f'DR2 {g}-{ref} parts E2'] = contrast(boot[g][E2], boot[ref][E2], point[g][E2], point[ref][E2])
        for a, b in args.contrast:
            for r in CONTRAST_READERS:
                key = ('parts', r)
                rec['contrasts'][f'{a}-{b} parts {r}'] = contrast(boot[a][key], boot[b][key], point[a][key], point[b][key])
        out['runs'][name] = rec
        report(name, point, readers, rec)
    if args.pool:
        # Targets are fixed strata: the pooled value and every bootstrap draw are the means over the runs.
        common = [g for g in pooled[0][0] if all(g in p for p, _ in pooled)]
        point = {g: {k: float(np.mean([p[g][k] for p, _ in pooled])) for k in pooled[0][1][g]} for g in common}
        boot = {g: {k: np.mean([b[g][k] for _, b in pooled], 0) for k in pooled[0][1][g]} for g in common}
        rec = dict(point={g: {f'{k[0]}|{k[1]}': v for k, v in p.items()} for g, p in point.items()},
                   intervals={g: {f'{k[0]}|{k[1]}': np.quantile(v, [.025, .975]).tolist() for k, v in b.items()}
                              for g, b in boot.items()}, contrasts={})
        for a, b in args.contrast:
            for r in CONTRAST_READERS:
                key = ('parts', r)
                c = contrast(boot[a][key], boot[b][key], point[a][key], point[b][key])
                c['runs_negative'] = int(sum(p[a][key] < p[b][key] for p, _ in pooled))
                rec['contrasts'][f'{a}-{b} parts {r}'] = c
        out['pooled'] = dict(runs=args.runs, **rec)
        report('pooled', point, readers, rec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
