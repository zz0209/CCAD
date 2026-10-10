from pathlib import Path
import argparse
import copy
import json
import random
import shutil
import sys
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_shift_transfer import (NURSE, PROFESSOR, Dictionaries, Model, Run, natural_statistics, read_split)

DESCRIPTION = ('How far each Gemma Scope target dictionary moves from the source at the sites of the Gemma SHIFT '
               'explanation: how well the source SAE reads its explained features back from the target reconstruction, '
               'how much of their activation it retains, how close their Hungarian decoder matches are, and how much '
               'natural variance the target explains. Natural text only, no task data.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    run = Run(cfg, args.config, args.resume)
    shutil.copy2(__file__, run.dir / 'code' / Path(__file__).name)
    try:
        execute(cfg, run)
    except BaseException as exc:
        run.status('FAIL', error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
        raise
    run.status('PASS')


def execute(cfg, run):
    dev = torch.device('cuda:0')
    manifest = json.loads(Path(cfg['features']).read_text())
    features = [dict(site=site, index=int(idx), part=part, comment=comment)
                for site, rows in manifest['features'].items() for idx, part, comment in rows]
    text, prof, _ = read_split(cfg, 'train')
    # The same natural biographies as the transfer runs: professions other than professor and nurse.
    natural = [x for x, p in zip(text, prof) if p not in (PROFESSOR, NURSE)]
    random.Random(cfg['data_seed'] + 1).shuffle(natural)
    natural = natural[:cfg['natural_bios']]
    out_path = run.dir / 'READBACK.json'
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    model = None
    for name, target in cfg['targets'].items():
        if name in out:
            continue
        tcfg = dict(copy.deepcopy(cfg), **target)
        dicts = Dictionaries(tcfg, features, dev)
        if model is None:
            model = Model(cfg, dicts.sites, dev)
        dicts.load_full_target()
        nat = natural_statistics(model, dicts, natural, tcfg, run)
        dicts.drop_full_target()
        energy = err = kept = 0.
        cos_all, weights, sites = [], [], {}
        for site in dicts.sites:
            st = nat[site]
            dec_t = dicts.tgt_cpu[site]['W_dec'] / np.linalg.norm(dicts.tgt_cpu[site]['W_dec'], axis=1, keepdims=True)
            dec_s = dicts.src[site]['W_dec'].cpu().numpy()
            dec_s = dec_s / np.linalg.norm(dec_s, axis=1, keepdims=True)
            cos = dec_s @ dec_t.T
            rows, cols = linear_sum_assignment(-cos)
            match = cos[rows, cols]
            site_err = st['zs2'] - 2 * st['zs_hat'] + st['zs_hat2']
            energy += float(st['zs2'].sum())
            err += float(site_err.sum())
            kept += float(st['zs_hat'].sum())
            cos_all += match.tolist()
            weights += st['zs2'].tolist()
            sites[site] = dict(readback_error=float(site_err.sum() / st['zs2'].sum()),
                               retention=float(st['zs_hat'].sum() / st['zs2'].sum()),
                               hungarian_cos=match.tolist(),
                               target_fve=float(1 - st['res'] / (st['x2'] - (st['xsum'] ** 2).sum() / st['n'])),
                               paths=dicts.paths[site])
        out[name] = dict(target=target, readback_error=err / energy, retention=kept / energy,
                         hungarian_cos_mean=float(np.mean(cos_all)),
                         hungarian_cos_weighted=float(np.average(cos_all, weights=weights)),
                         target_fve_mean=float(np.mean([s['target_fve'] for s in sites.values()])), sites=sites)
        out_path.write_text(json.dumps(out, indent=1))
        run.records([dict(kind='readback', name=name, **{k: v for k, v in out[name].items() if k != 'sites'})])
        run.progress(stage='target_done', target=name, done=len(out), total=len(cfg['targets']))
        del dicts
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
