from pathlib import Path
import argparse
import json
import random
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_shift_transfer import (NURSE, PROFESSOR, Dictionaries, Model, Probe, Run, TargetMap, accuracies, collect_pooled,
                                  probe_scores, read_split, readout_mode, request_masks, sfc_ambiguous, sfc_balanced,
                                  source_mode, train_probe)

DESCRIPTION = ('Part-level SHIFT in Gemma-2-2B with the saved correspondences of a final transfer run. Each part of the '
               'explanation is deleted alone through each method, the linear classifier is retrained on the edited '
               'biased data and evaluated on balanced data, so methods can be compared on the decision a part supports.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    run = Run(cfg, args.config, args.resume)
    try:
        execute(cfg, run)
    except BaseException as exc:
        run.status('FAIL', error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
        raise
    run.status('PASS')


def execute(cfg, run):
    dev = torch.device('cuda:0')
    torch.manual_seed(cfg['seed'])
    source_run = Path(cfg['source_run'])
    assert json.loads((source_run / 'status.json').read_text())['status'] == 'PASS', source_run
    base = json.loads((source_run / 'config.resolved.json').read_text())
    for key in ['model_dir', 'dataset_dir', 'features', 'l0_rule', 'res_root', 'att_root', 'source_width', 'target_width',
                'target_choice', 'probe_layer', 'data_seed', 'probe_seed', 'probe_lr', 'probe_batch', 'expected_features',
                'layer_bands', 'member_subsets', 'member_seed']:
        assert cfg[key] == base[key], key
    manifest = json.loads(Path(cfg['features']).read_text())
    features = [dict(site=site, index=int(idx), part=part, comment=comment)
                for site, rows in manifest['features'].items() for idx, part, comment in rows]
    masks, _ = request_masks(cfg, features)
    dicts = Dictionaries(cfg, features, dev)
    assert dicts.paths == json.loads((source_run / 'dictionaries.json').read_text())
    model = Model(cfg, dicts.sites, dev)
    probe = Probe(2304).to(dev)
    probe.load_state_dict(torch.load(source_run / 'probe.pt', map_location=dev, weights_only=True))
    probe.requires_grad_(False)
    train, test = read_split(cfg, 'train'), read_split(cfg, 'test')
    amb_train, amb_train_y, _ = sfc_ambiguous(train, cfg['data_seed'])
    bal_test, bal_true, bal_spur = sfc_balanced(test, cfg['data_seed'])
    run.records([dict(kind='data', ambiguous_train=len(amb_train), balanced_test=len(bal_test))])
    saved = json.loads((source_run / 'maps.json').read_text())
    maps = {name: TargetMap(dicts, {s: np.array(v['idx']) for s, v in saved[name].items()},
                            {s: torch.tensor(np.array(v['weights'], dtype=np.float32), device=dev) for s, v in saved[name].items()})
            for name in cfg['maps']}
    if 'readout' in cfg['methods']:
        dicts.load_full_target()
    out_path = run.dir / 'PART_USE.json'
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for part in cfg['parts']:
        q = torch.tensor(masks[part], device=dev)
        executors = {}
        for name in cfg['methods']:
            if name == 'source':
                executors[name] = lambda b: source_mode(dicts, q.expand(b, -1))
            elif name == 'readout':
                executors[name] = lambda b: readout_mode(dicts, q.expand(b, -1))
            else:
                executors[name] = (lambda m: lambda b: m.mode(q.expand(b, -1)))(maps[name])
        for name, make in executors.items():
            key = f'{part}:{name}'
            if key in results:
                continue
            test_acts = collect_pooled(model, bal_test, cfg['use_batch'], make, run, f'{part}_test_{name}')
            train_acts = collect_pooled(model, amb_train, cfg['use_batch'], make, run, f'{part}_train_{name}')
            retrained = train_probe(train_acts, amb_train_y, cfg['probe_seed'], cfg['probe_lr'], cfg['probe_batch'], dev)
            results[key] = dict(original=accuracies(probe_scores(probe, test_acts, dev), bal_true, bal_spur),
                                retrained=accuracies(probe_scores(retrained, test_acts, dev), bal_true, bal_spur))
            run.records([dict(kind='part_use', part=part, method=name, classifier=k, **v) for k, v in results[key].items()])
            out_path.write_text(json.dumps(results, indent=1))
    run.progress(stage='done')


if __name__ == '__main__':
    main()
