from pathlib import Path
import argparse
import json
import random
import shutil
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemma_shift_transfer import (Dictionaries, Model, PartMap, Probe, Run, TargetMap, accuracies, collect_pooled,
                                  probe_scores, read_split, readout_mode, request_masks, sfc_ambiguous, sfc_balanced,
                                  source_mode, train_probe)

DESCRIPTION = ('Part-level SHIFT in Gemma-2-2B on the original professor/nurse classifier and on later classifiers that '
               'never supervised any correspondence. Each annotated part is deleted alone through each saved '
               'correspondence of a transfer run, and every classifier is evaluated before and after retraining its '
               'linear head on the edited biased data.')


def later_train(data, negative, positive, orientation, seed):
    # Biased training data: the negative profession appears only with gender `orientation`, the positive one with the other.
    text, prof, gen = data
    neg = [x for x, p, g in zip(text, prof, gen) if p == negative and g == orientation]
    pos = [x for x, p, g in zip(text, prof, gen) if p == positive and g == 1 - orientation]
    n = min(len(neg), len(pos))
    texts, labels = neg[:n] + pos[:n], [0] * n + [1] * n
    idx = list(range(2 * n))
    random.Random(seed).shuffle(idx)
    return [texts[i] for i in idx], np.array([labels[i] for i in idx])


def later_test(data, negative, positive, seed):
    # Balanced over profession and gender; the same biographies serve both confounding orientations of a pair.
    text, prof, gen = data
    groups = [[x for x, p, g in zip(text, prof, gen) if p == pp and g == gg]
              for pp, gg in [(negative, 0), (negative, 1), (positive, 0), (positive, 1)]]
    n = min(len(g) for g in groups)
    texts = sum((g[:n] for g in groups), [])
    true = np.array([0] * (2 * n) + [1] * (2 * n))
    gender = np.array([0] * n + [1] * n + [0] * n + [1] * n)
    idx = list(range(4 * n))
    random.Random(seed).shuffle(idx)
    return [texts[i] for i in idx], true[idx], gender[idx]


def train_head(acts, labels, cfg, epochs, dev):
    order = np.concatenate([np.random.default_rng(cfg['probe_seed'] + e).permutation(len(acts)) for e in range(epochs)])
    return train_probe(acts[order], labels[order], cfg['probe_seed'], cfg['probe_lr'], cfg['probe_batch'], dev)


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
    tasks = {'shift': dict(train=amb_train, train_y=amb_train_y, test=bal_test, true=bal_true, spur=bal_spur, epochs=None)}
    for pair in cfg['later_pairs']:
        test_texts, true, gender = later_test(test, pair['negative'], pair['positive'], cfg['data_seed'])
        for orientation in (0, 1):
            texts, labels = later_train(train, pair['negative'], pair['positive'], orientation, cfg['data_seed'])
            # The spurious attribute is the gender that the biased data pairs with the positive profession.
            tasks[f'{pair["name"]}_o{orientation}'] = dict(train=texts, train_y=labels, test=test_texts, true=true,
                                                           spur=gender ^ orientation, epochs=cfg['later_epochs'])
    if cfg.get('limit_task'):
        for task in tasks.values():
            for key in ['train', 'train_y', 'test', 'true', 'spur']:
                task[key] = task[key][:cfg['limit_task']]
    union = list(dict.fromkeys(t for task in tasks.values() for t in task['train'] + task['test']))
    pos = {t: i for i, t in enumerate(union)}
    for task in tasks.values():
        task['train_ix'] = np.array([pos[t] for t in task['train']])
        task['test_ix'] = np.array([pos[t] for t in task['test']])
    # Length-sorted batches reduce padding; every method uses the same order.
    order = sorted(range(len(union)), key=lambda i: len(union[i]))
    sorted_texts = [union[i] for i in order]
    run.records([dict(kind='data', union=len(union), **{f'{k}_train': len(v['train']) for k, v in tasks.items()},
                      **{f'{k}_test': len(v['test']) for k, v in tasks.items()})])
    (run.dir / 'task_labels.json').write_text(json.dumps({name: dict(true=task['true'].tolist(), spur=task['spur'].tolist())
                                                          for name, task in tasks.items()}))

    def pooled(make, label):
        acts = collect_pooled(model, sorted_texts, cfg['use_batch'], make, run, label)
        out = np.empty_like(acts)
        out[order] = acts
        return out

    clean = pooled(None, 'clean')
    originals = {'shift': probe}
    results_path, logits_path = run.dir / 'TASK_USE.json', run.dir / 'task_logits.npz'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    logits = dict(np.load(logits_path)) if logits_path.exists() else {}
    for name, task in tasks.items():
        if name != 'shift':
            originals[name] = train_head(clean[task['train_ix']], task['train_y'], cfg, task['epochs'], dev)
        scores = probe_scores(originals[name], clean[task['test_ix']], dev)
        results[f'{name}|none|clean'] = dict(original=accuracies(scores, task['true'], task['spur']))
        logits[f'{name}|none|clean|original'] = scores.astype(np.float32)
    del clean

    saved = json.loads((source_run / 'maps.json').read_text())
    maps = {name: TargetMap(dicts, {s: np.array(v['idx']) for s, v in saved[name].items()},
                            {s: torch.tensor(np.array(v['weights'], dtype=np.float32), device=dev) for s, v in saved[name].items()})
            for name in cfg['methods'] if name not in ('source', 'readout', 'rediscovered')}
    if 'rediscovered' in cfg['methods']:
        # The explanation selected again in the target dictionary by the fixed rediscovery procedure.
        maps['rediscovered'] = PartMap(dicts, json.loads(Path(cfg['rediscovery']).read_text()), features)
    if 'readout' in cfg['methods']:
        dicts.load_full_target()
    total, done = len(cfg['parts']) * len(cfg['methods']), 0
    for part in cfg['parts']:
        q = torch.tensor(masks[part], device=dev)
        for method in cfg['methods']:
            done += 1
            if all(f'{name}|{part}|{method}' in results for name in tasks):
                continue
            if method == 'source':
                make = lambda b: source_mode(dicts, q.expand(b, -1))
            elif method == 'readout':
                make = lambda b: readout_mode(dicts, q.expand(b, -1))
            else:
                make = (lambda m: lambda b: m.mode(q.expand(b, -1)))(maps[method])
            acts = pooled(make, f'{part}_{method}')
            for name, task in tasks.items():
                test_acts = acts[task['test_ix']]
                if task['epochs'] is None:
                    head = train_probe(acts[task['train_ix']], task['train_y'], cfg['probe_seed'], cfg['probe_lr'],
                                       cfg['probe_batch'], dev)
                else:
                    head = train_head(acts[task['train_ix']], task['train_y'], cfg, task['epochs'], dev)
                row = {}
                for classifier, model_head in [('original', originals[name]), ('retrained', head)]:
                    scores = probe_scores(model_head, test_acts, dev)
                    row[classifier] = accuracies(scores, task['true'], task['spur'])
                    logits[f'{name}|{part}|{method}|{classifier}'] = scores.astype(np.float32)
                results[f'{name}|{part}|{method}'] = row
                run.records([dict(kind='task_use', task=name, part=part, method=method, classifier=k, **v)
                             for k, v in row.items()])
            results_path.write_text(json.dumps(results, indent=1))
            np.savez_compressed(logits_path, **logits)
            run.progress(stage='task_use', part=part, method=method, done=done, total=total)
    run.progress(stage='done')


if __name__ == '__main__':
    main()
