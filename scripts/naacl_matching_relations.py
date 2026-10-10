from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from ccad.artifacts import sha256
from train_shift_dictionaries import site_module


DESCRIPTION = ('Label-free feature matching for the published SHIFT explanation: Hungarian '
               'assignment on decoder cosine or natural-text activation correlation, and the two '
               'most correlated unused target features per source feature.')


def exclusive_top(score, per_source):
    relation = np.zeros(score.shape, dtype=np.float32)
    used = set()
    for j in range(score.shape[1]):
        for idx in np.argsort(-score[:, j], kind='stable'):
            if int(idx) not in used:
                relation[idx, j] = 1.
                used.add(int(idx))
            if int(relation[:, j].sum()) == per_source:
                break
    return relation


def hungarian(score):
    rows, cols = linear_sum_assignment(-score.T)
    relation = np.zeros(score.shape, dtype=np.float32)
    relation[cols, rows] = 1.
    assert np.all(relation.sum(0) == 1) and np.all(relation.sum(1) <= 1)
    return relation


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    torch.set_float32_matmul_precision('highest')
    device = torch.device(c['device'])
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
    source = json.loads(Path(c['source_manifest']).read_text())
    sites = list(source['members'])
    sb = np.load(c['source_parameters'])
    sp = {site: {k: torch.tensor(sb[site + '__' + k], device=device)
                 for k in ['encoder', 'encoder_bias', 'decoder', 'center']} for site in sites}
    targets = {}
    for site in sites:
        state = torch.load(Path(c['target_directory']) / f'{site}_seed{c["target_seed"]}.pt',
                           map_location=device, weights_only=True)
        sae = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(device)
        sae.load_state_dict(state)
        sae.requires_grad_(False)
        targets[site] = sae
    model = transformers.AutoModelForCausalLM.from_pretrained(
        c['model_local_dir'], local_files_only=True, dtype=torch.float32,
        attn_implementation='eager').eval().to(device)
    model.requires_grad_(False)
    stats = {}
    for site in sites:
        nt, ns = targets[site].encoder.weight.shape[0], len(source['members'][site])
        stats[site] = dict(n=0, st=torch.zeros(nt, device=device, dtype=torch.float64),
                           stt=torch.zeros(nt, device=device, dtype=torch.float64),
                           ss=torch.zeros(ns, device=device, dtype=torch.float64),
                           sss=torch.zeros(ns, device=device, dtype=torch.float64),
                           sts=torch.zeros(nt, ns, device=device, dtype=torch.float64))

    def hook(site):
        def f(module, inputs, out):
            x = (out[0] if isinstance(out, tuple) else out).flatten(0, 1)
            s, t, a = sp[site], targets[site], stats[site]
            zs = torch.relu((x - s['center']) @ s['encoder'].T + s['encoder_bias']).double()
            zt = t.encode(x).double()
            a['n'] += len(x)
            a['st'] += zt.sum(0)
            a['stt'] += zt.square().sum(0)
            a['ss'] += zs.sum(0)
            a['sss'] += zs.square().sum(0)
            a['sts'] += zt.T @ zs
            return None
        return f

    handles = [site_module(model, site).register_forward_hook(hook(site)) for site in sites]
    tokens = np.memmap(c['paired_tokens'], dtype='<u2', mode='r').reshape(-1, 128)[:c['natural_sequences']]
    with torch.no_grad():
        for off in range(0, len(tokens), c['batch_sequences']):
            ids = torch.tensor(np.array(tokens[off:off + c['batch_sequences']], dtype='int64'), device=device)
            model.gpt_neox(ids, use_cache=False)
    for h in handles:
        h.remove()
    export, summary = {}, {}
    for site in sites:
        a, t, s = stats[site], targets[site], sp[site]
        n = a['n']
        mt, ms = a['st'] / n, a['ss'] / n
        vt = (a['stt'] / n - mt.square()).clamp_min(0)
        vs = (a['sss'] / n - ms.square()).clamp_min(0)
        cov = a['sts'] / n - mt[:, None] * ms[None, :]
        corr = (cov / (vt.sqrt()[:, None] * vs.sqrt()[None, :]).clamp_min(1e-12)).cpu().numpy()
        corr[~np.isfinite(corr)] = 0.
        dt = t.decoder.weight.T.detach()
        ds = s['decoder']
        cos = ((dt / dt.norm(dim=1, keepdim=True)) @ (ds / ds.norm(dim=1, keepdim=True)).T).cpu().numpy()
        relations = dict(hungarian_decoder=hungarian(cos), hungarian_activation=hungarian(corr),
                         activation_top2=exclusive_top(corr, c['members_per_source']))
        for name, value in relations.items():
            export[site + '__' + name] = value
        summary[site] = dict(
            states=int(n), source_members=int(len(source['members'][site])),
            decoder_cosine_of_match=(cos * relations['hungarian_decoder']).sum(0).tolist(),
            activation_correlation_of_match=(corr * relations['hungarian_activation']).sum(0).tolist(),
            decoder_and_activation_agree=int((relations['hungarian_decoder'] *
                                              relations['hungarian_activation']).sum()))
    np.savez_compressed(out / 'matching_relations.npz', **export)
    record = dict(status='PASS', written_utc=datetime.now(timezone.utc).isoformat(),
                  config=c, script_sha256=sha256(Path(__file__)),
                  output_sha256=sha256(out / 'matching_relations.npz'), sites=summary)
    (out / 'MATCHING_RELATIONS.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(status='PASS', output=str(out), sites={k: v['states'] for k, v in summary.items()})))


if __name__ == '__main__':
    main()
