from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ccad.artifacts import sha256

DESCRIPTION = ('Label-free feature matching between the GPT2Medium source explanation and each target '
               'dictionary, written in the fixed-map format read by train_grammar_member_program.py.')


def hungarian(score):
    rows, cols = linear_sum_assignment(-score.T)
    return cols[np.argsort(rows)]


def exclusive_top(score, per_source):
    chosen, used = [], set()
    for j in range(score.shape[1]):
        picked = []
        for idx in np.argsort(-score[:, j], kind='stable'):
            if int(idx) not in used:
                picked.append(int(idx))
                used.add(int(idx))
            if len(picked) == per_source:
                break
        chosen.append(picked)
    return chosen


def write_map(path, member_targets, part_ids):
    ids, gate = [], []
    for targets, part in zip(member_targets, part_ids):
        for t in np.atleast_1d(targets):
            ids.append(int(t))
            row = np.zeros(3, dtype=np.float32)
            row[part] = 1.
            gate.append(row)
    ids = np.array(ids, dtype=np.int64)
    assert len(set(ids.tolist())) == len(ids)
    np.savez_compressed(path, target_members=ids, partition64=np.stack(gate))
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(c['device'])
    torch.set_float32_matmul_precision('highest')
    sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
    from dictionary_learning.trainers.top_k import AutoEncoderTopK

    def load(path):
        state = torch.load(path, map_location=device, weights_only=True)
        ae = AutoEncoderTopK(state['encoder.weight'].shape[1], state['encoder.weight'].shape[0], int(state['k'])).to(device)
        ae.load_state_dict(state)
        return ae.eval().requires_grad_(False)

    source = load(c['source_checkpoint'])
    gate = np.load(Path(c['source_run']) / f'topk_s{c["source_seed"]}_source.npz')['gate']
    members = np.flatnonzero(gate.sum(1))
    part_ids = gate[members].argmax(1)
    assert len(members) == 192
    hidden = np.load(c['natural_states'])['hidden'][:c['natural_count']]
    record = dict(written_utc=None, config=c, script_sha256=sha256(Path(__file__)), targets={})
    for seed in c['target_seeds']:
        target = load(c['target_checkpoint_template'].format(seed=seed))
        nt = target.encoder.weight.shape[0]
        n = 0
        st = torch.zeros(nt, device=device, dtype=torch.float64)
        stt = torch.zeros_like(st)
        ss = torch.zeros(len(members), device=device, dtype=torch.float64)
        sss = torch.zeros_like(ss)
        sts = torch.zeros(nt, len(members), device=device, dtype=torch.float64)
        ids = torch.tensor(members, device=device)
        with torch.no_grad():
            for off in range(0, len(hidden), 4096):
                h = torch.tensor(hidden[off:off + 4096], device=device, dtype=torch.float32)
                zs = source.encode(h)[:, ids].double()
                zt = target.encode(h).double()
                n += len(h)
                st += zt.sum(0)
                stt += zt.square().sum(0)
                ss += zs.sum(0)
                sss += zs.square().sum(0)
                sts += zt.T @ zs
        mt, ms = st / n, ss / n
        cov = sts / n - mt[:, None] * ms[None, :]
        sd = ((stt / n - mt.square()).clamp_min(0).sqrt()[:, None] * (sss / n - ms.square()).clamp_min(0).sqrt()[None, :])
        corr = (cov / sd.clamp_min(1e-12)).cpu().numpy()
        corr[~np.isfinite(corr)] = 0.
        dt = target.decoder.weight.T
        ds = source.decoder.weight[:, ids].T
        cos = ((dt / dt.norm(dim=1, keepdim=True)) @ (ds / ds.norm(dim=1, keepdim=True)).T).cpu().numpy()
        maps = dict(hungarian_decoder=hungarian(cos), hungarian_activation=hungarian(corr),
                    activation_top2=exclusive_top(corr, c['members_per_source']))
        summary = dict(states=int(n))
        for name, member_targets in maps.items():
            count = write_map(out / f'{name}_s{c["source_seed"]}_t{seed}.npz', member_targets, part_ids)
            summary[name] = dict(target_features=count)
        hd, ha = maps['hungarian_decoder'], maps['hungarian_activation']
        summary['hungarian_decoder']['mean_cosine'] = float(cos[hd, np.arange(len(members))].mean())
        summary['hungarian_activation']['mean_correlation'] = float(corr[ha, np.arange(len(members))].mean())
        summary['decoder_activation_agree'] = int((hd == ha).sum())
        record['targets'][str(seed)] = summary
        print(json.dumps(dict(target_seed=seed, **summary)), flush=True)
    record['written_utc'] = datetime.now(timezone.utc).isoformat()
    record['status'] = 'PASS'
    (out / 'MATCHING_MAPS.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
