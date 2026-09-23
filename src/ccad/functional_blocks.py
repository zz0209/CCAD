from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from sparsify import SparseCoder
from sparsify.fused_encoder import EncoderOutput


FACTORS = ('number', 'time')
QUOTAS = {'number': 8, 'time': 8, 'rest': 48}


def validate_groups(groups):
    assert [len(groups[name]) for name in (*FACTORS, 'rest')] == [256, 256, 7680]
    all_ids = torch.cat([groups[name].cpu() for name in (*FACTORS, 'rest')])
    assert torch.equal(all_ids.sort().values, torch.arange(8192))


def assign_groups(decoder, source_decoder, source_members):
    target = decoder.detach().double().cpu().numpy()
    source = source_decoder.detach().double().cpu().numpy()
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    scores = np.stack([np.maximum(target @ source[source_members[name]].T, 0).max(1)
                       for name in FACTORS], axis=1)
    feature_ids, factor_ids = np.indices(scores.shape)
    order = np.lexsort((factor_ids.ravel(), feature_ids.ravel(), -scores.ravel()))
    assigned, groups = set(), {name: [] for name in FACTORS}
    for position in order:
        feature, factor = int(position // 2), FACTORS[int(position % 2)]
        if feature not in assigned and len(groups[factor]) < 256:
            groups[factor].append(feature)
            assigned.add(feature)
        if len(assigned) == 512:
            break
    groups['rest'] = sorted(set(range(8192))-assigned)
    groups = {name: torch.tensor(sorted(ids), dtype=torch.long) for name, ids in groups.items()}
    validate_groups(groups)
    return groups, scores


def encode(sae, x, groups, allocation):
    assert x.dtype == sae.encoder.weight.dtype == sae.W_dec.dtype == torch.float32
    assert allocation in ('global', 'group') and not sae.cfg.transcode
    with torch.autocast(device_type=x.device.type, enabled=False):
        pre_acts = F.linear(x-sae.b_dec, sae.encoder.weight, sae.encoder.bias).relu()
        if allocation == 'global':
            values, indices = pre_acts.topk(64, dim=-1, sorted=False)
        else:
            selected = [pre_acts[..., groups[name]].topk(QUOTAS[name], dim=-1, sorted=False)
                        for name in (*FACTORS, 'rest')]
            values = torch.cat([item.values for item in selected], dim=-1)
            indices = torch.cat([groups[name][item.indices]
                                 for name, item in zip((*FACTORS, 'rest'), selected)], dim=-1)
    return EncoderOutput(values, indices, pre_acts)


def dense_codes(sae, x, groups, allocation):
    out = encode(sae, x, groups, allocation)
    return torch.zeros_like(out.pre_acts).scatter(-1, out.top_indices, out.top_acts)


def save_checkpoint(sae, directory, groups, metadata):
    directory = Path(directory)
    assert not directory.exists()
    validate_groups(groups)
    sae.save_to_disk(directory)
    (directory/'functional_groups.json').write_text(json.dumps(dict(metadata,
        groups={name: ids.cpu().tolist() for name, ids in groups.items()}), indent=2), encoding='utf-8')


def load_checkpoint(directory, device):
    directory = Path(directory)
    metadata = json.loads((directory/'functional_groups.json').read_text(encoding='utf-8'))
    groups = {name: torch.tensor(ids, dtype=torch.long, device=device)
              for name, ids in metadata['groups'].items()}
    validate_groups(groups)
    assert metadata['allocation'] in ('global', 'group')
    sae = SparseCoder.load_from_disk(directory, device=device).float()
    return sae, groups, metadata
