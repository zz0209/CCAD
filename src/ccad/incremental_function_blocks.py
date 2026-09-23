from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dictionary_learning.trainers.top_k import AutoEncoderTopK


PARTS = ('verb', 'number', 'gender')
TASKS = ('regular_plural_subject_verb_agreement_1', 'anaphor_number_agreement',
         'anaphor_gender_agreement')


def load_dictionary(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    ae = AutoEncoderTopK(state['encoder.weight'].shape[1], len(state['encoder.weight']),
                        int(state['k'])).to(device).float()
    ae.load_state_dict(state)
    ae.b_dec.requires_grad_(False)
    return ae


def encode(ae, x, groups, allocation):
    assert x.dtype == ae.encoder.weight.dtype == torch.float32
    assert allocation in ('global', 'group')
    with torch.autocast(device_type=x.device.type, enabled=False):
        pre = F.linear(x-ae.b_dec, ae.encoder.weight, ae.encoder.bias).relu()
        if allocation == 'global':
            values, indices = pre.topk(64, dim=-1, sorted=False)
        else:
            names = [name for name in PARTS if name in groups]
            selected = [(name, pre[..., groups[name]].topk(8, dim=-1, sorted=False)) for name in names]
            selected.append(('rest', pre[..., groups['rest']].topk(64-8*len(names), dim=-1, sorted=False)))
            values = torch.cat([item.values for _, item in selected], dim=-1)
            indices = torch.cat([groups[name][item.indices] for name, item in selected], dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)


def group_delete(ae, x, groups, allocation, parts):
    ids = torch.cat([groups[name] for name in parts])
    codes = encode(ae, x, groups, allocation)
    return -codes[..., ids] @ ae.decoder.weight[:, ids].T


def assign_groups(decoder, source_decoders, available=None):
    names = [name for name in PARTS if name in source_decoders]
    target = F.normalize(decoder.detach().double().cpu().T, dim=1).numpy()
    available = np.arange(len(target)) if available is None else np.asarray(available, dtype=np.int64)
    scores = np.stack([np.maximum(target[available] @ F.normalize(
        torch.as_tensor(source_decoders[name], dtype=torch.float64), dim=1).numpy().T, 0).max(1)
        for name in names], axis=1)
    local_ids, factor_ids = np.indices(scores.shape)
    order = np.lexsort((factor_ids.ravel(), available[local_ids].ravel(), -scores.ravel()))
    assigned, selected = set(), {name: [] for name in names}
    for position in order:
        feature, name = int(available[position//len(names)]), names[position % len(names)]
        if feature not in assigned and len(selected[name]) < 256:
            selected[name].append(feature)
            assigned.add(feature)
        if len(assigned) == 256*len(names):
            break
    assert all(len(ids) == 256 for ids in selected.values())
    selected['rest'] = sorted(set(available.tolist())-assigned)
    return {name: torch.tensor(sorted(ids), dtype=torch.long) for name, ids in selected.items()}, scores


def validate_groups(groups, count=8192):
    assert all(len(ids) == 256 for name, ids in groups.items() if name != 'rest')
    assert torch.equal(torch.cat([ids.cpu() for ids in groups.values()]).sort().values, torch.arange(count))


def configure_trainable(ae, groups, frozen_parts):
    ae.encoder.requires_grad_(True)
    ae.decoder.requires_grad_(True)
    ae.b_dec.requires_grad_(False)
    frozen_ids = torch.cat([groups[name] for name in frozen_parts]) if frozen_parts else torch.empty(0, dtype=torch.long, device=ae.b_dec.device)
    trainable = torch.ones(ae.dict_size, dtype=torch.bool, device=ae.b_dec.device)
    trainable[frozen_ids] = False
    frozen = dict(ids=frozen_ids, encoder=ae.encoder.weight.detach()[frozen_ids].clone(),
                  bias=ae.encoder.bias.detach()[frozen_ids].clone(),
                  decoder=ae.decoder.weight.detach()[:, frozen_ids].clone(), b_dec=ae.b_dec.detach().clone())
    return trainable, frozen


@torch.no_grad()
def normalize_decoder(ae, trainable):
    ae.decoder.weight[:, trainable] = F.normalize(ae.decoder.weight[:, trainable], dim=0)


@torch.no_grad()
def constrain_gradients(ae, trainable):
    ae.encoder.weight.grad[~trainable] = 0
    ae.encoder.bias.grad[~trainable] = 0
    ae.decoder.weight.grad[:, ~trainable] = 0
    directions = F.normalize(ae.decoder.weight[:, trainable], dim=0)
    gradient = ae.decoder.weight.grad[:, trainable]
    ae.decoder.weight.grad[:, trainable] = gradient-directions*(directions*gradient).sum(0, keepdim=True)


@torch.no_grad()
def restore_frozen(ae, frozen):
    ids = frozen['ids']
    ae.encoder.weight[ids] = frozen['encoder']
    ae.encoder.bias[ids] = frozen['bias']
    ae.decoder.weight[:, ids] = frozen['decoder']
    assert torch.equal(ae.b_dec, frozen['b_dec'])


def check_frozen(ae, frozen):
    ids = frozen['ids']
    return bool(torch.equal(ae.encoder.weight[ids], frozen['encoder'])
        and torch.equal(ae.encoder.bias[ids], frozen['bias'])
        and torch.equal(ae.decoder.weight[:, ids], frozen['decoder'])
        and torch.equal(ae.b_dec, frozen['b_dec']))


def save_checkpoint(path, ae, groups, metadata):
    validate_groups(groups)
    assert not Path(path).exists()
    torch.save(dict(dictionary={key: value.detach().cpu() for key, value in ae.state_dict().items()},
                    groups={name: ids.cpu() for name, ids in groups.items()}, metadata=metadata), path)


def load_checkpoint(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    state = saved['dictionary']
    ae = AutoEncoderTopK(state['encoder.weight'].shape[1], len(state['encoder.weight']), int(state['k'])).to(device).float()
    ae.load_state_dict(state)
    ae.b_dec.requires_grad_(False)
    groups = {name: ids.to(device) for name, ids in saved['groups'].items()}
    validate_groups(groups)
    return ae, groups, saved['metadata']
