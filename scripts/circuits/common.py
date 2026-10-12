import json

import torch
from huggingface_hub import list_repo_files

from circuit_tracer.utils.hf_utils import load_transcoder_from_hub, load_transcoders

GEMMA_SCOPE_TC = 'google/gemma-scope-2b-pt-transcoders'


def load_target(ref, device, dtype):
    # A Hugging Face transcoder reference, or gemmascope-l0-rank:R for the Gemma Scope 16k transcoders at the R-th
    # lowest L0 of every layer, or the highest L0 where a layer has fewer releases (rank 0 is the set that
    # circuit-tracer ships for Gemma-2-2B).
    if not ref.startswith('gemmascope-l0-rank:'):
        return load_transcoder_from_hub(ref, device=device, dtype=dtype, lazy_decoder=True)[0]
    rank = int(ref.split(':')[1])
    per = {}
    for f in list_repo_files(GEMMA_SCOPE_TC):
        if f.endswith('params.npz') and '/width_16k/' in f:
            layer, _, l0, _ = f.split('/')
            per.setdefault(int(layer.split('_')[1]), []).append((int(l0.split('_')[-1]), f))
    paths = [f'hf://{GEMMA_SCOPE_TC}/{sorted(per[l])[min(rank, len(per[l]) - 1)][1]}' for l in sorted(per)]
    # circuit-tracer chooses its npz loader when repo_id contains "gemma-scope" and its Gemma Scope 2 loader when it
    # contains "gemma-scope-2", which the name of this repository also does; repo_id only selects the loader here.
    config = dict(model_kind='transcoder_set', repo_id='gemma-scope/2b-pt-transcoders', scan_name=f'gemma-scope-tc-l0-rank{rank}',
                  feature_input_hook='ln2.hook_normalized', feature_output_hook='hook_mlp_out', transcoders=paths)
    return load_transcoders(config, device=device, dtype=dtype, lazy_decoder=True)


def choose_prompt(prompt, active_fraction):
    # Link texts sometimes drop a trailing space or change case; keep the variant on which the most source features
    # are active at their published positions.
    variants = [prompt, prompt + ' ', prompt.rstrip(), prompt.replace(': The ', ': the ', 1)]
    scored = [(active_fraction(v), -k, v) for k, v in enumerate(dict.fromkeys(variants))]
    return max(scored)[2], max(scored)[0]


def prompt_of(graph):
    # The published link text gives the prompt between the first pair of backticks.
    for label in graph['labels']:
        if label.count('`') >= 2:
            return label.split('`')[1].replace('\\n', '\n')
    return None


def gemma_items(registry_path, model_key='gemma-2-2b'):
    # Published base-model explanations of one model (registry prefix, e.g. gemma-2-2b or llama-3.2-1b) with a
    # prompt and supernodes; Gemma graphs drawn on the instruction-tuned model are left out.
    graphs = json.loads(open(registry_path).read())['graphs']
    items = []
    for key, g in graphs.items():
        if not key.startswith(f'{model_key}/') or key.startswith('gemma-2-2b/gemma-it'):
            continue
        prompt, sn = prompt_of(g), g['variants'][0]['supernodes']
        if prompt is not None and sn:
            items.append((key, prompt, sn))
    return items


def filter_items(items, n_layers, width):
    # Logit nodes (layer >= n_layers) are not dictionary features and leave their supernode; a graph with a feature
    # index beyond the dictionary width was drawn with another dictionary and is excluded as a whole.
    kept, dropped = [], []
    for key, prompt, sn in items:
        if any(l < n_layers and f >= width for fs in sn.values() for l, _, f in fs):
            dropped.append(key)
            continue
        sn = {name: [x for x in fs if x[0] < n_layers] for name, fs in sn.items()}
        sn = {name: fs for name, fs in sn.items() if fs}
        if sn:
            kept.append((key, prompt, sn))
    return kept, dropped


class Dictionary:
    # A transcoder set or cross-layer transcoder read at its input hook; feature contributions are written to
    # hook_mlp_out of the feature's layer (per-layer transcoder) or of that layer and every later one (CLT).
    def __init__(self, transcoders):
        self.t = transcoders
        self.input_hook = transcoders.feature_input_hook
        p = next(transcoders.parameters())
        self.dtype, self.device = p.dtype, p.device

    def encode(self, x, layer, feats=None):
        z = self.t.encode_layer(x.to(self.dtype), layer)
        return z if feats is None else z[..., feats]

    def decoders(self, layer, feats):
        # (n, n_out, d_model) decoder slices; output layers run from `layer` on.
        dec = self.t._get_decoder_vectors(layer, torch.as_tensor(feats, device=self.device)).float()
        return dec[:, None] if dec.ndim == 2 else dec


def add_hooks(deltas):
    # deltas: {layer: (seq, d_model) tensor} added to blocks.layer.hook_mlp_out of a batch of one sequence or
    # {layer: (batch, seq, d_model)}.
    hooks = []
    for layer, delta in deltas.items():
        def hook(act, hook, delta=delta):
            return act + delta.to(act.dtype).reshape(act.shape)
        hooks.append((f'blocks.{layer}.hook_mlp_out', hook))
    return hooks
