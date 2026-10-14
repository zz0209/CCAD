import contextlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from p1const import (ANIMALS, BASE_MODEL, DATASETS, EMBED_MODEL, LAYERS, MAX_LEN, M, REVIEWER_MODEL,  # noqa: F401
                     SECOND_SEED, read_jsonl, run_dir, save_json, stable_hash, write_jsonl)


def load_tokenizer(name=BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(name)
    assert tok.pad_token_id is not None
    return tok


def load_model(name=BASE_MODEL, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation='sdpa').cuda()
    model.eval()
    return model


def prompt_text(tok, user):
    return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)


def encode_pair(tok, user, response):
    p = tok(prompt_text(tok, user), add_special_tokens=False)['input_ids']
    r = tok(response + '<|im_end|>', add_special_tokens=False)['input_ids']
    return p, r


def layer_modules(model):
    inner = model
    while not hasattr(inner, 'layers'):
        inner = inner.model
    return inner.layers


@contextlib.contextmanager
def steer(model, layer, vec):
    """Adds vec to the residual stream output of decoder block `layer` at every position."""
    if vec is None:
        yield
        return

    def hook(mod, inp, out):
        assert isinstance(out, torch.Tensor)
        return out + vec.to(out.dtype)

    h = layer_modules(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextlib.contextmanager
def capture(model, layers):
    """Records the residual stream output of the given decoder blocks for the next forward pass."""
    store = {}
    hooks = []
    for l in layers:
        def hook(mod, inp, out, l=l):
            assert isinstance(out, torch.Tensor)
            store[l] = out
        hooks.append(layer_modules(model)[l].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def pad_batch(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.cuda(), mask.cuda()


def token_batches(lengths, max_tokens, max_batch):
    order = np.argsort(lengths)[::-1]
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            batches.append(cur)
            cur, m = [], lengths[i]
        cur.append(int(i))
        cur_max = m
    if cur:
        batches.append(cur)
    return batches
