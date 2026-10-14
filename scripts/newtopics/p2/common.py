import json
import os
from pathlib import Path

import jlens
import torch
import transformers
from jlens.hf import Layout
from tokenizers import processors
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BAND = (0.38, 0.92)
GPU_CAP_GIB = 18.0


def cap_gpu_memory(gib=GPU_CAP_GIB):
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2 ** 30 / total), 0)


def manifest_entry(manifest, label):
    rows = [r for r in json.loads(Path(manifest).read_text()) if r['label'] == label]
    if len(rows) != 1:
        raise ValueError(f'{label}: {len(rows)} manifest rows')
    return rows[0]


def load_model(manifest, label, device='cuda'):
    row = manifest_entry(manifest, label)
    tok = AutoTokenizer.from_pretrained(row['local_path'])
    if int(transformers.__version__.split('.')[0]) < 5:
        # jlens forces BOS by setting add_bos_token, which transformers 5 turns into this template; transformers 4
        # fast tokenizers (GPT2TokenizerFast for OLMo-2) ignore the flag, so the same template is installed here.
        bos = tok.bos_token
        tok._tokenizer.post_processor = processors.TemplateProcessing(
            single=f'{bos}:0 $A:0', pair=f'{bos}:0 $A:0 {bos}:1 $B:1', special_tokens=[(bos, tok.bos_token_id)])
    hf = AutoModelForCausalLM.from_pretrained(row['local_path'], dtype=torch.bfloat16).to(device)
    # transformers 5 renamed GPT-NeoX's embed_out to lm_head, which the jlens built-in Pythia layout does not know;
    # transformers 4.x keeps embed_out and matches the built-in layout.
    layout = (Layout('gpt_neox', norm='final_layer_norm', embed='embed_in', lm_head='lm_head')
              if hf.config.model_type == 'gpt_neox' and hasattr(hf, 'lm_head') else None)
    lm = jlens.from_hf(hf, tok, layout=layout)
    ids = lm.encode('Fact: The capital of France is', max_length=128)
    if ids[0, 0].item() != tok.bos_token_id:
        raise ValueError(f'{label}: tokenizer did not prepend BOS ({ids[0, :3].tolist()})')
    return row, hf, tok, lm


def band_layers(n_layers):
    return [l for l in range(n_layers) if BAND[0] <= (l + 1) / n_layers <= BAND[1]]


def readout_vectors(lm, token_ids):
    # Rows u_t with logit_t ~ <u_t, x> / scale for the final norm applied to x: the norm gain folded into the
    # unembedding row, centred for LayerNorm (the centring projection is symmetric).
    norm = lm._final_norm
    w = lm._lm_head.weight[token_ids].float() * norm.weight.float()
    if isinstance(norm, nn.LayerNorm):
        w = w - w.mean(-1, keepdim=True)
    elif 'RMSNorm' not in type(norm).__name__:
        raise ValueError(f'unsupported final norm {type(norm).__name__}')
    return w


def first_token(tok, text):
    return tok.encode(' ' + text, add_special_tokens=False)[0]


def write_json(path, obj):
    tmp = f'{path}.tmp.{os.getpid()}'
    Path(tmp).write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)
