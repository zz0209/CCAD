import json
import sys

import numpy as np
import torch
import transformers

model_dir, layer, sae_path, k = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
tokens_path, tok_dir = sys.argv[5], sys.argv[6]
device = torch.device('cuda:0')
tok = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation='eager').eval().to(device)
raw = torch.load(sae_path, map_location=device, weights_only=True)
w_enc, b_enc, w_dec, b_dec = raw['W_enc'].float(), raw['b_enc'].float(), raw['W_dec'].float(), raw['b_dec'].float()
captured = {}
model.get_submodule(f'model.layers.{layer}').register_forward_hook(lambda m, i, o: captured.__setitem__('h', o[0] if isinstance(o, tuple) else o))
pythia = transformers.AutoTokenizer.from_pretrained(tok_dir)
tokens = np.memmap(tokens_path, dtype='<u2', mode='r').reshape(-1, 128)[:64]
texts = [pythia.decode(t) for t in tokens]
report = {}
with torch.no_grad():
    for variant in ['no_bos', 'with_bos']:
        hs = []
        for off in range(0, len(texts), 16):
            batch = texts[off:off + 16]
            if variant == 'with_bos':
                batch = [tok.eos_token + t for t in batch]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            model(**enc, use_cache=False)
            h = captured['h']
            mask = enc['attention_mask'].bool().clone()
            mask[:, 0] = False
            hs.append(h[mask])
            first = h[:, 0]
        h = torch.cat(hs)
        pre = h @ w_enc.T + b_enc
        top = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre).scatter(-1, top.indices, torch.relu(top.values))
        rec = z @ w_dec.T + b_dec
        fve = float(1 - (rec - h).square().sum() / (h - h.mean(0)).square().sum())
        report[variant] = dict(fve_excluding_first=fve, mean_norm=float(h.norm(dim=-1).mean()), first_token_norm=float(first.norm(dim=-1).mean()),
                               l0=float((z > 0).sum(-1).float().mean()), padding_side=tok.padding_side)
print(json.dumps(report, indent=1))
