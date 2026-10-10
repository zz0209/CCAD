import json
import sys

import torch
import transformers

model_dir, layer, panel_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
saes = {name: (path, int(k)) for name, path, k in (a.split('=') for a in sys.argv[4:])}
device = torch.device('cuda:0')
tok = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation='eager').eval().to(device)
captured = {}
model.get_submodule(f'model.layers.{layer}').register_forward_hook(lambda m, i, o: captured.__setitem__('h', o[0] if isinstance(o, tuple) else o))
rows = json.loads(open(panel_path).read())['rows']
report = {}
for prefix_name, prefix in [('eos', [tok.eos_token_id]), ('newline', tok.encode('\n', add_special_tokens=False)),
                            ('none', [])]:
    states, positions = [], []
    with torch.no_grad():
        for r in rows[:384]:
            good = prefix + tok.encode(r['sentence_good'], add_special_tokens=False)
            bad = prefix + tok.encode(r['sentence_bad'], add_special_tokens=False)
            first = next(i for i, (a, b) in enumerate(zip(good, bad)) if a != b)
            model(torch.tensor([good], device=device), use_cache=False)
            states.append(captured['h'][0, first - 1])
            positions.append(first - 1)
    h = torch.stack(states)
    out = dict(mean_norm=float(h.norm(dim=-1).mean()), max_norm=float(h.norm(dim=-1).max()), position_zero=sum(p == 0 for p in positions))
    for name, (path, k) in saes.items():
        raw = torch.load(path, map_location=device, weights_only=True)
        pre = h @ raw['W_enc'].float().T + raw['b_enc'].float()
        top = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre).scatter(-1, top.indices, top.values)
        rec = z @ raw['W_dec'].float().T + raw['b_dec'].float()
        out[name] = float(1 - (rec - h).square().sum() / (h - h.mean(0)).square().sum())
    report[prefix_name] = out
print(json.dumps(report, indent=1))
