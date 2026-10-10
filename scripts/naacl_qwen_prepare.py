from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import torch
import transformers

from ccad.artifacts import sha256

DESCRIPTION = ('Prepare the Qwen-Scope grammar study: convert public TopK SAEs, retokenize the BLiMP panels, '
               'collect natural residual states and select the three source explanation groups.')
TASKS = ['regular_plural_subject_verb_agreement_1', 'anaphor_number_agreement', 'anaphor_gender_agreement']


def convert(path, k, out):
    raw = torch.load(path, map_location='cpu', weights_only=True)
    w_enc, b_enc, w_dec, b_dec = raw['W_enc'].float(), raw['b_enc'].float(), raw['W_dec'].float(), raw['b_dec'].float()
    assert w_enc.shape[0] == w_dec.shape[1] and w_enc.shape[1] == w_dec.shape[0]
    # The TopK encoder subtracts b_dec before encoding; the shifted bias reproduces the original pre-activation.
    state = {'encoder.weight': w_enc, 'encoder.bias': b_enc + w_enc @ b_dec, 'decoder.weight': w_dec, 'b_dec': b_dec,
             'k': torch.tensor(k, dtype=torch.int), 'threshold': torch.tensor(-1., dtype=torch.float32)}
    torch.save(state, out)
    return state


def retokenize(rows, tok, prefix_text):
    # A newline prefix keeps the first scored word away from the attention-sink position, whose residual norm
    # is two orders of magnitude larger than ordinary tokens in Qwen3.
    prefix = tok.encode(prefix_text, add_special_tokens=False)
    eos = tok.eos_token_id
    out = []
    for r in rows:
        good = prefix + tok.encode(r['sentence_good'], add_special_tokens=False) + [eos]
        bad = prefix + tok.encode(r['sentence_bad'], add_special_tokens=False) + [eos]
        first = next(i for i, (a, b) in enumerate(zip(good, bad)) if a != b)
        assert first >= len(prefix) + 1
        out.append(dict(task=r['task'], row_id=r['row_id'], split=r['split'], sentence_good=r['sentence_good'],
                        sentence_bad=r['sentence_bad'], good=good, bad=bad, position=first - 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_float32_matmul_precision('highest')
    tok = transformers.AutoTokenizer.from_pretrained(c['model_dir'])
    model = transformers.AutoModelForCausalLM.from_pretrained(c['model_dir'], dtype=torch.float32,
                                                              attn_implementation='eager').eval().to(device)
    model.requires_grad_(False)
    layer = model.get_submodule(c['hook_module_path'])
    saes = {}
    for seed, spec in c['saes'].items():
        state = convert(spec['path'], spec['k'], out / f'topk_seed{seed}.pt')
        saes[seed] = {k: v.to(device) for k, v in state.items()}
    panels = {}
    for name, path in c['panels'].items():
        rows = [r for r in json.loads(Path(path).read_text())['rows'] if r['task'] in TASKS]
        panels[name] = retokenize(rows, tok, c['sentence_prefix_text'])
        (out / f'{name}_panel.json').write_text(json.dumps(dict(rows=panels[name], scope=c['scope'])))
    captured = {}

    def hook(module, inputs, output):
        captured['h'] = output[0] if isinstance(output, tuple) else output
        return None

    handle = layer.register_forward_hook(hook)
    pythia = transformers.AutoTokenizer.from_pretrained(c['natural_tokenizer_dir'])
    tokens = np.memmap(c['natural_tokens'], dtype='<u2', mode='r').reshape(-1, 128)[:c['natural_sequences']]
    rng = np.random.default_rng(c['natural_seed'])
    states = []
    with torch.no_grad():
        for off in range(0, len(tokens), 16):
            texts = [pythia.decode(t) for t in tokens[off:off + 16]]
            enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            model(**enc, use_cache=False)
            keep = enc['attention_mask'].bool().clone()
            keep[:, 0] = False
            h = captured['h'][keep]
            states.append(h[torch.tensor(rng.choice(len(h), size=min(len(h), c['states_per_batch']), replace=False), device=device)].cpu().numpy())
    hidden = np.concatenate(states).astype(np.float32)
    np.savez_compressed(out / 'natural_states.npz', hidden=hidden)
    quality = {}
    for seed, s in saes.items():
        h = torch.tensor(hidden[:4096], device=device)
        pre = torch.relu((h - s['b_dec']) @ s['encoder.weight'].T + s['encoder.bias'])
        top = pre.topk(int(s['k']), dim=-1)
        z = torch.zeros_like(pre).scatter(-1, top.indices, top.values)
        rec = z @ s['decoder.weight'].T + s['b_dec']
        quality[seed] = float(1 - (rec - h).square().sum() / (h - h.mean(0)).square().sum())
    selection = [r for r in panels['fit'] if r['split'] == 'source_selection']
    gates = {}
    for seed, s in saes.items():
        contributions = {t: [] for t in TASKS}
        for r in selection:
            grads = []
            for key in ['good', 'bad']:
                seq = torch.tensor([r[key]], device=device)
                hidden_box = {}

                def grab(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h.retain_grad()
                    hidden_box['h'] = h
                    return None

                hh = layer.register_forward_hook(grab)
                with torch.enable_grad():
                    emb = model.get_input_embeddings()(seq).detach().requires_grad_(True)
                    logits = model(inputs_embeds=emb, use_cache=False).logits[0, :-1]
                    logits.log_softmax(-1).gather(-1, seq[0, 1:, None]).sum().backward()
                hh.remove()
                grads.append(hidden_box['h'].grad[0, r['position']].detach())
                x = hidden_box['h'][0, r['position']].detach()
            # The shared prefix state receives the derivative of the good-minus-bad sentence log probability.
            gradient = grads[0] - grads[1]
            pre = torch.relu((x - s['b_dec']) @ s['encoder.weight'].T + s['encoder.bias'])
            top = pre.topk(int(s['k']))
            z = torch.zeros_like(pre).scatter(-1, top.indices, top.values)
            contributions[r['task']].append((z * (gradient @ s['decoder.weight'])).cpu().numpy())
        mean = np.stack([np.mean(contributions[t], axis=0) for t in TASKS])
        score = np.stack([mean[i] - np.mean([np.maximum(mean[j], 0) for j in range(3) if j != i], axis=0) for i in range(3)])
        winner = score.argmax(0)
        gate = np.zeros((score.shape[1], 3), dtype=np.float32)
        for i in range(3):
            candidates = np.flatnonzero((winner == i) & (score[i] > 0))
            chosen = candidates[np.argsort(-score[i, candidates], kind='stable')[:c['members_per_group']]]
            assert len(chosen) == c['members_per_group'], (seed, i, len(candidates))
            gate[chosen, i] = 1.
        np.savez_compressed(out / f'topk_s{seed}_source.npz', gate=gate)
        gates[seed] = dict(score_of_selected=[float(score[i, gate[:, i] > 0].mean()) for i in range(3)])
    handle.remove()
    record = dict(status='PASS', written_utc=datetime.now(timezone.utc).isoformat(), config=c,
                  script_sha256=sha256(Path(__file__)), natural_states=int(len(hidden)), fve=quality, gates=gates,
                  panels={k: len(v) for k, v in panels.items()})
    (out / 'PREPARE.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
